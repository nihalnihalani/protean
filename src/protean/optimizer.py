"""Iterative kernel-improvement loop.

This is the first lean version of Protean's overnight goal: start from the
current best kernel, edit it, grade it, keep improvements, and log everything.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU, seed_kernel_for
from protean.model.policy import (
    config_action_space,
    effective_action_space,
    learned_kernel_edits,
    local_kernel_edits,
    make_config_edit,
)
from protean.model.rl_layer import accept_candidate, score, score_delta
from protean.model.tiny_policy import ACTION_BLOCK_SIZES, TinyPolicyHead, state_features
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


class BanditSearch:
    """UCB1 bandit over the joint launch-config space (KernelBand, arXiv:2511.18868).

    Each arm is a ``(block_size, num_warps, num_stages)`` tuple. Instead of the
    flat deterministic sweep that re-evaluates every block size each round, the
    bandit concentrates trials on empirically strong arms while keeping enough
    exploration to discover new winners. Pure Python; no CUDA dependency.
    """

    def __init__(
        self,
        arms: list[tuple[int, int, int]] | None = None,
        *,
        c: float = math.sqrt(2.0),
        seed: int = 0,
    ) -> None:
        self.arms = list(arms) if arms is not None else config_action_space()
        if not self.arms:
            raise ValueError("BanditSearch requires at least one arm")
        self.c = float(c)
        self._rng = random.Random(seed)
        self.counts: dict[tuple[int, int, int], int] = {arm: 0 for arm in self.arms}
        self.values: dict[tuple[int, int, int], float] = {arm: 0.0 for arm in self.arms}
        self.total = 0

    def select(self) -> tuple[int, int, int]:
        """Return the next arm to try via UCB1.

        Unvisited arms get priority (infinite UCB); ties are broken randomly so
        the search does not deterministically favour the enumeration order.
        """

        unvisited = [arm for arm in self.arms if self.counts[arm] == 0]
        if unvisited:
            return self._rng.choice(unvisited)

        log_total = math.log(self.total)
        best_arm = self.arms[0]
        best_index = -math.inf
        order = list(self.arms)
        self._rng.shuffle(order)
        for arm in order:
            n = self.counts[arm]
            ucb = self.values[arm] + self.c * math.sqrt(log_total / n)
            if ucb > best_index:
                best_index = ucb
                best_arm = arm
        return best_arm

    def update(self, arm: tuple[int, int, int], reward: float) -> None:
        """Incremental-mean update of the chosen arm's value estimate."""

        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        prev = self.values[arm]
        self.values[arm] = prev + (float(reward) - prev) / n

    def best_arm(self) -> tuple[int, int, int]:
        """Arm with the highest empirical mean among those visited."""

        visited = [arm for arm in self.arms if self.counts[arm] > 0]
        pool = visited or self.arms
        return max(pool, key=lambda arm: self.values[arm])

    def state(self) -> dict:
        """JSON-serialisable snapshot for the trial trace."""

        return {
            "c": self.c,
            "total": self.total,
            "arms": [
                {
                    "block_size": arm[0],
                    "num_warps": arm[1],
                    "num_stages": arm[2],
                    "count": self.counts[arm],
                    "value": round(self.values[arm], 6),
                }
                for arm in self.arms
            ],
        }


def evaluate_kernel(source: str, *, op: str = "elementwise_add_relu", reps: int, warmup: int) -> dict:
    rows = []
    for split, shapes in (("train", TRAIN_SHAPES), ("held_out", HELD_OUT_SHAPES)):
        for shape in shapes:
            rows.append(grade_source(source, op=op, split=split, shape=shape, reps=reps, warmup=warmup))

    held_out = [row for row in rows if row["split"] == "held_out"]
    correct_held_out = [row for row in held_out if row["correct"] and not row["caps"]]
    mean_held_out_speedup = (
        sum(row["speedup"] for row in correct_held_out) / len(correct_held_out) if correct_held_out else 0.0
    )
    mean_reward = sum(row["reward"] for row in rows) / len(rows)
    return {
        "rows": rows,
        "correct_held_out": len(correct_held_out),
        "mean_held_out_speedup": round(mean_held_out_speedup, 6),
        "mean_reward": round(mean_reward, 6),
    }


def bandit_reward(summary: dict) -> float:
    """Scalar bandit feedback derived from a candidate evaluation summary.

    Mirrors the ranking in :func:`protean.model.rl_layer.score` (held-out speed
    first, then mean reward) collapsed into a single value the UCB estimator can
    average. Failed evaluations score 0.0.
    """

    return float(summary.get("mean_held_out_speedup", 0.0)) + 0.1 * float(summary.get("mean_reward", 0.0))


def failed_evaluation_summary(exc: Exception) -> dict:
    """Represent verifier/runtime failures as rejected trace rows."""

    return {
        "rows": [],
        "correct_held_out": 0,
        "mean_held_out_speedup": 0.0,
        "mean_reward": 0.0,
        "eval_error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def controller_decision(summary: dict, controller_path: str | Path | None) -> dict | None:
    """Return the 1M policy-head decision for trace/HUD metadata if available."""

    if controller_path is None:
        return None
    path = Path(controller_path)
    if not path.exists():
        return {
            "controller": str(path),
            "available": False,
            "reason": "controller file not found",
        }
    policy = TinyPolicyHead.load(path)
    ranked = policy.ranked_actions(state_features(summary))
    return {
        "controller": str(path),
        "available": True,
        "parameter_count": policy.parameter_count,
        "ranked_actions": [
            {
                "action": int(action),
                "edit": f"block_size_{ACTION_BLOCK_SIZES[action]}",
                "block_size": ACTION_BLOCK_SIZES[action],
            }
            for action in ranked
        ],
    }


def run_optimization(
    *,
    out_dir: str | Path = "runs/protean-overnight",
    max_rounds: int = 1,
    seed_source: str | None = None,
    policy_path: str | Path | None = None,
    controller_path: str | Path | None = None,
    edit_policy: str = "local",
    op: str = "elementwise_add_relu",
    fireworks_model: str | None = None,
    stream_hud: bool = False,
    hud_env_source: str | Path = "src/protean/env.py",
    hud_timeout: float = 180.0,
    hud_job_name: str | None = None,
    hud_group: int = 1,
    hud_session=None,
    powered_eval: bool = False,
    powered_eval_n_per_op: int = 40,
    bandit_c: float = math.sqrt(2.0),
    bandit_seed: int = 0,
    bandit_patience: int = 0,
    bandit_arms: list[tuple[int, int, int]] | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    log_path = out / "trials.jsonl"
    best_path = out / f"best_kernel_{op}.py"
    summary_path = out / f"summary_{op}.json"

    if seed_source is None:
        seed_source = seed_kernel_for(op)

    best_source = seed_source
    started = time.time()
    best_summary = evaluate_kernel(best_source, op=op, reps=30, warmup=8)
    best_score = score(best_summary)
    best_path.write_text(best_source)
    seed_path = candidate_dir / "0000_seed.py"
    seed_path.write_text(seed_source)
    trial_count = 0
    accepted_count = 0
    total_model_cost_usd = 0.0
    total_tokens = 0
    pricing_misses = 0
    if edit_policy == "bandit":
        # Prune the arm space to axes that actually change this op's kernel, so
        # logged arm counts reflect distinct candidates (block_size is inert for
        # rmsnorm/softmax_rows -- see policy.effective_action_space).
        arms = bandit_arms if bandit_arms is not None else effective_action_space(best_source)
        bandit = BanditSearch(arms=arms, c=bandit_c, seed=bandit_seed)
    else:
        bandit = None
    rounds_since_improvement = 0
    stopped_early = False
    rounds_run = 0
    if stream_hud and hud_session is None:
        from protean.hud_stream import start_hud_stream_session

        hud_session = start_hud_stream_session(
            name=hud_job_name or f"protean-{edit_policy}-{op}-{int(started)}",
            group=hud_group,
        )

    with log_path.open("a") as log:
        log.write(
            json.dumps(
                {
                    "event": "seed",
                    "time": time.time(),
                    "elapsed_sec": round(time.time() - started, 6),
                    "policy": "seed",
                    "source_path": str(seed_path),
                    "model_cost_usd": 0.0,
                    "tokens": 0,
                    "score": best_score,
                    "summary": best_summary,
                },
                sort_keys=True,
            )
            + "\n"
        )

        for round_idx in range(max_rounds):
            rounds_run += 1
            trial_controller_decision = controller_decision(best_summary, controller_path)
            selected_arm = None
            if edit_policy == "fireworks":
                from protean.model.fireworks_policy import DEFAULT_FIREWORKS_MODEL, fireworks_kernel_edit

                edits = [
                    fireworks_kernel_edit(
                        op=op,
                        current_best=best_source,
                        best_summary=best_summary,
                        model=fireworks_model or DEFAULT_FIREWORKS_MODEL,
                    )
                ]
            elif edit_policy == "bandit":
                # UCB1 selects a single arm in the joint launch-config space.
                selected_arm = bandit.select()
                edits = [make_config_edit(best_source, *selected_arm)]
            elif policy_path is not None or edit_policy == "learned":
                if policy_path is None:
                    raise ValueError("policy_path is required when edit_policy='learned'")
                edits = learned_kernel_edits(best_source, best_summary, str(policy_path))
            else:
                edits = local_kernel_edits(best_source)
            round_improved = False
            for edit in edits:
                trial_count += 1
                total_model_cost_usd += float(edit.model_cost_usd)
                total_tokens += int(edit.tokens)
                edit_pricing_miss = bool(getattr(edit, "pricing_miss", False))
                if edit_pricing_miss:
                    pricing_misses += 1
                candidate_path = candidate_dir / f"{trial_count:04d}_{edit.name}.py"
                candidate_path.write_text(edit.source)
                before_score = best_score
                before_summary = best_summary
                eval_error = None
                try:
                    candidate_summary = evaluate_kernel(
                        edit.source,
                        op=op,
                        reps=int(edit.harness["reps"]),
                        warmup=int(edit.harness["warmup"]),
                    )
                except Exception as exc:  # noqa: BLE001 - model kernels can fail in many ways.
                    eval_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    candidate_summary = failed_evaluation_summary(exc)
                candidate_score = score(candidate_summary)
                accepted = eval_error is None and accept_candidate(candidate_score, best_score)
                if accepted:
                    accepted_count += 1
                    round_improved = True
                    best_source = edit.source
                    best_summary = candidate_summary
                    best_score = candidate_score
                    best_path.write_text(best_source)

                # Feed the bandit its reward regardless of acceptance so UCB
                # learns the true value of every arm it pulls.
                bandit_arm_info = None
                if bandit is not None and selected_arm is not None:
                    reward_signal = 0.0 if eval_error is not None else bandit_reward(candidate_summary)
                    bandit.update(selected_arm, reward_signal)
                    bandit_arm_info = {
                        "block_size": selected_arm[0],
                        "num_warps": selected_arm[1],
                        "num_stages": selected_arm[2],
                        "reward": round(reward_signal, 6),
                        "count": bandit.counts[selected_arm],
                        "value": round(bandit.values[selected_arm], 6),
                    }

                hud_stream = None
                hud_stream_error = None
                if stream_hud:
                    try:
                        from protean.hud_stream import stream_candidate_to_hud

                        hud_stream = stream_candidate_to_hud(
                            source=edit.source,
                            op=op,
                            trial=trial_count,
                            edit=edit.name,
                            accepted=accepted,
                            policy=edit.policy,
                            reason=edit.reason,
                            tokens=edit.tokens,
                            model_cost_usd=edit.model_cost_usd,
                            controller_decision=trial_controller_decision,
                            source_path=str(candidate_path),
                            summary=candidate_summary,
                            eval_error=eval_error,
                            env_source=hud_env_source,
                            timeout=hud_timeout,
                            session=hud_session,
                            group=hud_group,
                        )
                    except Exception as exc:  # noqa: BLE001 - streaming should not kill optimization.
                        hud_stream_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }

                log.write(
                    json.dumps(
                        {
                            "event": "trial",
                            "time": time.time(),
                            "elapsed_sec": round(time.time() - started, 6),
                            "trial": trial_count,
                            "round": round_idx,
                            "op": op,
                            "edit": edit.name,
                            "reason": edit.reason,
                            "policy": edit.policy,
                            "harness": edit.harness,
                            "source_path": str(candidate_path),
                            "model_cost_usd": edit.model_cost_usd,
                            "tokens": edit.tokens,
                            "pricing_miss": edit_pricing_miss,
                            "controller_decision": trial_controller_decision,
                            "best_score_before": before_score,
                            "best_summary_before": before_summary,
                            "score": candidate_score,
                            "delta_vs_best": score_delta(candidate_score, before_score),
                            "accepted": accepted,
                            "eval_error": eval_error,
                            "hud_stream": hud_stream,
                            "hud_stream_error": hud_stream_error,
                            "bandit_arm": bandit_arm_info,
                            "summary": candidate_summary,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            # Anytime stopping: bail out once the search fails to improve the
            # best kernel for `bandit_patience` consecutive rounds. This is a
            # bandit-only feature -- it is gated on `bandit is not None` so a
            # caller that sets `bandit_patience` while running a non-bandit edit
            # policy (local/learned/fireworks) is never silently truncated. The
            # patience counter only accumulates once UCB has finished its
            # mandatory initial exploration (every arm visited at least once):
            # stagnation during forced exploration is expected, so it must not
            # abort the search while unexplored arms could still win.
            exploration_done = bandit is None or bandit.total >= len(bandit.arms)
            if round_improved or not exploration_done:
                rounds_since_improvement = 0
            else:
                rounds_since_improvement += 1
            if (
                bandit is not None
                and bandit_patience > 0
                and exploration_done
                and rounds_since_improvement >= bandit_patience
            ):
                stopped_early = True
                break

    final = {
        "best_score": best_score,
        "best_summary": best_summary,
        "best_kernel": str(best_path),
        "log": str(log_path),
        "trials": trial_count,
        "accepted": accepted_count,
        "total_model_cost_usd": round(total_model_cost_usd, 8),
        "total_tokens": total_tokens,
        "pricing_misses": pricing_misses,
        "elapsed_sec": round(time.time() - started, 6),
        "policy_path": str(policy_path) if policy_path is not None else None,
        "controller_path": str(controller_path) if controller_path is not None else None,
        "edit_policy": edit_policy,
        "op": op,
        "stream_hud": stream_hud,
        "hud_job_url": hud_session.job_url if hud_session is not None else None,
        "hud_job_name": hud_session.name if hud_session is not None else None,
        "hud_group": hud_group if stream_hud else None,
        "rounds_run": rounds_run,
        "stopped_early": stopped_early,
        "bandit": bandit.state() if bandit is not None else None,
        "bandit_best_arm": (
            {
                "block_size": bandit.best_arm()[0],
                "num_warps": bandit.best_arm()[1],
                "num_stages": bandit.best_arm()[2],
            }
            if bandit is not None
            else None
        ),
    }

    # Optional final reporting step: powered base(seed)-vs-trained(best) held-out eval (GPU).
    # Off by default so the CPU test suite never touches the grader's CUDA path.
    if powered_eval:
        from protean.eval_protocol import powered_eval_from_run_dir

        powered_path = out / "powered-eval.json"
        try:
            powered = powered_eval_from_run_dir(out, [op], n_per_op=powered_eval_n_per_op)
        except Exception as exc:  # noqa: BLE001 - reporting must not fail the run
            powered = {"powered_eval_error": {"type": type(exc).__name__, "message": str(exc)}}
        powered_path.write_text(json.dumps(powered, indent=2, sort_keys=True, default=str) + "\n")
        final["powered_eval"] = str(powered_path)

    summary_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final
