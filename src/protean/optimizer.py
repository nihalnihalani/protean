"""Iterative kernel-improvement loop.

This is the first lean version of Protean's overnight goal: start from the
current best kernel, edit it, grade it, keep improvements, and log everything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU, seed_kernel_for
from protean.model.policy import learned_kernel_edits, local_kernel_edits
from protean.model.rl_layer import accept_candidate, score, score_delta
from protean.model.tiny_policy import ACTION_BLOCK_SIZES, TinyPolicyHead, state_features
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


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
            trial_controller_decision = controller_decision(best_summary, controller_path)
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
            elif policy_path is not None or edit_policy == "learned":
                if policy_path is None:
                    raise ValueError("policy_path is required when edit_policy='learned'")
                edits = learned_kernel_edits(best_source, best_summary, str(policy_path))
            else:
                edits = local_kernel_edits(best_source)
            for edit in edits:
                trial_count += 1
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
                    best_source = edit.source
                    best_summary = candidate_summary
                    best_score = candidate_score
                    best_path.write_text(best_source)

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
                            "controller_decision": trial_controller_decision,
                            "best_score_before": before_score,
                            "best_summary_before": before_summary,
                            "score": candidate_score,
                            "delta_vs_best": score_delta(candidate_score, before_score),
                            "accepted": accepted,
                            "eval_error": eval_error,
                            "hud_stream": hud_stream,
                            "hud_stream_error": hud_stream_error,
                            "summary": candidate_summary,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    final = {
        "best_score": best_score,
        "best_summary": best_summary,
        "best_kernel": str(best_path),
        "log": str(log_path),
        "trials": trial_count,
        "accepted": accepted_count,
        "elapsed_sec": round(time.time() - started, 6),
        "policy_path": str(policy_path) if policy_path is not None else None,
        "controller_path": str(controller_path) if controller_path is not None else None,
        "edit_policy": edit_policy,
        "op": op,
        "stream_hud": stream_hud,
        "hud_job_url": hud_session.job_url if hud_session is not None else None,
        "hud_job_name": hud_session.name if hud_session is not None else None,
        "hud_group": hud_group if stream_hud else None,
    }
    summary_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final
