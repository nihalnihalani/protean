"""Iterative kernel-improvement loop.

This is the first lean version of Protean's overnight goal: start from the
current best kernel, edit it, grade it, keep improvements, and log everything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.model.policy import learned_kernel_edits, local_kernel_edits
from protean.model.rl_layer import accept_candidate, score, score_delta
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


def evaluate_kernel(source: str, *, reps: int, warmup: int) -> dict:
    rows = []
    for split, shapes in (("train", TRAIN_SHAPES), ("held_out", HELD_OUT_SHAPES)):
        for shape in shapes:
            rows.append(grade_source(source, split=split, shape=shape, reps=reps, warmup=warmup))

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


def run_optimization(
    *,
    out_dir: str | Path = "runs/protean-overnight",
    max_rounds: int = 1,
    seed_source: str = HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
    policy_path: str | Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    log_path = out / "trials.jsonl"
    best_path = out / "best_kernel.py"
    summary_path = out / "summary.json"

    best_source = seed_source
    started = time.time()
    best_summary = evaluate_kernel(best_source, reps=30, warmup=8)
    best_score = score(best_summary)
    best_path.write_text(best_source)
    seed_path = candidate_dir / "0000_seed.py"
    seed_path.write_text(seed_source)
    trial_count = 0
    accepted_count = 0

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
            edits = (
                learned_kernel_edits(best_source, best_summary, str(policy_path))
                if policy_path is not None
                else local_kernel_edits(best_source)
            )
            for edit in edits:
                trial_count += 1
                candidate_path = candidate_dir / f"{trial_count:04d}_{edit.name}.py"
                candidate_path.write_text(edit.source)
                before_score = best_score
                before_summary = best_summary
                candidate_summary = evaluate_kernel(
                    edit.source,
                    reps=int(edit.harness["reps"]),
                    warmup=int(edit.harness["warmup"]),
                )
                candidate_score = score(candidate_summary)
                accepted = accept_candidate(candidate_score, best_score)
                if accepted:
                    accepted_count += 1
                    best_source = edit.source
                    best_summary = candidate_summary
                    best_score = candidate_score
                    best_path.write_text(best_source)

                log.write(
                    json.dumps(
                        {
                            "event": "trial",
                            "time": time.time(),
                            "elapsed_sec": round(time.time() - started, 6),
                            "round": round_idx,
                            "edit": edit.name,
                            "reason": edit.reason,
                            "policy": edit.policy,
                            "harness": edit.harness,
                            "source_path": str(candidate_path),
                            "model_cost_usd": edit.model_cost_usd,
                            "tokens": edit.tokens,
                            "best_score_before": before_score,
                            "best_summary_before": before_summary,
                            "score": candidate_score,
                            "delta_vs_best": score_delta(candidate_score, before_score),
                            "accepted": accepted,
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
    }
    summary_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final
