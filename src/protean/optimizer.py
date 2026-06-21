"""Iterative kernel-improvement loop.

This is the first lean version of Protean's overnight goal: start from the
current best kernel, edit it, grade it, keep improvements, and log everything.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


@dataclass(frozen=True)
class CandidateEdit:
    name: str
    reason: str
    source: str
    harness: dict


def _replace_block_size(source: str, block_size: int) -> str:
    source = re.sub(r"triton\.cdiv\(n_elements,\s*\d+\)", f"triton.cdiv(n_elements, {block_size})", source)
    source = re.sub(r"block_size=\d+", f"block_size={block_size}", source)
    return source


def local_kernel_edits(current_best: str) -> Iterable[CandidateEdit]:
    """Small deterministic edit policy.

    This intentionally edits the current best implementation instead of starting
    from scratch. A model-backed policy can replace this function later.
    """

    for block_size in (128, 256, 512, 1024, 2048):
        yield CandidateEdit(
            name=f"block_size_{block_size}",
            reason=f"Retune Triton block size to {block_size}.",
            source=_replace_block_size(current_best, block_size),
            harness={"reps": 30, "warmup": 8},
        )


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


def score(summary: dict) -> tuple[float, float, int]:
    return (
        float(summary["mean_held_out_speedup"]),
        float(summary["mean_reward"]),
        int(summary["correct_held_out"]),
    )


def score_delta(candidate: tuple[float, float, int], baseline: tuple[float, float, int]) -> dict:
    return {
        "held_out_speedup": round(candidate[0] - baseline[0], 6),
        "reward": round(candidate[1] - baseline[1], 6),
        "correct_held_out": candidate[2] - baseline[2],
    }


def run_optimization(
    *,
    out_dir: str | Path = "runs/protean-overnight",
    max_rounds: int = 1,
    seed_source: str = HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
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
            for edit in local_kernel_edits(best_source):
                trial_count += 1
                candidate_path = candidate_dir / f"{trial_count:04d}_{edit.name}.py"
                candidate_path.write_text(edit.source)
                before_score = best_score
                candidate_summary = evaluate_kernel(
                    edit.source,
                    reps=int(edit.harness["reps"]),
                    warmup=int(edit.harness["warmup"]),
                )
                candidate_score = score(candidate_summary)
                accepted = candidate_score > best_score
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
                            "policy": "local_deterministic",
                            "harness": edit.harness,
                            "source_path": str(candidate_path),
                            "model_cost_usd": 0.0,
                            "tokens": 0,
                            "best_score_before": before_score,
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
    }
    summary_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final
