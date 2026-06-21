#!/usr/bin/env python3
"""Generate the guaranteed base-vs-hand-optimized demo artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


def _row_markdown(row: dict) -> str:
    return (
        f"| {row['split']} | {row['shape']} | {row['correct']} | "
        f"{row['speedup']:.3f}x | {row['t_eager_ms']} | {row['t_kernel_ms']} | "
        f"{row['reward']} | {', '.join(row['caps']) or '-'} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="demo/protean-demo-results.json")
    parser.add_argument("--md-out", default="demo/protean-demo-results.md")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for split, shapes in (("train", TRAIN_SHAPES), ("held_out", HELD_OUT_SHAPES)):
        for shape in shapes:
            rows.append(
                grade_source(
                    HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
                    split=split,
                    shape=shape,
                    reps=args.reps,
                    warmup=args.warmup,
                )
            )

    json_path = ROOT / args.json_out
    md_path = ROOT / args.md_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    md = [
        "# Protean Demo Benchmark",
        "",
        "Base is PyTorch eager `relu(x + y)`. Candidate is the hand-optimized Triton kernel.",
        "",
        "| Split | Shape | Correct | Speedup | Eager ms | Triton ms | Reward | Caps |",
        "|---|---:|---|---:|---:|---:|---:|---|",
        *[_row_markdown(row) for row in rows],
        "",
    ]
    md_path.write_text("\n".join(md))
    print(md_path)
    print(json_path)

    if any("cuda_unavailable" in row["caps"] for row in rows):
        print("CUDA unavailable; benchmark artifacts record verifier unavailability.", file=sys.stderr)
        return 2
    held_out_ok = any(row["split"] == "held_out" and row["correct"] and row["speedup"] > 1.0 for row in rows)
    return 0 if held_out_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
