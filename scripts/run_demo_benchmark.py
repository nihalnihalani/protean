#!/usr/bin/env python3
"""Generate the guaranteed base-vs-hand-optimized demo artifact.

Single-op mode (default): grade one op's hand-optimized kernel across the train
and held-out shapes.

Combined mode (``--all-ops``): grade every public op's seed kernel over both
splits and emit one combined report. This is CPU-safe -- on a machine without
CUDA the verifier records ``cuda_unavailable`` rows rather than failing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.grader import grade_source
from protean.kernels import (
    HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
    HAND_OPTIMIZED_RMSNORM,
    SEED_KERNELS,
)
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


KERNELS = {
    "elementwise_add_relu": HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
    "rmsnorm": HAND_OPTIMIZED_RMSNORM,
}

# Every public op, each mapped to its best-known hand-optimized seed kernel.
ALL_OP_KERNELS = dict(SEED_KERNELS)


def _grade_rows(kernels: dict[str, str], reps: int, warmup: int) -> list[dict]:
    """Grade each op's kernel over both splits and all shapes."""
    rows: list[dict] = []
    for op in sorted(kernels):
        src = kernels[op]
        for split, shapes in (("train", TRAIN_SHAPES), ("held_out", HELD_OUT_SHAPES)):
            for shape in shapes:
                rows.append(
                    grade_source(
                        src,
                        op=op,
                        split=split,
                        shape=shape,
                        reps=reps,
                        warmup=warmup,
                    )
                )
    return rows


def _row_markdown(row: dict, *, with_op: bool) -> str:
    op_cell = f"| {row.get('op', '')} " if with_op else ""
    return (
        f"{op_cell}| {row['split']} | {row['shape']} | {row['correct']} | "
        f"{row['speedup']:.3f}x | {row['t_eager_ms']} | {row['t_kernel_ms']} | "
        f"{row['reward']} | {', '.join(row['caps']) or '-'} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="demo/protean-demo-results.json")
    parser.add_argument("--md-out", default="demo/protean-demo-results.md")
    parser.add_argument("--op", choices=sorted(KERNELS), default="elementwise_add_relu")
    parser.add_argument(
        "--all-ops",
        action="store_true",
        help="Grade every public op's seed kernel and emit one combined report.",
    )
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.all_ops:
        kernels = ALL_OP_KERNELS
        title = "# Protean Combined Demo Benchmark"
        subtitle = (
            "All public ops. Base is PyTorch eager. Candidate is each op's "
            "hand-optimized Triton seed kernel."
        )
        with_op = True
    else:
        kernels = {args.op: KERNELS[args.op]}
        title = "# Protean Demo Benchmark"
        subtitle = (
            f"Op: `{args.op}`. Base is PyTorch eager. "
            "Candidate is the hand-optimized Triton kernel."
        )
        with_op = False

    rows = _grade_rows(kernels, reps=args.reps, warmup=args.warmup)

    json_path = ROOT / args.json_out
    md_path = ROOT / args.md_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    if with_op:
        header = "| Op | Split | Shape | Correct | Speedup | Eager ms | Triton ms | Reward | Caps |"
        sep = "|---|---|---:|---|---:|---:|---:|---:|---|"
    else:
        header = "| Split | Shape | Correct | Speedup | Eager ms | Triton ms | Reward | Caps |"
        sep = "|---|---:|---|---:|---:|---:|---:|---|"

    md = [
        title,
        "",
        subtitle,
        "",
        header,
        sep,
        *[_row_markdown(row, with_op=with_op) for row in rows],
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
