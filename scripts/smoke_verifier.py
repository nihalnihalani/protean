#!/usr/bin/env python3
"""Smoke-test the known-good Triton kernel through the real grader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.grader import grade_source
from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU


def main() -> int:
    grade = grade_source(
        HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
        split="held_out",
        reps=20,
        warmup=5,
    )
    print(json.dumps(grade, indent=2, sort_keys=True))
    if "cuda_unavailable" in grade["caps"]:
        print("CUDA unavailable; smoke test could not run the GPU verifier.", file=sys.stderr)
        return 2
    return 0 if grade["correct"] and grade["reward"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
