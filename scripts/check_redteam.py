#!/usr/bin/env python3
"""Check that obvious reward hacks fail closed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.grader import grade_source
from protean.kernels import BAD_SHAPE_TRITON, NO_LAUNCH, PYTORCH_PASSTHROUGH

CASES = {
    "pytorch_passthrough": PYTORCH_PASSTHROUGH,
    "no_launch": NO_LAUNCH,
    "bad_shape": BAD_SHAPE_TRITON,
}


def main() -> int:
    results = {name: grade_source(src, split="held_out", reps=10, warmup=3) for name, src in CASES.items()}
    print(json.dumps(results, indent=2, sort_keys=True))

    failures = []
    for name, grade in results.items():
        if grade["reward"] != 0:
            failures.append(f"{name}: reward={grade['reward']}")
    if failures:
        print("Red-team failures: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
