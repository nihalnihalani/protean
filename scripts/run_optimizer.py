#!/usr/bin/env python3
"""Run Protean's iterative kernel optimizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.optimizer import run_optimization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/protean-overnight")
    parser.add_argument("--max-rounds", type=int, default=1)
    args = parser.parse_args()

    result = run_optimization(out_dir=ROOT / args.out_dir, max_rounds=args.max_rounds)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
