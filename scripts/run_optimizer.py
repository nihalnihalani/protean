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
from protean.task_catalog import OPS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/protean-overnight")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--op", default="elementwise_add_relu")
    parser.add_argument("--all-ops", action="store_true",
                        help="Run the optimizer on every registered op.")
    parser.add_argument("--edit-policy", choices=["local", "learned", "fireworks"], default="local")
    parser.add_argument("--policy-path", help="Optional tiny policy JSON used to order kernel edits.")
    parser.add_argument("--fireworks-model", default=None)
    parser.add_argument("--stream-hud", action="store_true",
                        help="Create a HUD eval job for every optimizer trial candidate.")
    parser.add_argument("--hud-env-source", default="src/protean/env.py")
    parser.add_argument("--hud-timeout", type=float, default=180.0)
    args = parser.parse_args()

    ops_to_run = [op.name for op in OPS] if args.all_ops else [args.op]
    results = {}

    for op in ops_to_run:
        result = run_optimization(
            out_dir=ROOT / args.out_dir,
            max_rounds=args.max_rounds,
            op=op,
            edit_policy=args.edit_policy,
            policy_path=args.policy_path,
            fireworks_model=args.fireworks_model,
            stream_hud=args.stream_hud,
            hud_env_source=args.hud_env_source,
            hud_timeout=args.hud_timeout,
        )
        results[op] = result

    if len(results) == 1:
        op = list(results.keys())[0]
        print(json.dumps(results[op], indent=2, sort_keys=True))
    else:
        print("=== All Ops Summary ===")
        for op, result in results.items():
            best = result.get("best_score", [0, 0, 0])
            print(f"  {op:30s}  score={best}  accepted={result.get('accepted',0)}/{result.get('trials',0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
