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
                        help="Stream every optimizer trial into one HUD job/session.")
    parser.add_argument("--hud-env-source", default="src/protean/env.py")
    parser.add_argument("--hud-timeout", type=float, default=180.0)
    parser.add_argument("--hud-job-name", default=None,
                        help="HUD job/session name for optimizer trial streaming.")
    parser.add_argument("--hud-group", type=int, default=1,
                        help="Repeat each HUD task this many times per streamed trial.")
    args = parser.parse_args()

    ops_to_run = [op.name for op in OPS] if args.all_ops else [args.op]
    results = {}
    hud_session = None
    if args.stream_hud:
        from protean.hud_stream import start_hud_stream_session

        hud_session = start_hud_stream_session(
            name=args.hud_job_name or f"protean-{args.edit_policy}-optimizer",
            group=args.hud_group,
        )

    for op in ops_to_run:
        op_out_dir = ROOT / args.out_dir
        if args.all_ops:
            op_out_dir = op_out_dir / op
        result = run_optimization(
            out_dir=op_out_dir,
            max_rounds=args.max_rounds,
            op=op,
            edit_policy=args.edit_policy,
            policy_path=args.policy_path,
            fireworks_model=args.fireworks_model,
            stream_hud=args.stream_hud,
            hud_env_source=args.hud_env_source,
            hud_timeout=args.hud_timeout,
            hud_job_name=args.hud_job_name,
            hud_group=args.hud_group,
            hud_session=hud_session,
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
