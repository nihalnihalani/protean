#!/usr/bin/env python3
"""Run Protean's optimizer as a HUD live-demo/control-plane agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bootstrap_hud_api_key() -> None:
    """Populate HUD_API_KEY before the HUD SDK creates any settings objects."""

    if os.environ.get("HUD_API_KEY"):
        return
    hud_env = Path.home() / ".hud" / ".env"
    if not hud_env.exists():
        return
    for line in hud_env.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HUD_API_KEY":
            os.environ["HUD_API_KEY"] = value.strip().strip('"').strip("'")
            return


_bootstrap_hud_api_key()

from protean.hud_stream import assert_hud_auth, start_hud_stream_session
from protean.optimizer import run_optimization
from protean.task_catalog import OPS

DEFAULT_ENVIRONMENT_URL = "https://www.hud.ai/environments/32bb1f0c-0737-4a58-8a5e-5c9ec8a2f01b"
DEFAULT_TASKSET_URL = "https://www.hud.ai/tasksets/3f2d2423-72d4-4541-bb18-b78e31151676"


def _ops_to_run(all_ops: bool, op: str | None) -> list[str]:
    if all_ops or op is None:
        return [spec.name for spec in OPS]
    known = {spec.name for spec in OPS}
    if op not in known:
        raise ValueError(f"Unknown op '{op}'. Expected one of: {', '.join(sorted(known))}")
    return [op]


def _preflight(args: argparse.Namespace) -> None:
    assert_hud_auth()
    if args.policy == "fireworks" and not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is required for --policy fireworks")
    if args.policy == "learned":
        if args.controller is None:
            raise RuntimeError("--controller is required for --policy learned")
        if not Path(args.controller).exists():
            raise RuntimeError(f"Learned controller does not exist: {args.controller}")
    env_source = ROOT / args.env_source
    if not env_source.exists():
        raise RuntimeError(f"HUD env source does not exist: {env_source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", default="protean-kernel-optimizer")
    parser.add_argument("--taskset-url", default=DEFAULT_TASKSET_URL)
    parser.add_argument("--environment-url", default=DEFAULT_ENVIRONMENT_URL)
    parser.add_argument("--env-source", default="src/protean/env.py")
    parser.add_argument("--policy", choices=["fireworks", "learned", "local"], default="fireworks")
    parser.add_argument("--controller", default="outputs/policy_head.pt")
    parser.add_argument("--fireworks-model", default=None)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--job-name", default="protean-live-kernel-optimizer")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--out-dir", default="runs/protean-live-kernel-optimizer")
    parser.add_argument("--op", default=None, help="Run one op. Defaults to all ops.")
    parser.add_argument("--all-ops", action="store_true", help="Run every registered op.")
    parser.add_argument("--hud-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.all_ops = args.all_ops or args.op is None

    _preflight(args)
    ops = _ops_to_run(args.all_ops, args.op)
    policy_path = args.controller if args.policy == "learned" else None
    out_root = ROOT / args.out_dir

    session = start_hud_stream_session(name=args.job_name, group=args.group)
    print(f"HUD job: {session.job_url}")
    print(f"HUD taskset: {args.taskset_url} ({args.taskset})")
    print(f"HUD environment: {args.environment_url}")

    results: dict[str, dict] = {}
    for op in ops:
        op_out_dir = out_root / op if len(ops) > 1 else out_root
        results[op] = run_optimization(
            out_dir=op_out_dir,
            max_rounds=args.max_rounds,
            op=op,
            edit_policy=args.policy,
            policy_path=policy_path,
            controller_path=args.controller,
            fireworks_model=args.fireworks_model,
            stream_hud=True,
            hud_env_source=args.env_source,
            hud_timeout=args.hud_timeout,
            hud_job_name=args.job_name,
            hud_group=args.group,
            hud_session=session,
        )

    summary = {
        "environment_url": args.environment_url,
        "taskset": args.taskset,
        "taskset_url": args.taskset_url,
        "job_name": session.name,
        "job_url": session.job_url,
        "group": args.group,
        "policy": args.policy,
        "controller": args.controller,
        "ops": results,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "hud_live_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
