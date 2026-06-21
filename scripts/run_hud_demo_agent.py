#!/usr/bin/env python3
"""Run HUD with a deterministic Protean demo agent.

This is the dashboard proof path: the agent submits Protean's known-good
hand-optimized kernels, while HUD still starts tasks and calls the real grader.
The resulting job appears on hud.ai with non-zero rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hud.agents.base import Agent
from hud.eval import LocalRuntime, Taskset
from hud.types import Step

from protean.kernels import seed_kernel_for
from protean.task_catalog import OPS, OPS_BY_NAME


def _op_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    match = re.search(r"^\s*-\s*op:\s*([a-z0-9_]+)\s*$", lowered, re.MULTILINE)
    if match and match.group(1) in OPS_BY_NAME:
        return match.group(1)
    for spec in sorted(OPS, key=lambda item: len(item.name), reverse=True):
        if spec.name in lowered:
            return spec.name
    return "elementwise_add_relu"


class ProteanDemoAgent(Agent):
    """Submit the correct hand-optimized kernel for the prompted op."""

    async def __call__(self, run) -> None:
        prompt = run.prompt_text or ""
        op = _op_from_prompt(prompt)
        answer = seed_kernel_for(op)
        run.trace.content = answer
        run.trace.extra["agent"] = "protean_demo_agent"
        run.trace.extra["op"] = op
        run.record(Step(source="agent", extra={"op": op, "answer_bytes": len(answer)}))


async def _run(args: argparse.Namespace) -> int:
    taskset = Taskset.from_api(args.taskset) if args.taskset else Taskset.from_file(args.source)
    job = await taskset.run(
        ProteanDemoAgent(),
        runtime=LocalRuntime(args.source),
        group=args.group,
        max_concurrent=args.max_concurrent,
        rollout_timeout=args.rollout_timeout,
    )

    rows = []
    for slug, runs in job.results.items():
        for run in runs:
            info = run.grade.info if isinstance(run.grade.info, dict) else {}
            # The new env.py nests grading under a "hud" key
            hud_info = info.get("hud", {}) if isinstance(info.get("hud"), dict) else {}
            rows.append(
                {
                    "slug": slug,
                    "reward": run.reward,
                    "status": run.trace.status,
                    "op": hud_info.get("op") or info.get("op"),
                    "split": hud_info.get("split") or info.get("split"),
                    "shape": hud_info.get("shape") or info.get("shape"),
                    "correct": hud_info.get("correct") or info.get("correct"),
                    "speedup": hud_info.get("speedup") or info.get("speedup"),
                    "caps": hud_info.get("caps") or info.get("caps"),
                }
            )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job.id,
        "job_url": f"https://hud.ai/jobs/{job.id}",
        "taskset": args.taskset,
        "taskset_id": taskset.api_id,
        "mean_reward": job.reward,
        "runs": rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(payload["job_url"])
    print(f"mean_reward={job.reward:.3f}")
    for row in rows:
        speedup_str = f"{row['speedup']:.2f}x" if row["speedup"] else "—"
        print(
            f"  {row['slug']}: reward={row['reward']:.3f} "
            f"correct={row['correct']} speedup={speedup_str} caps={row['caps'] or []}"
        )
    print(out)
    return 0 if rows and all(row["reward"] > 0.0 for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="src/protean/env.py")
    parser.add_argument("--taskset", help="HUD taskset slug or id to attach this eval job to.")
    parser.add_argument("--out", default="demo/hud-demo-agent-results.json")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--rollout-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY is required so HUD can create the dashboard job.", file=sys.stderr)
        return 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
