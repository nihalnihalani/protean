#!/usr/bin/env python3
"""Run HUD with a deterministic Protean demo agent.

This is the dashboard proof path: the agent submits Protean's known-good
hand-optimized kernels, while HUD still starts tasks and calls the real grader.
The resulting job appears on hud.ai with non-zero rewards.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mcp.types as mcp_types
from hud.agents.base import Agent
from hud.eval import LocalRuntime, Taskset
from hud.types import Step
from hud.utils.time import now_iso

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


def _message(role: str, text: str) -> mcp_types.PromptMessage:
    return mcp_types.PromptMessage(
        role=role,
        content=mcp_types.TextContent(type="text", text=text),
    )


def _emit_candidate_file_artifacts(op: str, source: str) -> None:
    """Send a minimal file snapshot/diff so HUD's Code tab has source to show."""

    from hud.telemetry.filetracking import emit_file_diff, emit_file_snapshot

    started_at = now_iso()
    encoded = source.encode("utf-8")
    path = f"candidates/{op}/solution.py"
    emit_file_snapshot(
        {
            "files_scanned": 0,
            "files": [],
        },
        started_at=started_at,
    )
    emit_file_diff(
        {
            "snapshot_timestamp": time.time(),
            "scan_duration_ms": 0.0,
            "files_scanned": 1,
            "files_changed": 1,
            "patches": [
                {
                    "path": path,
                    "status": "added",
                    "patch": (
                        f"--- /dev/null\n+++ b/{path}\n"
                        + "".join(f"+{line}\n" for line in source.splitlines())
                    ),
                    "size_before": 0,
                    "size_after": len(encoded),
                }
            ],
        },
        started_at=started_at,
    )


class ProteanDemoAgent(Agent):
    """Submit the correct hand-optimized kernel for the prompted op."""

    async def __call__(self, run) -> None:
        prompt = run.prompt_text or ""
        op = _op_from_prompt(prompt)
        answer = seed_kernel_for(op)
        source_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()

        run.record(
            Step(
                source="agent",
                messages=[
                    _message(
                        "assistant",
                        (
                            "I will submit Protean's known-good hand-optimized "
                            f"Triton seed kernel for op={op} and let the verifier grade it."
                        ),
                    )
                ],
                extra={"event": "model_decision", "op": op},
            )
        )
        run.record(
            Step(
                source="tool",
                messages=[_message("assistant", answer)],
                extra={
                    "event": "candidate_source",
                    "path": f"candidates/{op}/solution.py",
                    "sha256": source_hash,
                    "bytes": len(answer.encode("utf-8")),
                    "op": op,
                },
            )
        )
        _emit_candidate_file_artifacts(op, answer)
        run.record(
            Step(
                source="tool",
                messages=[
                    _message(
                        "assistant",
                        (
                            "Candidate source emitted to HUD filetracking. "
                            "Next step: Protean grader checks AST anti-hack rules, "
                            "Triton launch, correctness, timing, and reward."
                        ),
                    )
                ],
                extra={"event": "verifier_preflight", "op": op, "sha256": source_hash},
            )
        )

        run.trace.content = answer
        run.trace.extra["agent"] = "protean_demo_agent"
        run.trace.extra["op"] = op
        run.trace.extra["candidate_sha256"] = source_hash
        run.record(
            Step(
                source="agent",
                messages=[_message("assistant", "Submitting solution.py to the HUD task grader.")],
                extra={"event": "submit", "op": op, "answer_bytes": len(answer)},
            )
        )


async def _run(args: argparse.Namespace) -> int:
    taskset = Taskset.from_api(args.taskset) if args.taskset else Taskset.from_file(args.source)
    if args.task_ids:
        taskset = taskset.filter(slug.strip() for slug in args.task_ids.split(",") if slug.strip())
    job = await taskset.run(
        ProteanDemoAgent(),
        runtime=LocalRuntime(args.source),
        group=args.group,
        max_concurrent=args.max_concurrent,
        rollout_timeout=args.rollout_timeout,
    )
    try:
        from hud.telemetry.exporter import flush

        flush(timeout=20.0)
    except Exception:
        pass

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
    parser.add_argument("--task-ids", help="Comma-separated task slugs to run from the taskset.")
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
