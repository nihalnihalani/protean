"""HUD streaming helpers for optimizer trials."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


class HudStreamError(RuntimeError):
    """Raised when a trial cannot be streamed to HUD."""


def _task_slugs_for_op(op: str) -> list[str]:
    return [f"{op}_train", f"{op}_held_out"]


async def _stream_candidate_async(
    *,
    source: str,
    op: str,
    trial: int,
    edit: str,
    accepted: bool,
    env_source: str,
    timeout: float,
) -> dict[str, Any]:
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime, Taskset
    from hud.types import Step

    class CandidateAgent(Agent):
        async def __call__(self, run) -> None:
            run.trace.content = source
            run.trace.extra["agent"] = "protean_optimizer_stream"
            run.trace.extra["op"] = op
            run.trace.extra["trial"] = trial
            run.trace.extra["edit"] = edit
            run.trace.extra["accepted"] = accepted
            run.record(
                Step(
                    source="agent",
                    extra={
                        "op": op,
                        "trial": trial,
                        "edit": edit,
                        "accepted": accepted,
                        "answer_bytes": len(source),
                    },
                )
            )

    taskset = Taskset.from_file(env_source).filter(_task_slugs_for_op(op))
    job = await taskset.run(
        CandidateAgent(),
        runtime=LocalRuntime(env_source),
        max_concurrent=1,
        rollout_timeout=timeout,
    )
    rows = []
    for slug, runs in job.results.items():
        for run in runs:
            info = run.grade.info if isinstance(run.grade.info, dict) else {}
            hud_info = info.get("hud", {}) if isinstance(info.get("hud"), dict) else {}
            rows.append(
                {
                    "slug": slug,
                    "reward": run.reward,
                    "status": run.trace.status,
                    "op": hud_info.get("op") or info.get("op"),
                    "split": hud_info.get("split") or info.get("split"),
                    "shape": hud_info.get("shape") or info.get("shape"),
                    "correct": hud_info.get("correct") if "correct" in hud_info else info.get("correct"),
                    "speedup": hud_info.get("speedup") or info.get("speedup"),
                    "caps": hud_info.get("caps") or info.get("caps"),
                }
            )
    return {
        "job_id": job.id,
        "job_url": f"https://hud.ai/jobs/{job.id}",
        "mean_reward": job.reward,
        "rows": rows,
    }


def stream_candidate_to_hud(
    *,
    source: str,
    op: str,
    trial: int,
    edit: str,
    accepted: bool,
    env_source: str | Path = "src/protean/env.py",
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Create a HUD eval job for a single optimizer trial candidate."""

    if not os.environ.get("HUD_API_KEY"):
        raise HudStreamError("HUD_API_KEY is required to stream optimizer trials to HUD")
    env_path = Path(env_source)
    if not env_path.exists():
        raise HudStreamError(f"HUD env source does not exist: {env_path}")
    return asyncio.run(
        _stream_candidate_async(
            source=source,
            op=op,
            trial=trial,
            edit=edit,
            accepted=accepted,
            env_source=str(env_path),
            timeout=timeout,
        )
    )
