"""HUD streaming helpers for optimizer trials."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HudStreamError(RuntimeError):
    """Raised when a trial cannot be streamed to HUD."""


@dataclass
class HudStreamSession:
    """One HUD job that receives many optimizer trial rollouts."""

    job: Any
    name: str
    group: int = 1

    @property
    def job_id(self) -> str:
        return self.job.id

    @property
    def job_url(self) -> str:
        return f"https://hud.ai/jobs/{self.job.id}"


def _task_slugs_for_op(op: str) -> list[str]:
    return [f"{op}_train", f"{op}_held_out"]


def _summary_metrics(summary: dict[str, Any], eval_error: dict[str, Any] | None) -> dict[str, Any]:
    rows = summary.get("rows", []) if isinstance(summary, dict) else []
    correct_rows = [row for row in rows if row.get("correct") and not row.get("caps")]
    held_out_rows = [row for row in rows if row.get("split") == "held_out"]
    held_out_correct = [row for row in held_out_rows if row.get("correct") and not row.get("caps")]
    speedups = [float(row.get("speedup", 0.0)) for row in correct_rows]
    held_out_speedups = [float(row.get("speedup", 0.0)) for row in held_out_correct]
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    caps = sorted({cap for row in rows for cap in row.get("caps", [])})
    return {
        "compile_success": eval_error is None,
        "eval_error": eval_error,
        "rows": len(rows),
        "correct_rows": len(correct_rows),
        "held_out_rows": len(held_out_rows),
        "held_out_correct": len(held_out_correct),
        "mean_speedup": round(sum(speedups) / len(speedups), 6) if speedups else 0.0,
        "mean_held_out_speedup": round(sum(held_out_speedups) / len(held_out_speedups), 6)
        if held_out_speedups
        else 0.0,
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "caps": caps,
    }


def _row_from_run(slug: str, run: Any) -> dict[str, Any]:
    info = run.grade.info if isinstance(run.grade.info, dict) else {}
    hud_info = info.get("hud", {}) if isinstance(info.get("hud"), dict) else {}
    raw_reward = info.get("protean_reward_raw", info.get("reward"))
    trace = getattr(run, "trace", None)
    return {
        "slug": slug,
        "reward": getattr(run, "reward", None),
        "protean_reward_raw": raw_reward,
        "status": getattr(trace, "status", None),
        "trace_id": getattr(trace, "trace_id", None),
        "op": hud_info.get("op") or info.get("op"),
        "split": hud_info.get("split") or info.get("split"),
        "shape": hud_info.get("shape") or info.get("shape"),
        "correct": hud_info.get("correct") if "correct" in hud_info else info.get("correct"),
        "speedup": hud_info.get("speedup") or info.get("speedup"),
        "t_eager_ms": hud_info.get("t_eager_ms") or info.get("t_eager_ms"),
        "t_kernel_ms": hud_info.get("t_kernel_ms") or info.get("t_kernel_ms"),
        "caps": hud_info.get("caps") or info.get("caps"),
    }


async def _start_session_async(name: str, group: int) -> HudStreamSession:
    from hud.eval import Job

    job = await Job.start(name, group=group)
    return HudStreamSession(job=job, name=name, group=group)


def start_hud_stream_session(*, name: str | None = None, group: int = 1) -> HudStreamSession:
    """Create one HUD job for a whole optimizer run."""

    if not os.environ.get("HUD_API_KEY"):
        raise HudStreamError("HUD_API_KEY is required to stream optimizer trials to HUD")
    job_name = name or f"protean-optimizer-{int(time.time())}"
    return asyncio.run(_start_session_async(job_name, group))


async def _stream_candidate_async(
    *,
    source: str,
    op: str,
    trial: int,
    edit: str,
    policy: str,
    reason: str,
    tokens: int,
    model_cost_usd: float,
    source_path: str,
    summary: dict[str, Any],
    eval_error: dict[str, Any] | None,
    accepted: bool,
    env_source: str,
    timeout: float,
    session: HudStreamSession | None,
    group: int,
) -> dict[str, Any]:
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime, Taskset
    from hud.types import Step

    metrics = _summary_metrics(summary, eval_error)

    class CandidateAgent(Agent):
        async def __call__(self, run) -> None:
            run.trace.extra["agent"] = "protean_optimizer_stream"
            run.trace.extra.update(
                {
                    "op": op,
                    "trial": trial,
                    "edit": edit,
                    "policy": policy,
                    "tokens": tokens,
                    "model_cost_usd": model_cost_usd,
                    "accepted": accepted,
                    "source_path": source_path,
                    "compile_success": metrics["compile_success"],
                    "mean_reward": metrics["mean_reward"],
                    "mean_speedup": metrics["mean_speedup"],
                    "mean_held_out_speedup": metrics["mean_held_out_speedup"],
                    "held_out_correct": metrics["held_out_correct"],
                }
            )
            run.record(
                Step(
                    source="system",
                    extra={
                        "event": "model_prompt",
                        "op": op,
                        "trial": trial,
                        "policy": policy,
                    },
                )
            )
            run.record(
                Step(
                    source="agent",
                    extra={
                        "event": "model_response",
                        "edit": edit,
                        "reason": reason,
                        "tokens": tokens,
                        "model_cost_usd": model_cost_usd,
                        "answer_bytes": len(source),
                    },
                )
            )
            run.record(
                Step(
                    source="tool",
                    extra={
                        "event": "candidate_saved",
                        "source_path": source_path,
                        "answer_bytes": len(source),
                    },
                )
            )
            run.record(
                Step(
                    source="tool",
                    extra={
                        "event": "ast_check",
                        "caps": metrics["caps"],
                        "passed": "pytorch_passthrough" not in metrics["caps"]
                        and "no_triton_jit" not in metrics["caps"],
                    },
                )
            )
            run.record(
                Step(
                    source="tool",
                    extra={
                        "event": "compile",
                        "compile_success": metrics["compile_success"],
                        "eval_error": eval_error,
                    },
                )
            )
            run.record(
                Step(
                    source="tool",
                    extra={
                        "event": "correctness",
                        "correct_rows": metrics["correct_rows"],
                        "held_out_correct": metrics["held_out_correct"],
                        "rows": metrics["rows"],
                    },
                )
            )
            run.record(
                Step(
                    source="tool",
                    extra={
                        "event": "timing",
                        "mean_speedup": metrics["mean_speedup"],
                        "mean_held_out_speedup": metrics["mean_held_out_speedup"],
                    },
                )
            )
            run.record(
                Step(
                    source="task",
                    extra={
                        "event": "reward",
                        "mean_reward": metrics["mean_reward"],
                        "accepted": accepted,
                    },
                )
            )
            run.trace.content = source
            run.record(
                Step(
                    source="agent",
                    extra={
                        "event": "accepted" if accepted else "rejected",
                        "op": op,
                        "trial": trial,
                        "edit": edit,
                        "accepted": accepted,
                    },
                )
            )

    taskset = Taskset.from_file(env_source).filter(_task_slugs_for_op(op))
    start_run_count = len(session.job.runs) if session is not None else 0
    job = await taskset.run(
        CandidateAgent(),
        runtime=LocalRuntime(env_source),
        job=session.job if session is not None else None,
        group=group,
        max_concurrent=1,
        rollout_timeout=timeout,
    )
    rows = []
    new_runs = job.runs[start_run_count:] if session is not None else job.runs
    for run in new_runs:
        rows.append(_row_from_run(run.slug or "", run))
    return {
        "job_id": job.id,
        "job_url": f"https://hud.ai/jobs/{job.id}",
        "job_name": job.name,
        "group": job.group,
        "mean_reward": job.reward,
        "trial_metrics": metrics,
        "rows": rows,
    }


def stream_candidate_to_hud(
    *,
    source: str,
    op: str,
    trial: int,
    edit: str,
    accepted: bool,
    policy: str = "unknown",
    reason: str = "",
    tokens: int = 0,
    model_cost_usd: float = 0.0,
    source_path: str = "",
    summary: dict[str, Any] | None = None,
    eval_error: dict[str, Any] | None = None,
    env_source: str | Path = "src/protean/env.py",
    timeout: float = 180.0,
    session: HudStreamSession | None = None,
    group: int = 1,
) -> dict[str, Any]:
    """Stream one optimizer trial candidate into HUD."""

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
            policy=policy,
            reason=reason,
            tokens=tokens,
            model_cost_usd=model_cost_usd,
            source_path=source_path,
            summary=summary or {},
            eval_error=eval_error,
            accepted=accepted,
            env_source=str(env_path),
            timeout=timeout,
            session=session,
            group=group,
        )
    )
