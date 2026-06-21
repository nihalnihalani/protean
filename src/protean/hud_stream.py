"""HUD streaming helpers for optimizer trials.

Public surface
--------------
Two layers are exposed:

* Pure-async coroutines -- :func:`_start_session_async` and
  :func:`_stream_candidate_async`. Call these directly from inside an existing
  event loop (e.g. an async optimizer, a notebook, or a server handler). They
  never touch :func:`asyncio.run`, so they compose cleanly with any caller's
  loop.
* Sync shims -- :func:`start_hud_stream_session` and
  :func:`stream_candidate_to_hud`. These are convenience wrappers for plain
  synchronous code. They drive the coroutine to completion via :func:`_run_sync`,
  which only calls :func:`asyncio.run` when *no* loop is already running. If a
  loop is already running the work is dispatched to a dedicated worker thread
  with its own loop, so the sync shims never raise
  ``RuntimeError: asyncio.run() cannot be called from a running event loop``.

The ``group`` argument
----------------------
``group`` is currently **logging / bookkeeping only**. It is forwarded to the
HUD ``Job`` and rollout calls so trials can be visually grouped in the HUD
dashboard, but it is **not** wired into any reinforcement-learning advantage
computation: the HUD ``TrainingClient`` (GRPO-style group-relative advantage)
path is intentionally not connected here. Treat ``group`` as a tag, not as a
training signal.
"""

import asyncio
import concurrent.futures
import os
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


class HudStreamError(RuntimeError):
    """Raised when a trial cannot be streamed to HUD."""


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive ``coro`` to completion from synchronous code, loop-safe.

    When there is no running event loop in the current thread, this is just
    :func:`asyncio.run`. When a loop *is* already running (notebooks, async
    callers that reached this sync shim by mistake), calling ``asyncio.run``
    would raise ``RuntimeError``; instead the coroutine is executed on a
    dedicated worker thread that owns a fresh loop. ``asyncio.run`` is therefore
    only ever invoked where no loop is active.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread -- the common, fast path.
        return asyncio.run(coro)

    # A loop is already running here; run the coroutine in its own thread/loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


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


def _read_hud_user_env(path: Path | None = None) -> str | None:
    """Read HUD_API_KEY from the HUD CLI user config without sourcing shell."""

    hud_env = path or (Path.home() / ".hud" / ".env")
    if not hud_env.exists():
        return None
    for line in hud_env.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "HUD_API_KEY":
            continue
        value = value.strip().strip('"').strip("'")
        return value or None
    return None


def _sync_hud_settings(api_key: str) -> None:
    try:
        from hud.settings import settings
    except Exception:
        return
    try:
        settings.api_key = api_key
    except Exception:
        # Some HUD builds may make settings immutable. The process env fallback
        # still covers new API clients created after this point.
        pass


def resolve_hud_api_key() -> str:
    """Resolve and propagate HUD auth for CLI, SDK settings, and API clients."""

    key = os.environ.get("HUD_API_KEY")
    if not key:
        settings: Any | None
        try:
            from hud.settings import settings
        except Exception:
            settings = None
        key = getattr(settings, "api_key", None) if settings is not None else None
    if not key:
        key = _read_hud_user_env()
    if not key:
        raise HudStreamError("HUD_API_KEY is required to stream optimizer trials to HUD")

    os.environ["HUD_API_KEY"] = key
    _sync_hud_settings(key)
    return key


def assert_hud_auth() -> None:
    """Fail before GPU work if HUD auth is not available."""

    key = resolve_hud_api_key()
    if not key.strip():
        raise HudStreamError("HUD_API_KEY is empty")


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


async def _start_session_async(name: str, group: int = 1) -> HudStreamSession:
    """Async: open one HUD job. Safe to await from inside an existing loop.

    ``group`` is logging-only (see module docstring): it tags the HUD job for
    dashboard grouping and is not used for any advantage/training computation.
    """

    assert_hud_auth()
    from hud.eval import Job

    job = await Job.start(name, group=group)
    return HudStreamSession(job=job, name=name, group=group)


def start_hud_stream_session(*, name: str | None = None, group: int = 1) -> HudStreamSession:
    """Create one HUD job for a whole optimizer run (sync shim).

    Prefer awaiting :func:`_start_session_async` directly when you already have
    an event loop. This wrapper is loop-safe via :func:`_run_sync`.

    ``group`` is logging-only (see module docstring).
    """

    assert_hud_auth()
    job_name = name or f"protean-optimizer-{int(time.time())}"
    return _run_sync(_start_session_async(job_name, group))


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
    controller_decision: dict[str, Any] | None,
    source_path: str,
    summary: dict[str, Any],
    eval_error: dict[str, Any] | None,
    accepted: bool,
    env_source: str,
    timeout: float,
    session: HudStreamSession | None,
    group: int,
) -> dict[str, Any]:
    """Async: stream one candidate trial. Safe to await from inside a loop.

    ``group`` is logging-only (see module docstring): it tags the rollout for
    dashboard grouping; the HUD ``TrainingClient`` advantage path is not wired.
    """

    assert_hud_auth()
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime, Taskset
    from hud.types import Step

    metrics = _summary_metrics(summary, eval_error)

    class CandidateAgent(Agent):
        async def __call__(self, run: Any) -> None:
            run.trace.extra["agent"] = "protean_optimizer_stream"
            run.trace.extra.update(
                {
                    "op": op,
                    "trial": trial,
                    "edit": edit,
                    "policy": policy,
                    "tokens": tokens,
                    "model_cost_usd": model_cost_usd,
                    "controller_decision": controller_decision,
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
                        "controller_decision": controller_decision,
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
    assert_hud_auth()
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
    controller_decision: dict[str, Any] | None = None,
    source_path: str = "",
    summary: dict[str, Any] | None = None,
    eval_error: dict[str, Any] | None = None,
    env_source: str | Path = "src/protean/env.py",
    timeout: float = 180.0,
    session: HudStreamSession | None = None,
    group: int = 1,
) -> dict[str, Any]:
    """Stream one optimizer trial candidate into HUD (sync shim).

    Prefer awaiting :func:`_stream_candidate_async` directly when you already
    have an event loop. This wrapper is loop-safe via :func:`_run_sync` and will
    not raise the nested-event-loop ``RuntimeError``.

    ``group`` is logging-only (see module docstring): the HUD ``TrainingClient``
    advantage path is intentionally not wired here.
    """

    assert_hud_auth()
    env_path = Path(env_source)
    if not env_path.exists():
        raise HudStreamError(f"HUD env source does not exist: {env_path}")
    return _run_sync(
        _stream_candidate_async(
            source=source,
            op=op,
            trial=trial,
            edit=edit,
            policy=policy,
            reason=reason,
            tokens=tokens,
            model_cost_usd=model_cost_usd,
            controller_decision=controller_decision,
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
