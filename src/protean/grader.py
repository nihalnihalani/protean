"""Direct grader used by HUD, scripts, and future training code."""

from __future__ import annotations

import importlib.util
import logging
import os
import time
import uuid
from typing import Any

from protean.anti_hack import ast_clean, contains_triton_jit
from protean.rewards import DEFAULT_CONFIG, RewardConfig, compute_reward, load_reward_config
from protean.splits import Split, shapes_for_split
from protean.task_catalog import OPS_BY_NAME, get_op

# Public surface of this module. Declared explicitly so the typed-error
# hierarchy and the grader entry points are part of the stable, documented
# contract rather than incidental module attributes -- callers may rely on
# ``from protean.grader import ProteanValidationError`` not drifting.
__all__ = [
    "ProteanError",
    "ProteanValidationError",
    "grade_source",
    "to_eval_result",
]

# --- Typed domain errors ---------------------------------------------------
# A small exception hierarchy so callers can catch Protean-specific failures
# without swallowing unrelated exceptions. ProteanValidationError signals bad
# caller input at the public API boundary (unknown op, invalid split, etc.);
# it is raised *before* any I/O or CUDA call, so it is always safe to surface.


class ProteanError(Exception):
    """Root of the Protean domain-error hierarchy."""


class ProteanValidationError(ProteanError, ValueError):
    """Invalid caller input to a public Protean API.

    Subclasses ``ValueError`` so existing callers that already catch
    ``ValueError`` (the previous, untyped failure mode) keep working.
    """


# --- Structured logging ----------------------------------------------------
# A single module logger emitting structured (key=value) diagnostics. It is a
# no-op unless a handler is attached (NullHandler default), so importing the
# grader never spams stdout and existing tests that assert on stdout are
# unaffected. Set PROTEAN_LOG=1 (or DEBUG/INFO/...) to attach a stderr handler.

_LOG = logging.getLogger("protean.grader")
_LOG.addHandler(logging.NullHandler())


def _maybe_enable_logging() -> None:
    level_env = os.environ.get("PROTEAN_LOG")
    if not level_env:
        return
    if _LOG.handlers and any(not isinstance(h, logging.NullHandler) for h in _LOG.handlers):
        return
    level = {"0": logging.CRITICAL + 1, "1": logging.INFO}.get(
        level_env, getattr(logging, level_env.upper(), logging.INFO)
    )
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(name)s %(levelname)s %(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(level)


_maybe_enable_logging()


def _log_event(level: int, event: str, **fields: Any) -> None:
    if not _LOG.isEnabledFor(level):
        return
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    _LOG.log(level, "%s %s", event, payload)


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _resolve_reward_config() -> tuple[RewardConfig, bool]:
    """Load the reward config, degrading gracefully to ``DEFAULT_CONFIG``.

    Returns ``(config, missing)`` where ``missing`` is True when the on-disk
    config could not be read and the in-code default was substituted. This
    keeps ``grade_source`` self-contained: it always returns a well-formed
    grade dict rather than propagating a ``FileNotFoundError`` to the caller.
    """

    try:
        return load_reward_config(), False
    except FileNotFoundError:
        return DEFAULT_CONFIG, True
    except (OSError, ValueError):
        # Unreadable / malformed config -- still degrade rather than crash.
        return DEFAULT_CONFIG, True


def _validate_grade_inputs(src: str, op: str, split: Split, shape: int | None, reps: int, warmup: int) -> None:
    """Guard the public ``grade_source`` contract with typed errors.

    These checks fire before any I/O, CUDA, or compilation so callers get a
    clear, catchable ``ProteanValidationError`` at the boundary rather than an
    opaque exception raised several frames deep inside the grader.
    """

    if not isinstance(src, str) or not src.strip():
        raise ProteanValidationError("src must be a non-empty string")
    if op not in OPS_BY_NAME:
        raise ProteanValidationError(f"op {op!r} not in {sorted(OPS_BY_NAME)}")
    if split not in ("train", "held_out"):
        raise ProteanValidationError(f"split {split!r} must be 'train' or 'held_out'")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        raise ProteanValidationError(f"reps={reps!r} must be an integer >= 1")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ProteanValidationError(f"warmup={warmup!r} must be an integer >= 0")
    if shape is not None and (not isinstance(shape, int) or isinstance(shape, bool) or shape < 1):
        raise ProteanValidationError(f"shape={shape!r} must be None or a positive integer")


def grade_source(
    src: str,
    *,
    op: str = "elementwise_add_relu",
    split: Split = "held_out",
    shape: int | None = None,
    reps: int = 50,
    warmup: int = 10,
    run_id: str | None = None,
) -> dict[str, Any]:
    _validate_grade_inputs(src, op, split, shape, reps, warmup)
    run_id = run_id or uuid.uuid4().hex[:12]
    spec = get_op(op)
    shape = shape or shapes_for_split(split)[0]

    config, config_missing = _resolve_reward_config()
    base_caps = ["reward_config_missing"] if config_missing else None

    def _degraded(*, caps: list[str]) -> dict[str, Any]:
        merged = (base_caps or []) + caps
        return compute_reward(
            correct=False,
            speedup=0.0,
            launches_timed=0,
            dtype_ok=False,
            shape_ok=False,
            split=split,
            caps=merged,
            config=config,
        ) | {"op": op, "shape": shape, "run_id": run_id}

    started = time.perf_counter()
    _log_event(
        logging.DEBUG,
        "grade_source",
        run_id=run_id,
        op=op,
        split=split,
        shape=shape,
        src_len=len(src),
    )

    ok, reason = ast_clean(src)
    if not ok:
        result = _degraded(caps=[reason])
    elif not contains_triton_jit(src):
        result = _degraded(caps=["no_triton_jit"])
    elif not _cuda_available():
        result = _degraded(caps=["cuda_unavailable"])
    else:
        from protean.bench_core import bench_source

        bench = bench_source(src, n=shape, split=split, spec=spec, reps=reps, warmup=warmup)
        grade = compute_reward(
            correct=bench["correct"],
            speedup=bench["speedup"],
            launches_timed=bench["launches_timed"],
            dtype_ok=bench["dtype_ok"],
            shape_ok=bench["shape_ok"],
            split=split,
            t_eager_ms=bench["t_eager_ms"],
            t_kernel_ms=bench["t_kernel_ms"],
            pr_frac=bench.get("pr_frac", 0.0),
            caps=(base_caps or []) + (bench.get("caps") or []) or None,
            config=config,
        )
        # Advisory caps (e.g. launch_count_unverified on Triton builds without the
        # runtime launch-hook surface) are surfaced for auditability but do NOT gate
        # the reward -- they are kept separate from the hard-gating caps so a legit
        # kernel is not zeroed merely because its launch count could not be verified.
        extra: dict[str, Any] = {"op": op, "shape": shape, "run_id": run_id}
        advisory = bench.get("caps_advisory") or []
        if advisory:
            extra["caps_advisory"] = sorted(set(advisory))
        result = grade | extra

    _log_event(
        logging.DEBUG,
        "grade_result",
        run_id=run_id,
        op=op,
        split=split,
        reward=result.get("reward"),
        caps=result.get("caps"),
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
    return result


def to_eval_result(grade_dict: dict):
    raw_reward = float(grade_dict["reward"])
    hud_reward = max(0.0, min(raw_reward / 2.0, 1.0))
    correct = bool(grade_dict.get("correct", False)) and not grade_dict.get("caps")
    speedup_score = max(0.0, min(float(grade_dict.get("speedup_score", 0.0)), 1.0))
    held_out = 1.0 if grade_dict.get("split") == "held_out" and correct else 0.0
    anti_hack = 1.0 if not grade_dict.get("caps") else 0.0
    compile_success = 1.0 if "cuda_unavailable" not in grade_dict.get("caps", []) else 0.0
    info = {
        **grade_dict,
        "protean_reward_raw": raw_reward,
        "hud_reward_normalized": hud_reward,
    }
    try:
        from hud.graders import EvaluationResult, SubScore
    except Exception:  # pragma: no cover - local tests should not require HUD.
        from dataclasses import dataclass

        @dataclass
        class SubScore:
            name: str
            value: float
            weight: float

        @dataclass
        class EvaluationResult:
            reward: float
            subscores: list[SubScore]
            info: dict

    return EvaluationResult(
        reward=hud_reward,
        subscores=[
            SubScore(name="hud_reward", value=hud_reward, weight=1.0),
            SubScore(name="correctness", value=1.0 if correct else 0.0, weight=0.0),
            SubScore(name="speedup", value=speedup_score, weight=0.0),
            SubScore(name="held_out", value=held_out, weight=0.0),
            SubScore(name="anti_hack", value=anti_hack, weight=0.0),
            SubScore(name="compile_success", value=compile_success, weight=0.0),
        ],
        info=info,
    )
