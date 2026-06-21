"""Direct grader used by HUD, scripts, and future training code."""

# NOTE: this file deliberately omits ``from __future__ import annotations`` (kept in sync
# with env.py). grader.py defines NO ``@env.template`` functions, so the HUD TypeAdapter
# crash from env.py's comment block does not apply here today. But removing the import keeps
# annotation-evaluation symmetric across the two modules and closes the surface: if a graded
# coroutine is ever decorated with ``@env.template`` here, a lingering future-import would
# silently reintroduce the deploy-time PydanticUserError (-32000). Python 3.12 resolves
# ``int | None``/``str | None`` natively (PEP 604), so removal is runtime-safe. Do NOT add
# the future-import back; see env.py's comment block for the full rationale.

import importlib.util
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, cast

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


# Structural contracts for the HUD evaluation classes. Declaring them as
# Protocols (rather than erasing both names to ``Any`` via ``tuple[Any, Any]``)
# lets mypy type-check every kwarg passed to ``SubScore(...)`` and
# ``EvaluationResult(...)`` in ``to_eval_result`` -- a keyword typo, a wrong
# value type, or a missing field is now a mypy error instead of a silent pass.
# Both the real ``hud.graders`` classes and the local dataclass fallback satisfy
# these structurally, so no runtime behaviour changes.


class _SubScoreProto(Protocol):
    name: str
    value: float
    weight: float


class _SubScoreFactory(Protocol):
    def __call__(
        self, *, name: str, value: float, weight: float, metadata: dict[str, Any] | None = ...
    ) -> _SubScoreProto: ...


class _EvalResultProto(Protocol):
    reward: float
    subscores: list[Any]
    info: dict[str, Any]


class _EvalResultFactory(Protocol):
    def __call__(
        self,
        *,
        reward: float,
        subscores: list[Any],
        info: dict[str, Any],
        done: bool = ...,
        content: str | None = ...,
    ) -> _EvalResultProto: ...


# Local dataclass stand-ins used only when HUD is not installed. Defined at module
# scope (not inside the resolver) so they are never redefined -- eliminating the
# ``no-redef`` suppressions the in-function fallback previously required.


@dataclass
class _FallbackSubScore:
    name: str
    value: float
    weight: float
    metadata: dict[str, Any] | None = None


@dataclass
class _FallbackEvalResult:
    reward: float
    subscores: list[Any]
    info: dict[str, Any]
    done: bool = True
    content: str | None = None


def _eval_result_classes() -> tuple[_EvalResultFactory, _SubScoreFactory]:
    """Resolve (EvaluationResult, SubScore), falling back to local dataclasses.

    Returns the HUD classes when ``hud.graders`` is importable, otherwise the
    module-level fallback dataclasses. Both satisfy the factory Protocols, so the
    call site in ``to_eval_result`` is fully type-checked.
    """
    try:
        from hud.graders import EvaluationResult, SubScore
    except Exception:  # pragma: no cover - local tests should not require HUD.
        return _FallbackEvalResult, _FallbackSubScore

    # hud.graders is ignore_missing_imports in CI (HUD not installed), so mypy
    # erases EvaluationResult/SubScore to Any and cannot confirm they satisfy the
    # factory Protocols structurally. The local fallback dataclasses define the
    # ground-truth shape (reward/subscores/info and name/value/weight); the real
    # HUD classes are constructed with exactly these kwargs throughout the SDK, so
    # the cast is honest. It restores Protocol typing for the call site below.
    return cast("_EvalResultFactory", EvaluationResult), cast("_SubScoreFactory", SubScore)


def _reward_ceiling() -> float:
    """Resolve the raw-reward ceiling used to normalise into HUD's [0, 1] range.

    ``compute_reward`` returns a reward in ``[0, config.max_reward]`` (default
    ``2.0``), but HUD's ``EvaluationResult.reward`` is conventionally ``[0, 1]``.
    The normaliser therefore divides by ``max_reward`` rather than a hardcoded
    ``2.0`` so that a production ``reward_config.json`` changing ``max_reward``
    cannot silently push HUD rewards above 1.0 (or compress them). Degrades to
    ``DEFAULT_CONFIG.max_reward`` if the on-disk config is missing/unreadable --
    matching ``grade_source``'s graceful-degradation contract.
    """

    config, _missing = _resolve_reward_config()
    ceiling = float(config.max_reward)
    return ceiling if ceiling > 0.0 else float(DEFAULT_CONFIG.max_reward)


def to_eval_result(grade_dict: dict[str, Any]) -> Any:
    raw_reward = float(grade_dict["reward"])
    ceiling = _reward_ceiling()
    hud_reward = max(0.0, min(raw_reward / ceiling, 1.0))
    caps = grade_dict.get("caps") or []
    correct = bool(grade_dict.get("correct", False)) and not caps
    speedup = grade_dict.get("speedup")
    speedup_score = max(0.0, min(float(grade_dict.get("speedup_score", 0.0)), 1.0))
    split = grade_dict.get("split")
    held_out = 1.0 if split == "held_out" and correct else 0.0
    anti_hack = 1.0 if not caps else 0.0
    compile_success = 1.0 if "cuda_unavailable" not in caps else 0.0
    op = grade_dict.get("op")
    info = {
        **grade_dict,
        "protean_reward_raw": raw_reward,
        "protean_reward_max": ceiling,
        "hud_reward_normalized": hud_reward,
    }

    # Human-readable one-line summary for the HUD dashboard (content renders in
    # the run view). Mirrors the verilog-template's "<task> graded" pattern but
    # surfaces the substrate that actually drove the reward.
    cap_str = ",".join(caps) if caps else "none"
    speedup_str = f"{float(speedup):.2f}x" if isinstance(speedup, (int, float)) else "n/a"
    content = (
        f"{op or 'kernel'}/{split or '?'}: reward={hud_reward:.3f} "
        f"(raw={raw_reward:.3f}/{ceiling:g}) correct={correct} "
        f"speedup={speedup_str} caps={cap_str}"
    )

    # Per-subscore metadata exposes the raw timing/profiling substrate in the
    # dashboard's subscore drill-down. ``SubScore.metadata`` is excluded from the
    # scored wire payload (Field(exclude=True)) so it is diagnostic-only and does
    # not affect reward math.
    EvaluationResult, SubScore = _eval_result_classes()

    return EvaluationResult(
        reward=hud_reward,
        done=True,
        content=content,
        subscores=[
            SubScore(
                name="hud_reward",
                value=hud_reward,
                weight=1.0,
                metadata={
                    "raw_reward": raw_reward,
                    "max_reward": ceiling,
                    "correctness_reward": grade_dict.get("correctness_reward"),
                    "speedup_reward": grade_dict.get("speedup_reward"),
                    "pr_reward": grade_dict.get("pr_reward"),
                    "pr_mode": grade_dict.get("pr_mode"),
                },
            ),
            SubScore(name="correctness", value=1.0 if correct else 0.0, weight=0.0),
            SubScore(
                name="speedup",
                value=speedup_score,
                weight=0.0,
                metadata={
                    "speedup": speedup,
                    "t_eager_ms": grade_dict.get("t_eager_ms"),
                    "t_kernel_ms": grade_dict.get("t_kernel_ms"),
                    "pr_frac": grade_dict.get("pr_frac"),
                },
            ),
            SubScore(
                name="held_out",
                value=held_out,
                weight=0.0,
                metadata={"split": split},
            ),
            SubScore(
                name="anti_hack",
                value=anti_hack,
                weight=0.0,
                metadata={"caps": caps, "caps_advisory": grade_dict.get("caps_advisory")},
            ),
            SubScore(
                name="compile_success",
                value=compile_success,
                weight=0.0,
                metadata={"launches_timed": grade_dict.get("launches_timed")},
            ),
        ],
        info=info,
    )
