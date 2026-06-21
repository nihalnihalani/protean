"""Structured reward for Protean's verifier-first MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    p_target: float = 1.5
    speedup_floor: float = 1.1
    correct_floor: float = 0.3
    max_reward: float = 2.0


DEFAULT_CONFIG = RewardConfig()


def compute_reward(
    *,
    correct: bool,
    speedup: float,
    launches_timed: int,
    dtype_ok: bool,
    shape_ok: bool,
    split: str,
    t_eager_ms: float | None = None,
    t_kernel_ms: float | None = None,
    caps: list[str] | None = None,
    config: RewardConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return the public grade payload.

    Correctness, dtype/shape integrity, and @triton.jit usage are hard gates.
    Slow-but-correct kernels keep a correctness floor so the report can separate
    "correct" from "fast"; they do not earn speedup reward.
    """

    caps = list(caps or [])
    if not correct:
        caps.append("incorrect")
    if not dtype_ok:
        caps.append("dtype_mismatch")
    if not shape_ok:
        caps.append("shape_mismatch")
    if launches_timed <= 0 and not caps:
        caps.append("no_triton_jit")

    hard_failed = bool(caps)
    speedup_reward = 0.0
    correctness_reward = config.correct_floor if correct and dtype_ok and shape_ok else 0.0

    if not hard_failed and speedup >= config.speedup_floor:
        speedup_reward = min(speedup / config.p_target, 1.0)
    elif correct and dtype_ok and shape_ok and speedup < config.speedup_floor:
        caps.append("below_speedup_floor")

    reward = correctness_reward + speedup_reward
    if hard_failed:
        reward = 0.0

    return {
        "reward": round(min(reward, config.max_reward), 6),
        "correct": bool(correct),
        "speedup": round(float(speedup), 6) if speedup is not None else 0.0,
        "t_eager_ms": round(float(t_eager_ms), 6) if t_eager_ms is not None else None,
        "t_kernel_ms": round(float(t_kernel_ms), 6) if t_kernel_ms is not None else None,
        "split": split,
        "caps": sorted(set(caps)),
        "launches_timed": int(launches_timed),
        "dtype_ok": bool(dtype_ok),
        "shape_ok": bool(shape_ok),
    }
