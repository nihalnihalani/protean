"""Structured reward for Protean's verifier-first MVP."""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    p_target: float = 1.5
    speedup_floor: float = 1.1
    speedup_cap: float = 20.0
    correct_floor: float = 0.3
    speedup_reward_weight: float = 1.5
    pr_bonus: float = 0.2
    max_reward: float = 2.0


DEFAULT_CONFIG = RewardConfig()
CANONICAL_CONFIG_PATH = Path("/donotaccess/reward_config.json")
LOCAL_CONFIG_PATH = Path(__file__).with_name("reward_config.json")

# Backward compatibility aliases for tests
_CANONICAL_CFG_PATH = "/donotaccess/reward_config.json"
_LOCAL_CFG_PATH = str(LOCAL_CONFIG_PATH)


def _config_path() -> Path:
    # Try the canonical config first
    canonical = CANONICAL_CONFIG_PATH if CANONICAL_CONFIG_PATH is not None else Path(_CANONICAL_CFG_PATH)
    if canonical.exists():
        return canonical
    # Fallback to local/dev config
    local = LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH is not None else Path(_LOCAL_CFG_PATH)
    # Check if string-based backward compatibility path was monkeypatched
    default_local_str = str(Path(__file__).with_name("reward_config.json"))
    if _LOCAL_CFG_PATH != default_local_str:
        local = Path(_LOCAL_CFG_PATH)
    return local


def _cfg() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(f"reward config is missing: {path}")
    return json.loads(path.read_text())


def load_reward_config() -> RewardConfig:
    """Load reward thresholds from the hidden production config or local dev copy."""

    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(f"reward config is missing: {path}")
    data = json.loads(path.read_text())
    return RewardConfig(
        p_target=float(data.get("P_TARGET", data.get("p_target", DEFAULT_CONFIG.p_target))),
        speedup_floor=float(data.get("SPEEDUP_FLOOR", data.get("speedup_floor", DEFAULT_CONFIG.speedup_floor))),
        speedup_cap=float(data.get("SPEEDUP_CAP", data.get("speedup_cap", DEFAULT_CONFIG.speedup_cap))),
        correct_floor=float(data.get("CORRECT_FLOOR", data.get("correct_floor", DEFAULT_CONFIG.correct_floor))),
        speedup_reward_weight=float(
            data.get("SPEEDUP_REWARD_WEIGHT", data.get("speedup_reward_weight", DEFAULT_CONFIG.speedup_reward_weight))
        ),
        pr_bonus=float(data.get("PR_BONUS", data.get("pr_bonus", DEFAULT_CONFIG.pr_bonus))),
        max_reward=float(data.get("MAX_REWARD", data.get("max_reward", DEFAULT_CONFIG.max_reward))),
    )


def compute_reward(
    *,
    correct: bool,
    speedup: float,
    launches_timed: int,
    dtype_ok: bool,
    shape_ok: bool,
    split: str = "train",
    t_eager_ms: float | None = None,
    t_kernel_ms: float | None = None,
    pr_frac: float = 0.0,
    caps: list[str] | None = None,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Return the public grade payload.

    Correctness, dtype/shape integrity, and real Triton execution are hard
    gates. Correct kernels earn a small floor, then a continuous log-scaled
    speedup reward. Log scaling keeps 2x < 6x < 12x while damping timing
    outliers, so the optimizer still sees meaningful gains after clearing a
    threshold.
    """

    caps = list(caps or [])
    config = config or load_reward_config()
    if not correct:
        caps.append("incorrect")
    if not dtype_ok:
        caps.append("dtype_mismatch")
    if not shape_ok:
        caps.append("shape_mismatch")
    if launches_timed <= 0 and not caps:
        caps.append("no_triton_launch")

    hard_failed = bool(caps)
    speedup_reward = 0.0
    speedup_score = 0.0
    correctness_reward = config.correct_floor if correct and dtype_ok and shape_ok else 0.0

    if not hard_failed and speedup >= config.speedup_floor:
        capped_speedup = max(config.speedup_floor, min(float(speedup), config.speedup_cap))
        denominator = math.log(config.speedup_cap / config.speedup_floor)
        speedup_score = math.log(capped_speedup / config.speedup_floor) / denominator if denominator > 0 else 0.0
        speedup_score = max(0.0, min(speedup_score, 1.0))
        speedup_reward = config.speedup_reward_weight * speedup_score
    elif correct and dtype_ok and shape_ok and speedup < config.speedup_floor:
        caps.append("below_speedup_floor")

    pr_clamped = max(0.0, min(float(pr_frac), 1.0))
    reward = correctness_reward + speedup_reward + (config.pr_bonus * pr_clamped if not hard_failed else 0.0)
    if hard_failed:
        reward = 0.0

    return {
        "reward": round(min(reward, config.max_reward), 6),
        "correct": bool(correct),
        "speedup": round(float(speedup), 6) if speedup is not None else 0.0,
        "t_eager_ms": round(float(t_eager_ms), 6) if t_eager_ms is not None else None,
        "t_kernel_ms": round(float(t_kernel_ms), 6) if t_kernel_ms is not None else None,
        "pr_frac": round(pr_clamped, 6),
        "speedup_score": round(speedup_score, 6),
        "correctness_reward": round(correctness_reward, 6),
        "speedup_reward": round(speedup_reward, 6),
        "pr_reward": round(config.pr_bonus * pr_clamped if not hard_failed else 0.0, 6),
        "split": split,
        "caps": sorted(set(caps)),
        "launches_timed": int(launches_timed),
        "dtype_ok": bool(dtype_ok),
        "shape_ok": bool(shape_ok),
    }
