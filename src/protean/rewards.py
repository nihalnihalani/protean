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
    # Profiling-aware reward shaping. ``pr_mode`` selects how the profiling
    # ratio (pr_frac, the fraction of runtime spent in the generated kernel)
    # contributes to the reward:
    #   "bonus"          - legacy additive bonus (config.pr_bonus * pr_frac).
    #                      Default; preserves historical reward values exactly.
    #   "multiplicative" - Dr. Kernel form: speedup_reward is scaled by
    #                      (1 + pr_frac) and no separate additive term is added.
    #                      This makes speedup credit conditional on the kernel
    #                      actually being the runtime bottleneck, discouraging
    #                      lazy optimization of non-bottleneck kernels.
    #   "centered"       - gradient-neutral profiling bonus. The bonus is
    #                      config.pr_bonus * (pr_frac - pr_center), which is
    #                      zero-mean over a kernel population whose mean
    #                      pr_frac equals pr_center. At single-op scope this
    #                      re-ranks by profiling quality without shifting the
    #                      expected reward, so it adds no bias to the policy
    #                      gradient baseline.
    pr_mode: str = "bonus"
    pr_center: float = 0.5


DEFAULT_CONFIG = RewardConfig()
CANONICAL_CONFIG_PATH = Path("/donotaccess/reward_config.json")
LOCAL_CONFIG_PATH = Path(__file__).with_name("reward_config.json")


def _config_path() -> Path:
    if CANONICAL_CONFIG_PATH.exists():
        return CANONICAL_CONFIG_PATH
    return LOCAL_CONFIG_PATH


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
        pr_mode=str(data.get("PR_MODE", data.get("pr_mode", DEFAULT_CONFIG.pr_mode))),
        pr_center=float(data.get("PR_CENTER", data.get("pr_center", DEFAULT_CONFIG.pr_center))),
    )


def _pr_reward(config: RewardConfig, pr_clamped: float, hard_failed: bool) -> float:
    """Additive profiling-aware reward term, selected by ``config.pr_mode``.

    Returns 0.0 on any hard failure. In ``multiplicative`` mode the profiling
    signal is already folded into ``speedup_reward`` (scaled by 1 + pr_frac),
    so there is no separate additive term here. ``centered`` mode returns a
    zero-mean (gradient-neutral) bonus about ``pr_center``.
    """

    if hard_failed:
        return 0.0
    if config.pr_mode == "multiplicative":
        return 0.0
    if config.pr_mode == "centered":
        return config.pr_bonus * (pr_clamped - config.pr_center)
    # Default "bonus" mode: legacy additive term.
    return config.pr_bonus * pr_clamped


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
        caps.append("no_triton_jit")

    hard_failed = bool(caps)
    speedup_reward = 0.0
    speedup_score = 0.0
    correctness_reward = config.correct_floor if correct and dtype_ok and shape_ok else 0.0

    pr_clamped = max(0.0, min(float(pr_frac), 1.0))

    if not hard_failed and speedup >= config.speedup_floor:
        capped_speedup = max(config.speedup_floor, min(float(speedup), config.speedup_cap))
        denominator = math.log(config.speedup_cap / config.speedup_floor)
        speedup_score = math.log(capped_speedup / config.speedup_floor) / denominator if denominator > 0 else 0.0
        speedup_score = max(0.0, min(speedup_score, 1.0))
        speedup_reward = config.speedup_reward_weight * speedup_score
        if config.pr_mode == "multiplicative":
            # Dr. Kernel (arXiv:2602.05885): gate speedup credit on the kernel
            # being the runtime bottleneck. No separate additive PR term.
            #
            # The base weight is re-normalised by (1 + max_pr) where max_pr=1.0
            # (pr_clamped is bounded to [0, 1]). Without this, a high-pr kernel
            # at speedup_reward_weight=1.5 would reach 1.5*1.0*(1+1.0)=3.0 and,
            # together with correctness_reward, sit far above max_reward=2.0 --
            # the final clamp would then flatten the speedup gradient across the
            # entire top of the speedup_score range. Normalising keeps the
            # maximum pre-clamp speedup credit at the same ceiling as "bonus"
            # mode, so speedup_score stays informative for the policy gradient
            # while pr_clamped still re-weights bottleneck vs non-bottleneck
            # kernels.
            speedup_reward = speedup_reward * (1.0 + pr_clamped) / (1.0 + 1.0)
    elif correct and dtype_ok and shape_ok and speedup < config.speedup_floor:
        caps.append("below_speedup_floor")

    pr_reward = _pr_reward(config, pr_clamped, hard_failed)
    reward = correctness_reward + speedup_reward + pr_reward
    if hard_failed:
        reward = 0.0

    # Lower-bound clamp. "centered" mode subtracts pr_bonus*(pr_center-pr_frac)
    # for below-center kernels, which can otherwise push a correct kernel's
    # total below zero (e.g. a kernel just under speedup_floor with pr_frac=0).
    # The bound keeps reward non-negative without affecting "bonus" or
    # "multiplicative" modes, whose contributions are always >= 0.
    reward = max(0.0, reward)

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
        "pr_reward": round(pr_reward, 6),
        "pr_mode": config.pr_mode,
        "split": split,
        "caps": sorted(set(caps)),
        "launches_timed": int(launches_timed),
        "dtype_ok": bool(dtype_ok),
        "shape_ok": bool(shape_ok),
    }
