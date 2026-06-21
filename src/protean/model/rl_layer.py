"""Reward shaping and acceptance policy for the kernel-optimizer agent."""

from __future__ import annotations


def score(summary: dict) -> tuple[float, float, int]:
    """Rank candidates by held-out speed first, then reward, then coverage."""

    return (
        float(summary["mean_held_out_speedup"]),
        float(summary["mean_reward"]),
        int(summary["correct_held_out"]),
    )


def score_delta(candidate: tuple[float, float, int], baseline: tuple[float, float, int]) -> dict:
    return {
        "held_out_speedup": round(candidate[0] - baseline[0], 6),
        "reward": round(candidate[1] - baseline[1], 6),
        "correct_held_out": candidate[2] - baseline[2],
    }


def accept_candidate(candidate: tuple[float, float, int], baseline: tuple[float, float, int]) -> bool:
    """Accept only strict improvements over the current best."""

    return candidate > baseline
