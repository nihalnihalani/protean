"""Harness policy knobs the agent is allowed to tune later."""

from __future__ import annotations

DEFAULT_HARNESS = {
    "reps": 30,
    "warmup": 8,
}


def harness_for_kernel_edit() -> dict[str, int]:
    """Return a fresh harness config for one candidate evaluation."""

    return dict(DEFAULT_HARNESS)
