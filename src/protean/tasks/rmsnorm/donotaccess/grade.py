"""Hidden-grade compatible wrapper for the rmsnorm task."""

from __future__ import annotations

import hashlib
from pathlib import Path

from protean.grader import grade_source


REWARDS_HASH = "<sha256 baked at image build>"


def _preflight_rewards_hash() -> None:
    if REWARDS_HASH == "<sha256 baked at image build>":
        return
    rewards_path = Path("/donotaccess/rewards.py")
    if not rewards_path.exists():
        raise RuntimeError(f"REWARDS_HASH is baked but {rewards_path} does not exist")
    actual = hashlib.sha256(rewards_path.read_bytes()).hexdigest()
    if actual != REWARDS_HASH:
        raise RuntimeError(f"REWARDS_HASH mismatch {actual} != {REWARDS_HASH}")


def grade_kernel(op, shape, dtype, kernel_src, split="held_out", step=0):
    _preflight_rewards_hash()
    return grade_source(kernel_src, op=op, split=split, shape=int(shape))


def grade(workdir, override=None, hidden_root=None):
    _preflight_rewards_hash()
    path = Path(workdir) / "solution.py"
    source = override if override is not None else path.read_text()
    return grade_source(source, op="rmsnorm", split="held_out")
