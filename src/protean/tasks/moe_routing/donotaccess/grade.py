"""Hidden-grade compatible wrapper for Protean op tasks."""

from __future__ import annotations

import hashlib
import os
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


def _parse(workdir, override=None, hidden_root=None):
    import json

    metadata_path = os.path.join(workdir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("op_name", meta.get("op")), meta["split"], meta["seed"]
    op = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return op, "train", 0


def grade_kernel(op, shape, dtype, kernel_src, split="held_out", step=0):
    _preflight_rewards_hash()
    return grade_source(kernel_src, op=op, split=split, shape=int(shape))


def grade(workdir, override=None, hidden_root=None):
    _preflight_rewards_hash()
    op, split, _ = _parse(workdir, override, hidden_root)
    source = override if override is not None else (Path(workdir) / "solution.py").read_text()
    return grade_source(source, op=op, split=split)

