"""Deterministic task sampling over the frozen shape split."""

from __future__ import annotations

import hashlib
import random

from protean.splits import Split, shapes_for_split


def _rng(op: str, idx: int, split: Split) -> random.Random:
    seed = int(hashlib.sha256(f"{op}|{idx}|{split}".encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def sample_task(op: str, idx: int, split: Split) -> dict:
    shape = _rng(op, idx, split).choice(shapes_for_split(split))
    return {"op": op, "shape": shape, "dtype": "float16", "split": split}
