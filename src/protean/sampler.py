"""Deterministic task sampling over the frozen shape split."""

from __future__ import annotations

import hashlib
import random

from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES, Split, shapes_for_split


def _rng(op: str, idx: int, split: Split) -> random.Random:
    seed = int(hashlib.sha256(f"{op}|{idx}|{split}".encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def sample_task(op: str, idx: int, split: Split) -> dict:
    shape = _rng(op, idx, split).choice(shapes_for_split(split))
    return {"op": op, "shape": shape, "dtype": "float16", "split": split, "seed": idx}


_SORTED_TRAIN = tuple(sorted(TRAIN_SHAPES))


def l1_curriculum_pool(step: int, max_steps: int) -> tuple[int, ...]:
    """Return the active train-shape pool for early GRPO steps.

    The branch curriculum used small synthetic M/N grids. Current Protean v1 uses
    one-dimensional kernel shapes, so the same idea is applied as a prefix over
    the frozen train split: easiest shapes first, then the full train pool.
    Held-out shapes are never included in this pool.
    """

    if max_steps <= 0 or not _SORTED_TRAIN:
        return _SORTED_TRAIN
    frac = step / max_steps
    if frac < 0.15:
        return _SORTED_TRAIN[: max(1, min(2, len(_SORTED_TRAIN)))]
    if frac < 0.35:
        return _SORTED_TRAIN[: max(1, min(4, len(_SORTED_TRAIN)))]
    return _SORTED_TRAIN


def sample_shape_curriculum(
    op: str,
    split: Split,
    seed: int,
    *,
    step: int = 0,
    max_steps: int = 150,
) -> int:
    """Sample a shape with curriculum on train and full moat on held-out."""

    pool = l1_curriculum_pool(step, max_steps) if split == "train" else HELD_OUT_SHAPES
    return _rng(op, seed, split).choice(pool)


def sample_task_curriculum(
    op: str,
    idx: int,
    split: Split,
    *,
    step: int = 0,
    max_steps: int = 150,
) -> dict:
    shape = sample_shape_curriculum(op, split, idx, step=step, max_steps=max_steps)
    return {"op": op, "shape": shape, "dtype": "float16", "split": split, "seed": idx}
