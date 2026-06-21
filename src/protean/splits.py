"""Frozen train and held-out shape split for the verifier-first MVP."""

from __future__ import annotations

from typing import Literal

Split = Literal["train", "held_out"]

TRAIN_SHAPES = (1024, 2048, 4096)
HELD_OUT_SHAPES = (1536, 3072, 5632)


def assert_disjoint() -> None:
    overlap = set(TRAIN_SHAPES) & set(HELD_OUT_SHAPES)
    if overlap:
        raise AssertionError(f"train/held-out shape overlap: {sorted(overlap)}")


def shapes_for_split(split: Split) -> tuple[int, ...]:
    if split == "train":
        return TRAIN_SHAPES
    if split == "held_out":
        return HELD_OUT_SHAPES
    raise ValueError(f"unknown split: {split}")


assert_disjoint()
