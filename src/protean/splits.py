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


# ---------------------------------------------------------------------------
# Powered held-out EVALUATION sampler (fixes the n=24 underpowering, OPEN_ISSUES).
# The fixed TRAIN_SHAPES / HELD_OUT_SHAPES above are the verifier-first DEV shapes (3+3) used to
# smoke-test one delta. The statistically-powered generalization CLAIM uses many CONTINUOUS off-grid
# held-out shapes sampled below — see protean.eval_protocol + docs/TECHNICAL_SPEC.md §5.4.
# Purely additive: does not change the dev-shape API the MVP/tests rely on.

N_OPS = 5                       # spread the claim over >=5 ops (clustering-immune across-op sign test)
N_HELDOUT_PER_OP = 40           # 5 x 40 = 200 paired held-out tasks (was 3)
TILING_BLOCK = 64               # continuous draws must be off this grid too (structural novelty)

_INTERP_BAND = (min(TRAIN_SHAPES) + 1, max(TRAIN_SHAPES) - 1)    # interpolate between train points
_EXTRAP_BAND = (max(TRAIN_SHAPES) + 1, 4 * max(TRAIN_SHAPES))     # extrapolate beyond max train
_INTERP_FRACTION = 0.6
# Structural anchors always graded (tiling-boundary + prime-adjacent), incl. the dev held-out shapes.
STRUCTURAL_ANCHORS = tuple(HELD_OUT_SHAPES) + (1535, 3073, 6143)


def is_offgrid(m: int) -> bool:
    """Held-out-eligible iff off the training grid AND off the tiling grid."""
    return (m not in TRAIN_SHAPES) and (m % TILING_BLOCK != 0)


def sample_heldout_shape(rng) -> int:
    """Draw one continuous off-grid held-out size. `rng` is a caller-seeded random.Random (determinism)."""
    band = _INTERP_BAND if rng.random() < _INTERP_FRACTION else _EXTRAP_BAND
    while True:
        m = rng.randint(band[0], band[1])
        if is_offgrid(m):
            return m


def _assert_powered_split() -> None:
    assert set(STRUCTURAL_ANCHORS).isdisjoint(set(TRAIN_SHAPES)), "anchor on the train grid — moat regression"
    assert _INTERP_BAND[0] > min(TRAIN_SHAPES) and _EXTRAP_BAND[0] > max(TRAIN_SHAPES)


_assert_powered_split()
