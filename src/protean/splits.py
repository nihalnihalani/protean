"""Frozen train and held-out shape split for the verifier-first MVP."""

from __future__ import annotations

import random
from typing import Literal

Split = Literal["train", "held_out"]

TRAIN_SHAPES = (1024, 2048, 4096)
# Dev held-out shapes are deliberately OFF the tiling grid (mod 64 != 0) and prime-adjacent so that a
# kernel which hardcodes BLOCK=64/128/256/512 cannot tile them evenly. This defeats the
# tiling-memorisation failure mode (G1): 1535 = 5*307 (mod 64 = 63), 3073 prime (mod 64 = 1),
# 6143 prime (mod 64 = 63). They also span the interpolation (1535, 3073) and extrapolation (6143)
# bands relative to TRAIN_SHAPES. See docs/GAP_ANALYSIS.md G1.
HELD_OUT_SHAPES = (1535, 3073, 6143)


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

N_OPS = 5  # spread the claim over >=5 ops (clustering-immune across-op sign test)
N_HELDOUT_PER_OP = 40  # 5 x 40 = 200 paired held-out tasks (was 3)

# The ops that are ACTUALLY IMPLEMENTED (have a kernel, grader, and task dir). N_OPS=5 is the
# clustering-immune TARGET width for the across-op sign test; REAL_OPS is the subset that exists
# today (3). Any path that touches the real grader (powered_eval_from_run_dir, paired_sources_report)
# must use REAL_OPS so it cannot crash on a non-existent op (see task_catalog.OPS). The synthetic
# demo report also uses REAL_OPS so its GPU regen command does not reference fictional ops.
REAL_OPS = ("elementwise_add_relu", "rmsnorm", "softmax_rows")
TILING_BLOCK = 64  # continuous draws must be off this grid too (structural novelty)

_INTERP_BAND = (min(TRAIN_SHAPES) + 1, max(TRAIN_SHAPES) - 1)  # interpolate between train points
_EXTRAP_BAND = (max(TRAIN_SHAPES) + 1, 4 * max(TRAIN_SHAPES))  # extrapolate beyond max train
_INTERP_FRACTION = 0.6
# Structural anchors always graded (tiling-boundary + prime-adjacent). These are a PROPER SUPERSET of
# the dev HELD_OUT_SHAPES: the three dev shapes PLUS additional off-grid near-block-boundary shapes that
# exercise off-by-one tiling errors (block sizes 64/128/256). HELD_OUT_SHAPES is referenced (not copied)
# so this constant cannot silently diverge if the dev shapes change. _EXTRA_ANCHORS are all off the
# train grid AND off the tiling grid (mod 64 != 0), and are NOT already in HELD_OUT_SHAPES.
_EXTRA_ANCHORS = (65, 129, 257, 1537, 3079)  # near 64/128/256 boundaries + prime-adjacent off-grid
STRUCTURAL_ANCHORS = tuple(dict.fromkeys(tuple(HELD_OUT_SHAPES) + _EXTRA_ANCHORS))


def is_offgrid(m: int) -> bool:
    """Held-out-eligible iff off the training grid AND off the tiling grid."""
    return (m not in TRAIN_SHAPES) and (m % TILING_BLOCK != 0)


def sample_heldout_shape(rng: random.Random) -> int:
    """Draw one continuous off-grid held-out size. `rng` is a caller-seeded random.Random (determinism)."""
    band = _INTERP_BAND if rng.random() < _INTERP_FRACTION else _EXTRAP_BAND
    while True:
        m = rng.randint(band[0], band[1])
        if is_offgrid(m):
            return m


def _assert_powered_split() -> None:
    assert set(STRUCTURAL_ANCHORS).isdisjoint(set(TRAIN_SHAPES)), "anchor on the train grid — moat regression"
    assert _INTERP_BAND[0] > min(TRAIN_SHAPES) and _EXTRAP_BAND[0] > max(TRAIN_SHAPES)
    # Dev held-out shapes must themselves be off-grid (mod TILING_BLOCK != 0) — this is the G1 moat.
    assert all(is_offgrid(m) for m in HELD_OUT_SHAPES), "dev held-out shape on the tiling grid — G1 regression"
    # STRUCTURAL_ANCHORS must be a PROPER superset of HELD_OUT_SHAPES (carries info beyond the dev shapes),
    # have no duplicates, and every anchor must itself be off-grid. (65/129/257 are mod 64 != 0.)
    assert len(STRUCTURAL_ANCHORS) == len(set(STRUCTURAL_ANCHORS)), "duplicate structural anchor"
    assert set(HELD_OUT_SHAPES).issubset(STRUCTURAL_ANCHORS), "anchors must include the dev held-out shapes"
    assert len(STRUCTURAL_ANCHORS) > len(HELD_OUT_SHAPES), "anchors must add coverage beyond HELD_OUT_SHAPES"
    assert all(m % TILING_BLOCK != 0 for m in STRUCTURAL_ANCHORS), "structural anchor on the tiling grid"


_assert_powered_split()
