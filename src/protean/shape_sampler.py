"""Thin convenience wrapper re-exporting the split + sampler API (the generalization moat).

The real logic lives in:
  - splits.py  : the FROZEN train/held-out invariant (disjoint-by-construction), asserted at import
  - sampler.py : deterministic shape draws (sha256 of a canonical string, NOT builtin hash())
See docs/TECHNICAL_SPEC.md §5 and §9.

FIX (devil's-advocate R2/R3): this file previously imported TRAIN_BANDS / HELD_OUT_BANDS /
assert_disjoint / sample_shape — names that do not exist in splits.py or sampler.py — which raised
ImportError at import time and meant the moat invariant never fired. Re-exporting the REAL names.
"""
from protean.splits import TRAIN_M, TEST_M, _assert_split_disjoint  # noqa: F401
from protean.sampler import sample_task  # noqa: F401

# Invariant check runs at import: train and held-out shape sets must be provably disjoint.
_assert_split_disjoint()
