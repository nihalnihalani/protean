"""Thin convenience wrapper re-exporting the split + sampler API (the generalization moat).

The real logic lives in:
  - splits.py  : the FROZEN train/held-out invariant (disjoint-by-construction), asserted at import
  - sampler.py : deterministic continuous-shape draws (sha256 of a canonical string, NOT builtin hash())
See IMPLEMENTATION_PLAN.md §4.6.
"""
from .splits import TRAIN_BANDS, HELD_OUT_BANDS, assert_disjoint  # noqa: F401
from .sampler import sample_shape  # noqa: F401

# Invariant check runs at import: train and held-out shape bands must be provably disjoint.
assert_disjoint()
