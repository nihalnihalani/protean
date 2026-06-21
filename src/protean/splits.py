"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# splits.py — THE moat invariant, frozen & asserted at import time
from dataclasses import dataclass

TRAIN_M = (256, 320, 512, 640, 1024, 2048)    # fixed discrete grid (with off-grid shapes for boundary-checks)
TEST_M  = (400, 800, 1600,                     # multiples of 100 — rare in GPU code
           383, 769,                           # off-by-one prime-adjacent — provably unusual
           3072)                               # crosses a tiling boundary (block flips) — structural disjointness

TRAIN_BANDS = TRAIN_M
HELD_OUT_BANDS = TEST_M

def assert_disjoint():
    assert set(TRAIN_M).isdisjoint(set(TEST_M)), "SPLIT REGRESSION — moat destroyed"
    # also assert no test shape is on the train grid
    assert all(m not in TRAIN_M for m in TEST_M)

assert_disjoint()                        # runs on import, in CI, AND inside freeze()
