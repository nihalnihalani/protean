"""Tests for the L1 shape curriculum (Step 10).

Validates:
  - Pool expansion schedule matches the 3-tier plan
  - Held-out / test splits always use full TEST_M (moat invariant)
  - Edge cases: step=0, step=max, max_steps=0
  - curriculum-aware sampling produces shapes within the active pool
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from protean.sampler import l1_curriculum_pool, sample_shape_curriculum, _SORTED_TRAIN
from protean.splits import TRAIN_M, TEST_M


class TestL1CurriculumPool:
    def test_l1_returns_2_smallest_at_step_0(self):
        pool = l1_curriculum_pool(0, 150)
        assert len(pool) == 2
        assert pool == _SORTED_TRAIN[:2]
        # Should be (256, 320) given the current TRAIN_M
        assert 256 in pool
        assert 320 in pool

    def test_l2_returns_4_at_step_30(self):
        # step 30 / 150 = 0.20, which is in [0.15, 0.35) → L2
        pool = l1_curriculum_pool(30, 150)
        assert len(pool) == 4
        assert pool == _SORTED_TRAIN[:4]

    def test_l3_returns_full_at_step_60(self):
        # step 60 / 150 = 0.40, which is >= 0.35 → L3 (full)
        pool = l1_curriculum_pool(60, 150)
        assert len(pool) == len(TRAIN_M)
        assert set(pool) == set(TRAIN_M)

    def test_boundary_l1_to_l2(self):
        # At exactly 15% boundary
        pool_before = l1_curriculum_pool(22, 150)  # 22/150 = 0.1467 < 0.15 → L1
        pool_at = l1_curriculum_pool(23, 150)       # 23/150 = 0.1533 >= 0.15 → L2
        assert len(pool_before) == 2
        assert len(pool_at) == 4

    def test_boundary_l2_to_l3(self):
        # At exactly 35% boundary
        pool_before = l1_curriculum_pool(52, 150)  # 52/150 = 0.3467 < 0.35 → L2
        pool_at = l1_curriculum_pool(53, 150)       # 53/150 = 0.3533 >= 0.35 → L3
        assert len(pool_before) == 4
        assert len(pool_at) == len(TRAIN_M)

    def test_step_equals_max_returns_full(self):
        pool = l1_curriculum_pool(150, 150)
        assert set(pool) == set(TRAIN_M)

    def test_max_steps_zero_returns_full(self):
        # Safety: if max_steps is 0 or negative, don't crash — return full pool
        pool = l1_curriculum_pool(0, 0)
        assert set(pool) == set(TRAIN_M)
        pool_neg = l1_curriculum_pool(0, -1)
        assert set(pool_neg) == set(TRAIN_M)

    def test_sorted_train_is_sorted(self):
        assert list(_SORTED_TRAIN) == sorted(TRAIN_M)


class TestSampleShapeCurriculum:
    def test_train_shapes_in_l1_pool_at_step_0(self):
        pool = l1_curriculum_pool(0, 150)
        for seed in range(20):
            M, N = sample_shape_curriculum("elementwise_add_relu", "train", seed, step=0, max_steps=150)
            assert M in pool, f"M={M} not in L1 pool {pool}"
            assert N in pool, f"N={N} not in L1 pool {pool}"

    def test_held_out_shapes_always_full_test_m(self):
        """Moat invariant: held-out eval always uses the full TEST_M pool."""
        for step in (0, 10, 50, 100, 150):
            for seed in range(10):
                M, N = sample_shape_curriculum("elementwise_add_relu", "held_out", seed, step=step, max_steps=150)
                assert M in TEST_M, f"Held-out M={M} not in TEST_M at step {step}"
                assert N in TEST_M, f"Held-out N={N} not in TEST_M at step {step}"

    def test_train_shapes_expand_over_time(self):
        """Shapes available at L3 should be a superset of those available at L1."""
        seen_l1 = set()
        seen_l3 = set()
        for seed in range(100):
            M1, _ = sample_shape_curriculum("elementwise_add_relu", "train", seed, step=0, max_steps=150)
            M3, _ = sample_shape_curriculum("elementwise_add_relu", "train", seed, step=150, max_steps=150)
            seen_l1.add(M1)
            seen_l3.add(M3)
        # L1 shapes should be a subset of L3 shapes
        assert seen_l1.issubset(seen_l3)
        # L3 should have strictly more shapes than L1 (since TRAIN_M has 6 entries, L1 has 2)
        assert len(seen_l3) > len(seen_l1)

    def test_disjoint_invariant_preserved(self):
        """Curriculum must never produce shapes from TEST_M for training."""
        for step in (0, 22, 30, 52, 53, 100, 150):
            pool = l1_curriculum_pool(step, 150)
            assert set(pool).isdisjoint(set(TEST_M)), \
                f"Curriculum pool {pool} overlaps with TEST_M {TEST_M} at step {step}"
