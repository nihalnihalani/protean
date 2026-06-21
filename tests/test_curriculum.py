from protean.sampler import _SORTED_TRAIN, l1_curriculum_pool, sample_shape_curriculum
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


def test_curriculum_starts_with_prefix_train_pool():
    pool = l1_curriculum_pool(0, 150)
    assert pool == _SORTED_TRAIN[: min(2, len(_SORTED_TRAIN))]
    assert set(pool).issubset(TRAIN_SHAPES)


def test_curriculum_expands_to_full_train_pool():
    assert set(l1_curriculum_pool(150, 150)) == set(TRAIN_SHAPES)
    assert set(l1_curriculum_pool(0, 0)) == set(TRAIN_SHAPES)


def test_held_out_sampling_never_uses_curriculum_pool():
    for step in (0, 25, 75, 150):
        for seed in range(10):
            assert sample_shape_curriculum("elementwise_add_relu", "held_out", seed, step=step) in HELD_OUT_SHAPES


def test_train_curriculum_never_contains_held_out_shapes():
    for step in (0, 23, 53, 150):
        assert set(l1_curriculum_pool(step, 150)).isdisjoint(HELD_OUT_SHAPES)
