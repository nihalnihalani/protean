from protean.splits import TRAIN_SHAPES, HELD_OUT_SHAPES, assert_disjoint

def test_disjoint():
    assert set(TRAIN_SHAPES).isdisjoint(set(HELD_OUT_SHAPES))

def test_no_leak():
    # Calling assert_disjoint should pass without raising AssertionError
    assert_disjoint()
