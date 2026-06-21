from protean.splits import TRAIN_M, TEST_M, assert_disjoint

def test_disjoint():
    assert set(TRAIN_M).isdisjoint(set(TEST_M))

def test_no_leak():
    # Calling assert_disjoint should pass without raising AssertionError
    assert_disjoint()
