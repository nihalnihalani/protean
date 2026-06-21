from protean.rewards import compute_reward


def test_incorrect_gets_zero_reward():
    grade = compute_reward(
        correct=False,
        speedup=10.0,
        launches_timed=1,
        dtype_ok=True,
        shape_ok=True,
        split="held_out",
    )
    assert grade["reward"] == 0.0
    assert "incorrect" in grade["caps"]


def test_slow_correct_reports_correctness_floor_only():
    grade = compute_reward(
        correct=True,
        speedup=1.0,
        launches_timed=1,
        dtype_ok=True,
        shape_ok=True,
        split="train",
    )
    assert grade["correct"] is True
    assert grade["reward"] == 0.3
    assert "below_speedup_floor" in grade["caps"]


def test_shape_mismatch_fails_before_reward():
    grade = compute_reward(
        correct=True,
        speedup=2.0,
        launches_timed=1,
        dtype_ok=True,
        shape_ok=False,
        split="held_out",
    )
    assert grade["reward"] == 0.0
    assert "shape_mismatch" in grade["caps"]
