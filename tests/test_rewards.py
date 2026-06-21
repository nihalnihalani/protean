from protean.rewards import compute_reward

def test_rewards_incorrect():
    res = compute_reward(correct=False, speedup=1.5, pr_frac=0.8, launches_timed=1, dtype_ok=True, shape_ok=True)
    assert res["reward"] == 0.0
    assert "incorrect" in res["caps"]

def test_rewards_no_launches():
    res = compute_reward(correct=True, speedup=1.5, pr_frac=0.8, launches_timed=0, dtype_ok=True, shape_ok=True)
    assert res["reward"] == 0.0
    assert "no_triton_launch" in res["caps"]

def test_rewards_slow():
    res = compute_reward(correct=True, speedup=1.0, pr_frac=0.6, launches_timed=1, dtype_ok=True, shape_ok=True)
    assert abs(res["reward"] - 0.42) < 1e-5

def test_rewards_fast():
    res = compute_reward(correct=True, speedup=1.5, pr_frac=0.6, launches_timed=1, dtype_ok=True, shape_ok=True)
    assert abs(res["reward"] - 1.42) < 1e-5
