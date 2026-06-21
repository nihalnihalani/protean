from dataclasses import replace

from protean.rewards import DEFAULT_CONFIG, compute_reward


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


def test_correct_faster_kernels_get_higher_rewards():
    base = dict(correct=True, launches_timed=1, dtype_ok=True, shape_ok=True, split="held_out")
    slow = compute_reward(**base, speedup=1.2)
    medium = compute_reward(**base, speedup=2.0)
    fast = compute_reward(**base, speedup=6.0)

    assert slow["reward"] < medium["reward"] < fast["reward"]
    assert slow["speedup_score"] < medium["speedup_score"] < fast["speedup_score"]


def test_speedup_reward_is_capped_for_timing_outliers():
    base = dict(correct=True, launches_timed=1, dtype_ok=True, shape_ok=True, split="held_out")
    capped = compute_reward(**base, speedup=20.0)
    outlier = compute_reward(**base, speedup=2000.0)

    assert capped["reward"] == outlier["reward"]
    assert capped["reward"] <= 2.0


def _correct_kwargs(**overrides):
    base = dict(
        correct=True,
        speedup=4.0,
        launches_timed=1,
        dtype_ok=True,
        shape_ok=True,
        split="held_out",
    )
    base.update(overrides)
    return base


def test_default_pr_mode_is_bonus_and_matches_legacy_value():
    cfg = DEFAULT_CONFIG
    assert cfg.pr_mode == "bonus"
    grade = compute_reward(**_correct_kwargs(pr_frac=0.5), config=cfg)
    # Legacy additive term: pr_bonus * pr_frac.
    assert grade["pr_reward"] == round(cfg.pr_bonus * 0.5, 6)
    assert grade["pr_mode"] == "bonus"


def test_multiplicative_mode_folds_pr_into_speedup_and_adds_no_separate_term():
    cfg = replace(DEFAULT_CONFIG, pr_mode="multiplicative")
    high_pr = compute_reward(**_correct_kwargs(pr_frac=0.9), config=cfg)
    low_pr = compute_reward(**_correct_kwargs(pr_frac=0.1), config=cfg)

    # No separate additive PR term in multiplicative mode.
    assert high_pr["pr_reward"] == 0.0
    assert low_pr["pr_reward"] == 0.0
    # Bottleneck kernels (high pr_frac) earn strictly more speedup credit.
    assert high_pr["speedup_reward"] > low_pr["speedup_reward"]
    assert high_pr["reward"] > low_pr["reward"]


def test_multiplicative_mode_respects_max_reward_bound():
    cfg = replace(DEFAULT_CONFIG, pr_mode="multiplicative")
    grade = compute_reward(**_correct_kwargs(speedup=2000.0, pr_frac=1.0), config=cfg)
    assert grade["reward"] <= cfg.max_reward


def test_multiplicative_mode_keeps_informative_speedup_gradient():
    # The weight re-normalisation must keep the speedup gradient strictly
    # positive across the full speedup range even at max pr_frac, i.e. the
    # max_reward clamp must NOT flatten the top of the range.
    cfg = replace(DEFAULT_CONFIG, pr_mode="multiplicative")
    speeds = [1.5, 2.0, 4.0, 8.0, 16.0, 20.0]
    rewards = [
        compute_reward(**_correct_kwargs(speedup=s, pr_frac=1.0), config=cfg)["reward"]
        for s in speeds
    ]
    for lo, hi in zip(rewards, rewards[1:]):
        assert hi > lo
    # And the top of the range stays at or under the bound (no clamp plateau).
    assert rewards[-1] <= cfg.max_reward


def test_multiplicative_max_credit_matches_bonus_ceiling():
    # Re-normalisation pins the max pre-clamp speedup credit to the same ceiling
    # as the legacy "bonus" weight, so a fully-bottlenecked top-speedup kernel
    # does not get silently clamped.
    bonus_cfg = replace(DEFAULT_CONFIG, pr_mode="bonus")
    mult_cfg = replace(DEFAULT_CONFIG, pr_mode="multiplicative")
    bonus = compute_reward(**_correct_kwargs(speedup=20.0, pr_frac=1.0), config=bonus_cfg)
    mult = compute_reward(**_correct_kwargs(speedup=20.0, pr_frac=1.0), config=mult_cfg)
    # At pr_frac=1.0 the multiplicative scale is (1+1)/(1+1) == 1, so the
    # speedup credit equals the bonus-mode speedup credit exactly.
    assert mult["speedup_reward"] == bonus["speedup_reward"]


def test_centered_mode_is_gradient_neutral_about_pr_center():
    cfg = replace(DEFAULT_CONFIG, pr_mode="centered", pr_center=0.5)
    above = compute_reward(**_correct_kwargs(pr_frac=0.8), config=cfg)
    below = compute_reward(**_correct_kwargs(pr_frac=0.2), config=cfg)
    at_center = compute_reward(**_correct_kwargs(pr_frac=0.5), config=cfg)

    # Symmetric pr_frac around the center yield opposite, zero-summing bonuses.
    assert at_center["pr_reward"] == 0.0
    assert above["pr_reward"] == -below["pr_reward"]
    assert above["pr_reward"] > 0.0 > below["pr_reward"]


def test_centered_mode_does_not_change_default_max_bound():
    cfg = replace(DEFAULT_CONFIG, pr_mode="centered")
    grade = compute_reward(**_correct_kwargs(speedup=2000.0, pr_frac=1.0), config=cfg)
    assert grade["reward"] <= cfg.max_reward


def test_centered_mode_below_center_kernel_reward_is_non_negative():
    # A correct kernel just under speedup_floor (no speedup credit) with
    # pr_frac=0 in centered mode subtracts pr_bonus*pr_center from the floor.
    # The lower-bound clamp must keep the total reward non-negative.
    cfg = replace(DEFAULT_CONFIG, pr_mode="centered", pr_center=0.5)
    grade = compute_reward(
        correct=True,
        speedup=1.0,  # below speedup_floor -> speedup_reward == 0
        launches_timed=1,
        dtype_ok=True,
        shape_ok=True,
        split="train",
        pr_frac=0.0,
        config=cfg,
    )
    assert grade["pr_reward"] < 0.0
    assert grade["reward"] >= 0.0


def test_lower_bound_clamp_does_not_change_bonus_mode_values():
    # The clamp is a no-op for bonus mode (contributions are always >= 0).
    cfg = replace(DEFAULT_CONFIG, pr_mode="bonus")
    grade = compute_reward(**_correct_kwargs(speedup=4.0, pr_frac=0.3), config=cfg)
    expected = (
        grade["correctness_reward"] + grade["speedup_reward"] + grade["pr_reward"]
    )
    assert grade["reward"] == round(min(expected, cfg.max_reward), 6)


def test_pr_modes_do_not_leak_reward_on_hard_failure():
    for mode in ("bonus", "multiplicative", "centered"):
        cfg = replace(DEFAULT_CONFIG, pr_mode=mode)
        grade = compute_reward(
            correct=False,
            speedup=10.0,
            launches_timed=1,
            dtype_ok=True,
            shape_ok=True,
            split="held_out",
            pr_frac=1.0,
            config=cfg,
        )
        assert grade["reward"] == 0.0
        assert grade["pr_reward"] == 0.0
