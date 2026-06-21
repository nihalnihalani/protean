"""Powered held-out evaluation protocol (resolves the n=24 underpowering)."""
import random

import pytest

from protean.eval_protocol import (
    build_eval_set, evaluate_policy, paired_report, mde_paired, required_n_paired, across_op_sign_test,
    hierarchical_bootstrap_ci, power_at, power_curve, BCA_MIN_OPS, synthetic_powered_report,
)
from protean.splits import N_OPS, N_HELDOUT_PER_OP, TRAIN_SHAPES, is_offgrid


def test_eval_set_is_large_and_offgrid():
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    assert len(tasks) == N_OPS * N_HELDOUT_PER_OP == 200
    # every held-out shape is off the train grid AND off the tiling grid
    assert all(is_offgrid(t["shape"]) for t in tasks)
    assert all(t["shape"] not in TRAIN_SHAPES for t in tasks)
    # deterministic
    assert [t["shape"] for t in tasks] == [t["shape"] for t in build_eval_set(ops)]


def test_power_improves_with_n():
    assert mde_paired(3) > mde_paired(200)          # bigger n -> smaller detectable effect
    assert mde_paired(200) < 0.25                    # n=200 detects small-medium effects at 80% power
    assert required_n_paired(0.4) == 50              # paired n for a medium standardized effect


def test_sign_test_is_clustering_immune():
    assert abs(across_op_sign_test({f"o{i}": 1.0 for i in range(5)})["p_one_sided"] - (1 / 2) ** 5) < 1e-9
    assert across_op_sign_test({"a": 1.0, "b": -1.0})["p_one_sided"] > 0.05


def test_paired_report_detects_a_real_effect():
    rng = random.Random(11)
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    off = {op: rng.gauss(0, 0.15) for op in ops}
    base = evaluate_policy(lambda t: max(0.0, 0.3 + off[t["op"]] + rng.gauss(0, 0.25)), tasks)
    trained = evaluate_policy(lambda t: max(0.0, 0.3 + 0.18 + off[t["op"]] + rng.gauss(0, 0.25)), tasks)
    rep = paired_report(base, trained, B=2000)
    assert rep["n_tasks"] == 200 and rep["n_ops"] == 5
    assert rep["gap_ci_excludes_zero"] is True
    assert rep["sign_test"]["p_one_sided"] <= 0.05
    assert rep["powered"] is True


def test_paired_report_null_effect_not_significant():
    rng = random.Random(3)
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    g = lambda t: max(0.0, 0.3 + rng.gauss(0, 0.25))  # noqa: E731  same dist for both -> no true effect
    base = evaluate_policy(g, tasks)
    trained = evaluate_policy(g, tasks)
    rep = paired_report(base, trained, B=2000)
    # no real effect -> CI should contain 0 (not powered on the CI axis)
    assert rep["gap_ci_excludes_zero"] is False


# ---- BCa bootstrap option ----------------------------------------------------
def _deltas_by_op(n_ops, per_op=40, mean=0.18, sd=0.25, seed=7):
    rng = random.Random(seed)
    off = {f"op{i}": rng.gauss(0, 0.05) for i in range(n_ops)}
    return {op: [mean + off[op] + rng.gauss(0, sd) for _ in range(per_op)] for op in off}


def test_default_method_is_percentile_and_backward_compatible():
    d = _deltas_by_op(N_OPS)
    res = hierarchical_bootstrap_ci(d, B=2000, seed=0)
    assert res["ci_method"] == "percentile"
    # default call must be byte-identical to an explicit percentile request (no behavior change)
    res2 = hierarchical_bootstrap_ci(d, B=2000, seed=0, method="percentile")
    assert (res["ci_lo"], res["ci_hi"], res["gap"]) == (res2["ci_lo"], res2["ci_hi"], res2["gap"])


def test_bca_runs_with_enough_ops_and_brackets_point():
    d = _deltas_by_op(N_OPS)  # 5 ops == BCA_MIN_OPS -> BCa allowed
    res = hierarchical_bootstrap_ci(d, B=3000, seed=0, method="bca")
    assert res["ci_method"] == "bca"
    assert res["ci_lo"] <= res["gap"] <= res["ci_hi"]
    # for a clear positive effect the BCa CI should exclude zero
    assert res["excludes_zero"] is True


def test_bca_upper_bound_never_returns_lone_extreme_under_heavy_bias():
    # Build deltas where the grand mean sits well ABOVE most of the data so the bootstrap distribution is
    # heavily left-skewed: the fraction of replicates below the point estimate is large -> z0 > 1.28, the
    # regime where the BCa upper-quantile adjustment is non-trivial and the off-by-one index bug bit.
    # One op carries a large positive mean (drives the grand mean up), the rest cluster near zero.
    d = {
        "hot": [5.0] * 40,
        "a": [0.0] * 40, "b": [0.0] * 40, "c": [0.0] * 40, "d": [0.0] * 40,
    }
    res = hierarchical_bootstrap_ci(d, B=3000, seed=0, method="bca")
    assert res["ci_method"] == "bca"
    # interval must be well-formed and bracket the point estimate
    assert res["ci_lo"] <= res["gap"] <= res["ci_hi"]
    # the upper bound must be a valid interior quantile, NOT the single most extreme bootstrap replicate.
    # The max possible grand mean (all 5 ops resampled to "hot") is 5.0; the all-hot replicate is the lone
    # extreme. A correct hi index stays strictly below it.
    assert res["ci_hi"] < 5.0


def test_bca_and_percentile_share_quantile_convention_on_symmetric_data():
    # On near-symmetric data with no bias (z0 ~ 0) and tiny acceleration, BCa quantiles approach the
    # nominal percentile quantiles, so the two intervals should be close -- confirming the unified index
    # convention (no systematic one-rank gap between the branches).
    d = _deltas_by_op(N_OPS)
    pct = hierarchical_bootstrap_ci(d, B=4000, seed=1, method="percentile")
    bca = hierarchical_bootstrap_ci(d, B=4000, seed=1, method="bca")
    assert bca["ci_method"] == "bca"
    spread = pct["ci_hi"] - pct["ci_lo"]
    assert abs(bca["ci_hi"] - pct["ci_hi"]) < 0.5 * spread
    assert abs(bca["ci_lo"] - pct["ci_lo"]) < 0.5 * spread


def test_bca_falls_back_to_percentile_below_min_ops():
    d = _deltas_by_op(BCA_MIN_OPS - 1)  # too few ops for a trustworthy jackknife a_hat
    res = hierarchical_bootstrap_ci(d, B=2000, seed=0, method="bca")
    assert res["ci_method"] == "percentile"  # auto fallback recorded honestly


def test_bca_invalid_method_raises():
    with pytest.raises(ValueError):
        hierarchical_bootstrap_ci(_deltas_by_op(N_OPS), B=200, seed=0, method="nope")


def test_paired_report_threads_bca_method():
    rng = random.Random(11)
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    off = {op: rng.gauss(0, 0.15) for op in ops}
    base = evaluate_policy(lambda t: max(0.0, 0.3 + off[t["op"]] + rng.gauss(0, 0.25)), tasks)
    trained = evaluate_policy(lambda t: max(0.0, 0.3 + 0.18 + off[t["op"]] + rng.gauss(0, 0.25)), tasks)
    rep = paired_report(base, trained, B=2000, ci_method="bca")
    assert rep["ci_method"] == "bca"
    assert rep["n_ops"] == N_OPS
    # default report still reports percentile
    rep_def = paired_report(base, trained, B=2000)
    assert rep_def["ci_method"] == "percentile"


# ---- Common-Random-Numbers (CRN) paired variance reduction -------------------
def test_crn_default_off_uses_single_arg_grader():
    tasks = build_eval_set([f"op{i}" for i in range(N_OPS)])
    calls = []
    out = evaluate_policy(lambda t: (calls.append(t), 1.0)[1], tasks, rollouts=3)
    assert len(out) == N_OPS * N_HELDOUT_PER_OP
    assert len(calls) == len(out) * 3  # 3 rollouts each, single-arg path


def test_crn_passes_rollout_seed_and_pairs_base_trained():
    tasks = build_eval_set([f"op{i}" for i in range(N_OPS)])
    seen_base, seen_trained = {}, {}

    def grader(store):
        def g(t, rollout_seed):
            store.setdefault((t["op"], t["idx"]), []).append(rollout_seed)
            return float(rollout_seed % 1000)
        return g

    base = evaluate_policy(grader(seen_base), tasks, rollouts=4, crn_seed=99)
    trained = evaluate_policy(grader(seen_trained), tasks, rollouts=4, crn_seed=99)
    # CRN: base and trained see the IDENTICAL rollout seeds per (op, idx, k) -> cancels shared noise
    assert seen_base == seen_trained
    # seeds are distinct across rollout index k within a task (real common-random-numbers, not a constant)
    any_key = next(iter(seen_base))
    assert len(set(seen_base[any_key])) == 4
    # identical seeds -> identical per-task mean reward -> exactly zero paired delta
    assert base == trained


def test_crn_variance_reduction_lowers_paired_delta_variance():
    tasks = build_eval_set([f"op{i}" for i in range(N_OPS)], seed=0)

    # Stochastic grader: shared input noise (seeded) + a fixed treatment effect for trained.
    def make(effect):
        def g(t, rollout_seed):
            r = random.Random(rollout_seed)
            return 0.4 + effect + r.gauss(0, 0.5)
        return g

    base = evaluate_policy(make(0.0), tasks, rollouts=8, crn_seed=2024)
    trained = evaluate_policy(make(0.1), tasks, rollouts=8, crn_seed=2024)
    crn_deltas = [trained[k] - base[k] for k in base]

    # Independent seeds (no CRN pairing) for the same generative process. Each call draws from a LOCAL
    # deterministic RNG keyed by (op, idx, tag, call-counter) -- no dependence on the global random state,
    # so the test result is invariant to pytest ordering / prior tests leaving the global RNG dirty.
    def make_indep(effect, tag):
        counter = [0]

        def g(t):
            counter[0] += 1
            r = random.Random(hash((t["op"], t["idx"], tag, counter[0])))
            return 0.4 + effect + r.gauss(0, 0.5)
        return g

    base_i = evaluate_policy(make_indep(0.0, "b"), tasks, rollouts=8)
    trained_i = evaluate_policy(make_indep(0.1, "t"), tasks, rollouts=8)
    indep_deltas = [trained_i[k] - base_i[k] for k in base_i]

    def var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    # Pairing on common random numbers must not increase variance of the paired delta; here it reduces it.
    assert var(crn_deltas) < var(indep_deltas)


# ---- power curve helper ------------------------------------------------------
def test_power_at_monotone_and_bounds():
    assert 0.0 <= power_at(3, 0.18) <= 1.0
    # power increases with n for a fixed effect
    assert power_at(200, 0.2) > power_at(50, 0.2) > power_at(10, 0.2)
    # power increases with effect size for fixed n
    assert power_at(50, 0.4) > power_at(50, 0.1)


def test_power_at_matches_required_n_definition():
    # at n = required_n_paired(d_z), achieved power should be ~the target (0.80)
    d_z = 0.4
    n = required_n_paired(d_z)
    assert abs(power_at(n, d_z) - 0.80) < 0.03


def test_power_curve_shape_and_target():
    pc = power_curve(0.3, ns=[3, 50, 200])
    assert pc["d_z"] == 0.3 and pc["target_power"] == 0.80
    assert pc["n_for_target"] == required_n_paired(0.3)
    ns = [n for n, _ in pc["curve"]]
    powers = [p for _, p in pc["curve"]]
    assert ns == [3, 50, 200]
    assert powers == sorted(powers)  # monotone non-decreasing in n
    assert all(0.0 <= p <= 1.0 for p in powers)


# ---- sign_test_advisory verdict-contract field -------------------------------
def test_synthetic_report_marks_sign_test_advisory_for_real_ops():
    # The real-op artifact has K=3 ops < N_OPS=5, so the across-op sign test is ADVISORY (the CI carries
    # the claim). This field is load-bearing in FIGURES.md / the demo narrative; pin it in the suite.
    rep = synthetic_powered_report(seed=1234)
    assert rep["n_ops"] < N_OPS
    assert rep["sign_test_advisory"] is True


def test_sign_test_advisory_is_false_at_or_above_n_ops():
    # With exactly N_OPS synthetic ops the sign test is NOT advisory (the across-op claim is supportable).
    ops = [f"op{i}" for i in range(N_OPS)]
    rep = synthetic_powered_report(ops=ops, seed=7)
    assert rep["n_ops"] == N_OPS
    assert rep["sign_test_advisory"] is False
