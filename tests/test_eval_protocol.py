"""Powered held-out evaluation protocol (resolves the n=24 underpowering)."""
import random

from protean.eval_protocol import (
    build_eval_set, evaluate_policy, paired_report, mde_paired, required_n_paired, across_op_sign_test,
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
