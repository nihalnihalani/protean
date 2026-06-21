"""G1b: powered held-out eval is wired into a usable, CPU-safe path + committed demo artifact.

All tests here are CPU-only: they exercise the SYNTHETIC path and the committed artifact.
The real grader (GPU) path is covered by the runbook (docs/RUNBOOK_POWERED_EVAL.md), not by tests.

The synthetic report uses ONLY the ops that actually exist (splits.REAL_OPS). It must NOT invent ops
like layernorm/gelu, because the GPU regen command grades the same op list through the real grader and
would raise ValueError('unknown op: ...') on a fictional op. With K=3 real ops the across-op sign test
cannot reach p<=0.05 (best is 3/3 -> p=0.125), so the per-task hierarchical-bootstrap CI carries the
power and sign_test_advisory is True. A synthetic report is never a REAL powered result: powered_real
is always False for synthetic, and cuda_unavailable is None (inapplicable, no GPU path was exercised).
"""

import json
import math
from pathlib import Path

import pytest

from protean.eval_protocol import powered_eval_from_run_dir, synthetic_powered_report
from protean.splits import (
    HELD_OUT_SHAPES,
    N_HELDOUT_PER_OP,
    REAL_OPS,
    STRUCTURAL_ANCHORS,
    TILING_BLOCK,
)
from protean.task_catalog import OPS_BY_NAME

_REPO = Path(__file__).resolve().parents[1]
_ARTIFACT = _REPO / "demo" / "powered-eval-200.json"


def test_real_ops_all_exist_in_the_catalog():
    # The synthetic default ops must be a subset of the implemented ops (no fictional layernorm/gelu).
    for op in REAL_OPS:
        assert op in OPS_BY_NAME, f"REAL_OPS contains an op with no implementation: {op}"
    assert len(REAL_OPS) == 3


def test_synthetic_report_is_shaped_and_deterministic():
    rep = synthetic_powered_report(B=2000)
    assert rep["synthetic"] is True
    assert rep["powered_real"] is False  # synthetic is never a real powered result
    assert rep["cuda_unavailable"] is None  # inapplicable for a no-GPU report
    # defaults to the REAL ops only
    assert rep["ops"] == list(REAL_OPS)
    assert rep["n_ops"] == len(REAL_OPS) == 3
    assert rep["n_tasks"] == len(REAL_OPS) * N_HELDOUT_PER_OP == 120
    # same machinery as the real path: CI, sign test, power all present
    assert set(rep) >= {"gap_mean", "ci95", "gap_ci_excludes_zero", "sign_test", "powered", "per_op_delta"}
    # deterministic given the seed (deltas AND the bootstrap CI, since seed threads into boot_seed)
    again = synthetic_powered_report(B=2000)
    assert again["gap_mean"] == rep["gap_mean"]
    assert again["per_op_delta"] == rep["per_op_delta"]
    assert again["ci95"] == rep["ci95"]


def test_seed_threads_into_the_bootstrap_ci():
    # Different seeds must give different CI bands (the seed controls deltas AND bootstrap resampling).
    a = synthetic_powered_report(seed=1, B=2000)
    b = synthetic_powered_report(seed=2, B=2000)
    assert a["ci95"] != b["ci95"]


def test_synthetic_powered_via_ci_not_sign_test_at_three_ops():
    rep = synthetic_powered_report(effect=0.22, B=4000)
    assert rep["gap_mean"] > 0.0
    # the CI carries the power at K=3
    assert rep["gap_ci_excludes_zero"] is True
    assert rep["powered"] is True
    # all real ops improve, but K=3 -> p=0.125 (NOT <=0.05): sign test is advisory, not the claim
    assert rep["sign_test"]["positive"] == 3
    assert math.isclose(rep["sign_test"]["p_one_sided"], 0.125, rel_tol=1e-9)
    assert rep["sign_test_advisory"] is True
    # honest: even when statistically "powered" on synthetic data, it is never a REAL powered result
    assert rep["powered_real"] is False


def test_synthetic_null_effect_is_not_powered_on_ci():
    rep = synthetic_powered_report(effect=0.0, B=4000)
    assert rep["gap_ci_excludes_zero"] is False
    assert rep["powered_real"] is False


def test_sign_test_advisory_flag_for_few_ops():
    rep = synthetic_powered_report(ops=["elementwise_add_relu", "rmsnorm"], B=2000)
    assert rep["n_ops"] == 2
    assert rep["sign_test_advisory"] is True  # K<5 -> advisory


def test_demo_artifact_exists_and_is_honestly_labeled():
    assert _ARTIFACT.exists(), "demo/powered-eval-200.json must be committed (TECHNICAL_SPEC §10)"
    rep = json.loads(_ARTIFACT.read_text())
    assert rep["synthetic"] is True, "committed artifact must be clearly labeled synthetic"
    # a reader who checks ANY honest top-level field sees it is not a real measured result
    assert rep["powered_real"] is False
    assert rep["cuda_unavailable"] is None
    # only real ops, no invented layernorm/gelu
    assert rep["ops"] == list(REAL_OPS)
    assert all(op in OPS_BY_NAME for op in rep["ops"])
    assert rep["n_ops"] == 3
    assert rep["n_tasks"] == 3 * N_HELDOUT_PER_OP == 120
    assert rep["sign_test_advisory"] is True
    assert "note" in rep and "SYNTHETIC" in rep["note"]


def test_demo_artifact_regenerates_byte_identical():
    # The committed artifact must match a fresh deterministic regeneration (default seed/effect).
    fresh = synthetic_powered_report()
    committed = json.loads(_ARTIFACT.read_text())
    assert math.isclose(fresh["gap_mean"], committed["gap_mean"], rel_tol=0, abs_tol=1e-12)
    assert fresh["per_op_delta"] == committed["per_op_delta"]


def test_run_dir_helper_rejects_unknown_ops_without_a_gpu():
    # The guard fires before any grader/GPU work, so this is CPU-safe.
    with pytest.raises(ValueError, match="unknown op"):
        powered_eval_from_run_dir("/tmp/does-not-matter", ["layernorm"])
    with pytest.raises(ValueError, match="unknown op"):
        powered_eval_from_run_dir("/tmp/does-not-matter", ["gelu", "elementwise_add_relu"])


def test_run_dir_helper_is_importable_and_callable():
    assert callable(powered_eval_from_run_dir)


def test_structural_anchors_are_a_proper_offgrid_superset():
    # G1a follow-through: anchors carry coverage BEYOND the dev held-out shapes, with no duplicates,
    # and every anchor is off the tiling grid (mod 64 != 0).
    assert len(STRUCTURAL_ANCHORS) == len(set(STRUCTURAL_ANCHORS)), "duplicate anchor"
    assert set(HELD_OUT_SHAPES).issubset(STRUCTURAL_ANCHORS)
    assert len(STRUCTURAL_ANCHORS) > len(HELD_OUT_SHAPES), "anchors must add coverage"
    assert all(m % TILING_BLOCK != 0 for m in STRUCTURAL_ANCHORS)
