from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import protean.bench_core as bench_core
from protean.bench_core import (
    _bootstrap_speedup_ci,
    _check_correct_multi_init,
    _compute_ratio_from_events,
    _time_cuda_graph_raw,
    _timing_stats,
)
from protean.task_catalog import OpSpec


def _event(key: str, self_cuda_time_total):
    event = MagicMock()
    event.key = key
    event.self_cuda_time_total = self_cuda_time_total
    event.cuda_time_total = self_cuda_time_total
    return event


def test_compute_ratio_all_triton():
    assert _compute_ratio_from_events([_event("triton_kernel", 500), _event("kernel_add", 500)]) == 1.0


def test_compute_ratio_mixed_gpu_work():
    ratio = _compute_ratio_from_events([_event("triton_kernel", 250), _event("Memcpy DtoD", 750)])
    assert ratio == 0.25


def test_compute_ratio_ignores_empty_or_negative_events():
    assert _compute_ratio_from_events([_event("triton_kernel", -10), _event("other", 0)]) == 0.0


# ----- timing distribution statistics (_timing_stats) -----


def test_timing_stats_empty_is_zeroed():
    stats = _timing_stats([])
    assert stats == {
        "median_ms": 0.0,
        "mean_ms": 0.0,
        "cv": 0.0,
        "iqr_ms": 0.0,
        "trimmed_mean_ms": 0.0,
    }


def test_timing_stats_constant_has_zero_cv_and_iqr():
    stats = _timing_stats([2.0] * 10)
    assert stats["median_ms"] == 2.0
    assert stats["mean_ms"] == 2.0
    assert stats["cv"] == 0.0
    assert stats["iqr_ms"] == 0.0
    assert stats["trimmed_mean_ms"] == 2.0


def test_timing_stats_cv_positive_for_noisy_samples():
    stats = _timing_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["cv"] > 0.0
    # 10% trim of 5 samples trims 0 each side, so trimmed == full mean here.
    assert stats["median_ms"] == 3.0


def test_timing_stats_cv_uses_sample_stdev():
    # Pin the estimator choice: sample stdev of [1,3] is sqrt(2), mean is 2.0,
    # so CV must equal sqrt(2)/2 (~0.707). Population stdev would give 0.5.
    stats = _timing_stats([1.0, 3.0])
    assert stats["cv"] == pytest.approx(math.sqrt(2) / 2, rel=1e-9)


def test_timing_stats_trimmed_mean_drops_extremes():
    # 20 samples: trim removes top/bottom 2, dropping the 1000.0 outlier.
    times = [1.0] * 19 + [1000.0]
    stats = _timing_stats(times)
    assert stats["trimmed_mean_ms"] == 1.0
    assert stats["mean_ms"] > 1.0


# ----- bootstrap speedup CI (_bootstrap_speedup_ci) -----


def test_bootstrap_speedup_ci_empty_returns_zeros():
    assert _bootstrap_speedup_ci([], [1.0]) == (0.0, 0.0)
    assert _bootstrap_speedup_ci([1.0], []) == (0.0, 0.0)


def test_bootstrap_speedup_ci_brackets_point_estimate():
    eager = [2.0] * 20
    kernel = [1.0] * 20
    p05, p95 = _bootstrap_speedup_ci(eager, kernel)
    # Constant inputs -> every resample yields the same 2.0 ratio.
    assert p05 == 2.0
    assert p95 == 2.0


def test_bootstrap_speedup_ci_is_ordered_and_deterministic():
    eager = [2.0, 2.1, 1.9, 2.05, 1.95, 2.2, 1.8, 2.0, 2.0, 2.0]
    kernel = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8, 1.0, 1.0, 1.0]
    a = _bootstrap_speedup_ci(eager, kernel, seed=7)
    b = _bootstrap_speedup_ci(eager, kernel, seed=7)
    assert a == b  # deterministic given seed
    assert a[0] <= a[1]
    # Point estimate (median 2.0 / median 1.0 = 2.0) should fall inside the CI.
    assert a[0] <= 2.0 <= a[1]


def test_bootstrap_speedup_ci_paired_perfect_correlation_is_tight():
    # Equal-length arrays => paired ratio bootstrap. When eager[i]/kernel[i] is
    # constant (perfectly correlated noise), every per-rep ratio is identical, so
    # the paired CI collapses to a point even though each array is itself noisy.
    eager = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    kernel = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # eager[i]/kernel[i] == 2.0 for all i
    p05, p95 = _bootstrap_speedup_ci(eager, kernel, seed=1)
    assert p05 == pytest.approx(2.0)
    assert p95 == pytest.approx(2.0)


def test_bootstrap_speedup_ci_unequal_lengths_uses_independent_path():
    # Different lengths fall back to the independent-resample (median) path and
    # must still produce an ordered, finite CI bracketing the point estimate.
    eager = [2.0] * 8
    kernel = [1.0] * 5
    p05, p95 = _bootstrap_speedup_ci(eager, kernel, seed=3)
    assert p05 == pytest.approx(2.0)
    assert p95 == pytest.approx(2.0)


# ----- CUDA-graph replay fallback (_time_cuda_graph_raw) -----


def _fake_torch(cuda_available: bool, has_cudagraph: bool = True):
    cuda = SimpleNamespace(is_available=lambda: cuda_available)
    if has_cudagraph:
        cuda.CUDAGraph = object
    return SimpleNamespace(cuda=cuda)


def test_time_cuda_graph_raw_returns_none_when_cuda_unavailable(monkeypatch):
    # CPU/unavailable fallback: returns None (callers then use event timing).
    monkeypatch.setattr(bench_core, "torch", _fake_torch(cuda_available=False))
    monkeypatch.setattr(bench_core, "triton", object())
    monkeypatch.setattr(bench_core, "tl", object())
    monkeypatch.setattr(bench_core, "_ensure_triton_cache_dir", lambda: None)
    assert _time_cuda_graph_raw(lambda: None, (), reps=5, warmup=2) is None


def test_time_cuda_graph_raw_returns_none_without_cudagraph_attr(monkeypatch):
    monkeypatch.setattr(bench_core, "torch", _fake_torch(cuda_available=True, has_cudagraph=False))
    monkeypatch.setattr(bench_core, "triton", object())
    monkeypatch.setattr(bench_core, "tl", object())
    monkeypatch.setattr(bench_core, "_ensure_triton_cache_dir", lambda: None)
    assert _time_cuda_graph_raw(lambda: None, (), reps=5, warmup=2) is None


# ----- multi-init correctness advisory contract (_check_correct_multi_init) -----


class _FakeTensor:
    """Minimal tensor stand-in exposing dtype/shape and a payload for allclose."""

    def __init__(self, value, dtype="float16", shape=(8,)):
        self.value = value
        self.dtype = dtype
        self.shape = shape


_SPEC = OpSpec(
    name="elementwise_add_relu",
    dtype="float16",
    rtol=1e-2,
    atol=1e-2,
    p_target=0.5,
    speedup_floor=1.0,
    prompt_path="x",
)


def _install_fake_allclose(monkeypatch):
    fake_torch = SimpleNamespace(allclose=lambda a, b, rtol, atol: a.value == b.value)
    monkeypatch.setattr(bench_core, "torch", fake_torch)


def test_check_correct_multi_init_passes_for_seed_invariant_kernel(monkeypatch):
    # make_inputs returns a per-seed sentinel; an honest kernel reproduces the
    # reference for every seed, so multi-init passes (no advisory).
    _install_fake_allclose(monkeypatch)
    monkeypatch.setattr(bench_core, "make_inputs", lambda n, dtype, seed, op: (_FakeTensor(seed),))
    honest = lambda x: _FakeTensor(x.value)  # echoes the seed-dependent input
    eager = lambda x: _FakeTensor(x.value)
    assert _check_correct_multi_init(honest, eager, n=8, spec=_SPEC, seeds=(101, 202, 303)) is True


def test_check_correct_multi_init_fails_for_hardcoded_output(monkeypatch):
    # A cheating kernel returns a fixed tensor regardless of seed; it matches the
    # reference for one seed but diverges for the others -> multi-init fails,
    # which bench_source surfaces as the non-gating caps_advisory 'multi_init_fail'.
    _install_fake_allclose(monkeypatch)
    monkeypatch.setattr(bench_core, "make_inputs", lambda n, dtype, seed, op: (_FakeTensor(seed),))
    cheat = lambda x: _FakeTensor(101)  # hardcoded to the first seed's output
    eager = lambda x: _FakeTensor(x.value)
    assert _check_correct_multi_init(cheat, eager, n=8, spec=_SPEC, seeds=(101, 202, 303)) is False


def test_check_correct_multi_init_fails_on_dtype_mismatch(monkeypatch):
    _install_fake_allclose(monkeypatch)
    monkeypatch.setattr(bench_core, "make_inputs", lambda n, dtype, seed, op: (_FakeTensor(seed),))
    wrong_dtype = lambda x: _FakeTensor(x.value, dtype="float32")
    eager = lambda x: _FakeTensor(x.value, dtype="float16")
    assert _check_correct_multi_init(wrong_dtype, eager, n=8, spec=_SPEC, seeds=(101,)) is False
