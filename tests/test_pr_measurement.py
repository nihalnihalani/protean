import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import torch
from protean.bench_core import _compute_ratio_from_events, measure_pr_frac
from protean.rewards import compute_reward

def _make_mock_event(key, cuda_time_us):
    """Build a fake profiler event with .key and .self_cuda_time_total."""
    ev = MagicMock()
    ev.key = key
    ev.self_cuda_time_total = cuda_time_us
    ev.cuda_time_total = cuda_time_us
    return ev

def test_compute_ratio_all_triton():
    events = [
        _make_mock_event("triton_add_kernel", 800),
        _make_mock_event("triton_relu_kernel", 200),
    ]
    ratio = _compute_ratio_from_events(events)
    assert ratio == 1.0

def test_compute_ratio_mixed():
    events = [
        _make_mock_event("triton_kernel_xyz", 500),
        _make_mock_event("Memcpy DtoD", 500),
    ]
    ratio = _compute_ratio_from_events(events)
    assert ratio == 0.5

def test_compute_ratio_no_activity():
    events = []
    ratio = _compute_ratio_from_events(events)
    assert ratio == 0.0

def test_compute_ratio_zero_and_negative():
    events = [
        _make_mock_event("triton_kernel", -100),
        _make_mock_event("Memcpy", 0),
    ]
    ratio = _compute_ratio_from_events(events)
    assert ratio == 0.0

def test_compute_ratio_cuda_time_total_fallback():
    ev1 = MagicMock()
    ev1.key = "triton_kernel"
    ev1.self_cuda_time_total = None
    ev1.cuda_time_total = 400
    
    ev2 = MagicMock()
    ev2.key = "other"
    ev2.self_cuda_time_total = None
    ev2.cuda_time_total = 600
    
    ratio = _compute_ratio_from_events([ev1, ev2])
    assert ratio == 0.4

def test_reward_responds_to_pr_frac():
    """Confirm pr_frac actually moves the reward by PR_BONUS * pr_frac."""
    base_kwargs = dict(correct=True, speedup=1.5, launches_timed=1, dtype_ok=True, shape_ok=True)
    
    r_low = compute_reward(**base_kwargs, pr_frac=0.1)["reward"]
    r_mid = compute_reward(**base_kwargs, pr_frac=0.5)["reward"]
    r_high = compute_reward(**base_kwargs, pr_frac=0.9)["reward"]
    
    # PR_BONUS = 0.2; expected gap between low and high is 0.2 * (0.9 - 0.1) = 0.16
    assert r_low < r_mid < r_high, f"Reward should increase with pr_frac: {r_low}, {r_mid}, {r_high}"
    assert abs((r_high - r_low) - 0.16) < 1e-5, f"Expected delta 0.16, got {r_high - r_low}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_measure_pr_frac_cuda():
    # Dummy kernel and make_inputs for testing on CUDA hosts
    def dummy_kernel(*x):
        return x[0] * 2
        
    def make_inputs(shape, seed, dtype):
        return (torch.randn(shape, device="cuda", dtype=dtype),)
        
    # Should run and return a valid float in [0, 1]
    ratio = measure_pr_frac(dummy_kernel, make_inputs, (1024,), torch.float16, n_iters=5)
    assert 0.0 <= ratio <= 1.0
