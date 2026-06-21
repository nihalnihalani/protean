from __future__ import annotations

from unittest.mock import MagicMock

from protean.bench_core import _compute_ratio_from_events


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

