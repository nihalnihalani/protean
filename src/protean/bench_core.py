"""CUDA verifier for Protean kernel candidates."""

from __future__ import annotations

import os
import statistics
import tempfile
import uuid
import importlib.util
import sys
from pathlib import Path

try:
    import torch
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only unit tests can still import helpers.
    torch = None
    triton = None
    tl = None

from protean.anti_hack import contains_triton_jit
from protean.task_catalog import OpSpec


def _ensure_triton_cache_dir() -> None:
    current = os.environ.get("TRITON_CACHE_DIR")
    candidates = [Path(current)] if current else []
    candidates.append(Path("/triton-cache"))
    candidates.append(Path.home() / ".cache" / "protean-triton")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write-test"
            test_file.write_text("ok")
            test_file.unlink()
            os.environ["TRITON_CACHE_DIR"] = str(path)
            return
        except Exception:
            continue
    raise RuntimeError("no writable Triton cache directory found")


def _require_torch():
    if torch is None or triton is None or tl is None:
        raise RuntimeError("torch and triton are required for CUDA benchmark verification")
    _ensure_triton_cache_dir()


def make_inputs(n: int, dtype: str, seed: int, op: str) -> tuple[torch.Tensor, ...]:
    _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Protean benchmark verification")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((n,), device="cuda", dtype=torch_dtype)
    if op == "elementwise_add_relu":
        y = torch.randn((n,), device="cuda", dtype=torch_dtype)
        return x, y
    if op == "rmsnorm":
        weight = torch.randn((n,), device="cuda", dtype=torch_dtype)
        return x, weight
    if op == "softmax_rows":
        # Shape: (64, n) where n is the columns size. Scale by 2.0 as in reference.py
        x = torch.randn((64, n), device="cuda", dtype=torch_dtype) * 2.0
        return (x,)
    raise ValueError(f"unknown op: {op}")


def eager_elementwise_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f32 = x.float()
    rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    return (x_f32 * rms * weight.float()).to(dtype=x.dtype)


def eager_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def eager_fn_for_op(op: str):
    if op == "elementwise_add_relu":
        return eager_elementwise_add_relu
    if op == "rmsnorm":
        return eager_rmsnorm
    if op == "softmax_rows":
        return eager_softmax_rows
    raise ValueError(f"unknown op: {op}")


def load_solution(src: str):
    _require_torch()
    temp_dir = Path(tempfile.gettempdir()) / "protean_candidates"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"candidate_{uuid.uuid4().hex}.py"
    path.write_text(src)
    name = f"protean_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("could not import candidate source")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update({"torch": torch, "triton": triton, "tl": tl})
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if "solution" not in module.__dict__:
        raise ValueError("candidate must define solution(...)")
    return module


def _time_cuda(fn, args: tuple[torch.Tensor, ...], reps: int, warmup: int) -> float:
    _require_torch()
    times: list[float] = []
    flush = torch.empty((16 * 1024 * 1024,), dtype=torch.int8, device="cuda")

    for i in range(warmup + reps):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))

    return statistics.median(times)


def _compute_ratio_from_events(events) -> float:
    triton_us = 0.0
    total_us = 0.0
    for event in events:
        duration = getattr(event, "self_cuda_time_total", None)
        if duration is None:
            duration = getattr(event, "cuda_time_total", 0.0)
        if duration <= 0:
            continue
        total_us += float(duration)
        key = (getattr(event, "key", "") or "").lower()
        if "triton" in key or key.startswith("kernel_") or "jit" in key:
            triton_us += float(duration)
    if total_us <= 0:
        return 0.0
    return max(0.0, min(triton_us / total_us, 1.0))


def measure_pr_frac(solution, *, n: int, spec: OpSpec, iters: int = 10) -> float:
    """Estimate Triton GPU time / total GPU time for the candidate."""

    from torch.profiler import ProfilerActivity, profile

    warm_inputs = make_inputs(n, spec.dtype, seed=99, op=spec.name)
    for _ in range(3):
        solution(*warm_inputs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for i in range(iters):
            inputs = make_inputs(n, spec.dtype, seed=200 + i, op=spec.name)
            solution(*inputs)
        torch.cuda.synchronize()

    return _compute_ratio_from_events(prof.key_averages())


def bench_source(
    src: str,
    *,
    n: int,
    split: str,
    spec: OpSpec,
    reps: int = 50,
    warmup: int = 10,
) -> dict:
    _require_torch()
    module = load_solution(src)
    solution = module.solution
    eager_fn = eager_fn_for_op(spec.name)

    inputs = make_inputs(n, spec.dtype, seed=42, op=spec.name)
    out_candidate = solution(*inputs)
    out_ref = eager_fn(*inputs)
    dtype_ok = out_candidate.dtype == out_ref.dtype
    shape_ok = tuple(out_candidate.shape) == tuple(out_ref.shape)
    correct_pre = False
    if dtype_ok and shape_ok:
        correct_pre = torch.allclose(out_candidate, out_ref, rtol=spec.rtol, atol=spec.atol)

    eager_args = make_inputs(n, spec.dtype, seed=43, op=spec.name)
    candidate_args = tuple(t.clone() for t in eager_args)
    t_eager_ms = _time_cuda(eager_fn, eager_args, reps=reps, warmup=warmup)
    t_kernel_ms = _time_cuda(solution, candidate_args, reps=reps, warmup=warmup)
    launches_timed = reps if contains_triton_jit(src) else 0

    post_inputs = make_inputs(n, spec.dtype, seed=44, op=spec.name)
    out_post = solution(*post_inputs)
    ref_post = eager_fn(*post_inputs)
    correct_post = False
    if tuple(out_post.shape) == tuple(ref_post.shape) and out_post.dtype == ref_post.dtype:
        correct_post = torch.allclose(out_post, ref_post, rtol=spec.rtol, atol=spec.atol)
    correct = bool(correct_pre and correct_post)
    speedup = t_eager_ms / max(t_kernel_ms, 1e-9)
    try:
        pr_frac = measure_pr_frac(solution, n=n, spec=spec, iters=min(reps, 10))
    except Exception:
        pr_frac = 0.0

    return {
        "op": spec.name,
        "shape": n,
        "split": split,
        "correct": correct,
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "speedup": speedup,
        "t_eager_ms": t_eager_ms,
        "t_kernel_ms": t_kernel_ms,
        "launches_timed": launches_timed,
        "pr_frac": pr_frac,
    }
