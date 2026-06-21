"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# bench_core.py — CUDA-event timing, fresh-input-per-iter, 3-seed protocol
import torch
try:
    import triton
except ImportError:
    triton = None

SEED_CORRECT, SEED_TIMED_BASE, SEED_POST = 42, 43, 44

_EAGER_TIME_CACHE = {}

def _cached_eager_time(op_ref, make_inputs, shape, dtype):
    key = (op_ref.__name__ if hasattr(op_ref, "__name__") else str(op_ref), tuple(shape), str(dtype))
    if key in _EAGER_TIME_CACHE:
        return _EAGER_TIME_CACHE[key]
        
    # Warmup reference
    xs = make_inputs(shape, seed=42, dtype=dtype)
    for _ in range(10):
        op_ref(*xs)
    torch.cuda.synchronize()
    
    times = []
    for i in range(50):
        xs = make_inputs(shape, seed=100 + i, dtype=dtype)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        op_ref(*xs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    med_time = times[len(times) // 2]
    _EAGER_TIME_CACHE[key] = med_time
    return med_time

@torch.no_grad()
def bench_kernel(op_ref, kernel_fn, make_inputs, shape, dtype, reps=100, warmup=25):
    # ---- correctness pass (pre-timing) ----
    xs = make_inputs(shape, seed=SEED_CORRECT, dtype=dtype)
    out_k = kernel_fn(*xs); out_r = op_ref(*xs)
    correct = torch.allclose(out_k, out_r, rtol=1e-2, atol=1e-2)
    dtype_ok = (out_k.dtype == out_r.dtype)
    shape_ok = (tuple(out_k.shape) == tuple(out_r.shape))

    # ---- eager baseline (cache per (op,shape) — do NOT re-time every rollout) ----
    t_eager = _cached_eager_time(op_ref, make_inputs, shape, dtype)

    # ---- timed pass: FRESH random inputs EVERY iter (defeats scratchpad/result cache) ----
    flush = torch.empty(int(64e6 // 4), dtype=torch.int, device="cuda")  # L2 flush buffer
    times = []
    for i in range(warmup + reps):
        inp = make_inputs(shape, seed=SEED_TIMED_BASE + i, dtype=dtype)
        flush.zero_()                          # L2 flush each iter
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); kernel_fn(*inp); e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(s.elapsed_time(e))
    times.sort(); t_kernel = times[len(times)//2]   # median
    speedup = t_eager / max(t_kernel, 1e-6)

    # ---- post-timing correctness on a THIRD seed materialized AFTER timing ----
    xp = make_inputs(shape, seed=SEED_POST, dtype=dtype)  # kernel never saw these → no precompute
    correct_post = torch.allclose(kernel_fn(*xp), op_ref(*xp), rtol=1e-2, atol=1e-2)

    # ---- PR measurement ----
    try:
        pr_frac = measure_pr_frac(kernel_fn, make_inputs, shape, dtype, n_iters=10)
    except Exception as e:
        print(f"[protean] PR measurement failed: {type(e).__name__}: {e}, defaulting to 0.5")
        pr_frac = 0.5

    return dict(
        correct=(correct and correct_post),
        dtype_ok=dtype_ok,
        shape_ok=shape_ok,
        speedup=speedup,
        t_eager=t_eager,
        t_kernel=t_kernel,
        pr_frac=pr_frac,
    )

def _compute_ratio_from_events(events) -> float:
    """Helper to compute the profiling ratio from profiler events."""
    triton_us = 0
    total_us = 0
    for event in events:
        dur = getattr(event, "self_cuda_time_total", None)
        if dur is None:
            dur = getattr(event, "cuda_time_total", 0)
        if dur <= 0:
            continue
        total_us += dur
        key = (event.key or "").lower()
        if "triton" in key or key.startswith("kernel_") or "jit" in key:
            triton_us += dur
            
    if total_us <= 0:
        return 0.0
    ratio = triton_us / total_us
    return max(0.0, min(1.0, ratio))

@torch.no_grad()
def measure_pr_frac(kernel_fn, make_inputs, shape, dtype, n_iters: int = 10) -> float:
    """Measure the profiling ratio: time inside Triton kernels / total GPU time.
    
    Wraps the kernel function in torch.profiler over several iterations and
    computes the ratio of cumulative Triton CUDA kernel time to total CUDA
    device time. Used as an additive reward bonus (daVinci-LITE, plan §5).
    
    Returns a float in [0, 1]. Returns 0.0 if no CUDA activity is recorded
    (e.g. the kernel never launched), 1.0 if Triton time equals total time.
    """
    from torch.profiler import profile, ProfilerActivity
    
    # Warmup outside the profiler so JIT compile isn't counted.
    warm_inp = make_inputs(shape, seed=99, dtype=dtype)
    for _ in range(3):
        kernel_fn(*warm_inp)
    torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for i in range(n_iters):
            inp = make_inputs(shape, seed=200 + i, dtype=dtype)
            kernel_fn(*inp)
        torch.cuda.synchronize()
        
    return _compute_ratio_from_events(prof.key_averages())

