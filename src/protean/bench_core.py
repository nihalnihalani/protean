"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# bench_core.py — CUDA-event timing, fresh-input-per-iter, 3-seed protocol
import torch, triton

SEED_CORRECT, SEED_TIMED_BASE, SEED_POST = 42, 43, 44

@torch.no_grad()
def bench_kernel(op_ref, kernel_fn, make_inputs, shape, dtype, reps=100, warmup=25):
    # ---- correctness pass (pre-timing) ----
    xs = make_inputs(shape, dtype, seed=SEED_CORRECT)
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
        inp = make_inputs(shape, dtype, seed=SEED_TIMED_BASE + i)
        flush.zero_()                          # L2 flush each iter
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); kernel_fn(*inp); e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(s.elapsed_time(e))
    times.sort(); t_kernel = times[len(times)//2]   # median
    speedup = t_eager / max(t_kernel, 1e-6)

    # ---- post-timing correctness on a THIRD seed materialized AFTER timing ----
    xp = make_inputs(shape, dtype, seed=SEED_POST)  # kernel never saw these → no precompute
    correct_post = torch.allclose(kernel_fn(*xp), op_ref(*xp), rtol=1e-2, atol=1e-2)

    return dict(correct=(correct and correct_post), dtype_ok=dtype_ok, shape_ok=shape_ok,
                speedup=speedup, t_eager=t_eager, t_kernel=t_kernel)
