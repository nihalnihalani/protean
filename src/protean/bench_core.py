"""CUDA verifier for Protean kernel candidates."""

from __future__ import annotations

import importlib.util
import os
import random
import statistics
import sys
import tempfile
import types
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

# torch/triton are configured with ignore_missing_imports in pyproject, so the
# imported names are typed as ``Any``; the ``= None`` fallback below is therefore a
# legal reassignment and the ``assert ... is not None`` guards in GPU-touching
# functions (after ``_require_torch()``) document the runtime contract.
try:
    import torch
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only unit tests can still import helpers.
    torch = None
    triton = None
    tl = None

from protean.anti_hack import _audit_import_hook as _audit_import_hook  # re-export
from protean.anti_hack import (
    contains_triton_jit,
    install_audit_hook,
    set_audit_armed,
)
from protean.task_catalog import OpSpec

# The PEP 578 import audit hook lives in anti_hack and is armed lazily by
# load_solution (NOT at module-import time) -- see install_audit_hook's docstring
# for why a stdlib-root-banning hook cannot be installed during interpreter
# bootstrap. The _audit_import_hook re-export above preserves
# bench_core._audit_import_hook for callers/tests that reference it on this module.


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


def _require_torch() -> None:
    if torch is None or triton is None or tl is None:
        raise RuntimeError("torch and triton are required for CUDA benchmark verification")
    _ensure_triton_cache_dir()


def make_inputs(n: int, dtype: str, seed: int, op: str) -> tuple[Any, ...]:
    _require_torch()
    assert torch is not None
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


def eager_elementwise_add_relu(x: Any, y: Any) -> Any:
    assert torch is not None
    return torch.relu(x + y)


def eager_rmsnorm(x: Any, weight: Any, eps: float = 1e-5) -> Any:
    assert torch is not None
    x_f32 = x.float()
    rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    return (x_f32 * rms * weight.float()).to(dtype=x.dtype)


def eager_softmax_rows(x: Any) -> Any:
    assert torch is not None
    return torch.softmax(x, dim=-1)


def eager_fn_for_op(op: str) -> Callable[..., Any]:
    if op == "elementwise_add_relu":
        return eager_elementwise_add_relu
    if op == "rmsnorm":
        return eager_rmsnorm
    if op == "softmax_rows":
        return eager_softmax_rows
    raise ValueError(f"unknown op: {op}")


def load_solution(src: str) -> types.ModuleType:
    _require_torch()
    temp_dir = Path(tempfile.gettempdir()) / "protean_candidates"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"candidate_{uuid.uuid4().hex}.py"
    path.write_text(src)
    name = f"protean_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        path.unlink(missing_ok=True)
        raise ValueError("could not import candidate source")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update({"torch": torch, "triton": triton, "tl": tl})
    sys.modules[name] = module
    # Arm the import audit-hook backstop immediately before executing untrusted
    # candidate source. This is the single chokepoint where candidate code runs,
    # so installing here guarantees the hook is active for every exec_module while
    # avoiding the interpreter-bootstrap crash a process-wide eager install causes
    # (stdlib lazily imports importlib.machinery/os, which the hook would block).
    install_audit_hook()
    # Enforce the import audit hook ONLY while executing untrusted candidate
    # source; disarm immediately after so the host process and torch/triton
    # threads (which run during benchmarking, after this returns) are unaffected.
    set_audit_armed(True)
    try:
        spec.loader.exec_module(module)
        if "solution" not in module.__dict__:
            raise ValueError("candidate must define solution(...)")
        return module
    finally:
        set_audit_armed(False)
        # Tombstone the candidate so its module name, monkey-patches, and globals
        # cannot survive into the next evaluation round, and remove its temp file
        # so the candidate directory does not accumulate untrusted source. The
        # returned module object stays alive via the caller's reference and its
        # compiled functions remain callable; only the sys.modules registration
        # and on-disk file are removed. Source: SOL-ExecBench arXiv:2603.19173
        # (state-caching family); CPython importlib docs; IMPROVEMENT_RESEARCH.md
        # item #4.
        sys.modules.pop(name, None)
        path.unlink(missing_ok=True)


class _TritonLaunchCounter:
    """Count real Triton kernel launches during a timed window.

    A source-grep for @triton.jit (contains_triton_jit) only proves a kernel was
    *defined*, not that it was *called*. An LLM can satisfy the grep with a
    decorative kernel and delegate the real work elsewhere (Dr. Kernel arXiv
    2602.05885). This hooks Triton's actual launch path so launches_timed
    reflects real GPU launches; a count of 0 over the timed reps means the
    @triton.jit kernel was never executed.

    Triton's launch hook API is internal and varies across versions, so this is
    a best-effort instrument guarded by hasattr. When the hook surface is
    unavailable, available() returns False and callers fall back to the
    source-grep heuristic (no behavior change vs. the previous code).
    """

    def __init__(self) -> None:
        self.count = 0
        self._knobs: Any | None = None
        self._prev_hook: Any | None = None
        self._installed = False
        if triton is None:
            return
        knobs = getattr(getattr(triton, "runtime", None), "knobs", None)
        runtime_knobs = getattr(knobs, "runtime", None) if knobs is not None else None
        if runtime_knobs is not None and hasattr(runtime_knobs, "launch_enter_hook"):
            self._knobs = runtime_knobs

    def available(self) -> bool:
        return self._knobs is not None

    def _hook(self, *args: object, **kwargs: object) -> Any:
        self.count += 1
        if callable(self._prev_hook):
            return self._prev_hook(*args, **kwargs)
        return None

    def __enter__(self) -> _TritonLaunchCounter:
        if self.available():
            assert self._knobs is not None
            try:
                self._prev_hook = self._knobs.launch_enter_hook
                self._knobs.launch_enter_hook = self._hook
                self._installed = True
            except Exception:
                self._knobs = None
                self._installed = False
        return self

    def __exit__(self, *exc: object) -> None:
        if self._installed and self._knobs is not None:
            try:
                self._knobs.launch_enter_hook = self._prev_hook
            except Exception:
                pass
            self._installed = False
        return None


def _time_cuda_raw(fn: Callable[..., Any], args: tuple[Any, ...], reps: int, warmup: int) -> list[float]:
    """Time ``fn`` over ``reps`` measured iterations and return the raw per-rep
    millisecond samples (after ``warmup`` discarded iterations).

    Returning the full sample array (rather than only the median) lets callers
    compute distribution statistics -- CV for measurement-noise flagging and a
    bootstrap confidence interval on the speedup ratio. Each rep flushes the L2
    cache so that back-to-back launches do not benefit from a warm cache, which
    is standard practice for high-fidelity kernel benchmarking.
    """

    _require_torch()
    assert torch is not None
    times: list[float] = []
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "l2_cache_size", 0) or (256 * 1024 * 1024)
    flush = torch.empty((l2 * 2,), dtype=torch.int8, device="cuda")

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

    return times


def _time_cuda(fn: Callable[..., Any], args: tuple[Any, ...], reps: int, warmup: int) -> float:
    return statistics.median(_time_cuda_raw(fn, args, reps=reps, warmup=warmup))


def _timing_stats(times: list[float]) -> dict[str, float]:
    """Distribution summary for a raw per-rep timing array (pure Python).

    Surfaces the median (the point estimate used for speedup) alongside the
    coefficient of variation, IQR, and a 10%-trimmed mean. A high CV means the
    measurement was noisy and the resulting speedup should be treated with
    suspicion by downstream RL -- a noise-aware signal that a bare median hides.
    """

    if not times:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "cv": 0.0,
            "iqr_ms": 0.0,
            "trimmed_mean_ms": 0.0,
        }
    ordered = sorted(times)
    n = len(ordered)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    # Sample stdev (N-1 denominator) so the reported CV matches the standard
    # Coefficient-of-Variation definition used by kernel-benchmarking noise
    # thresholds (e.g. CV>0.05 = suspicious). pstdev (N-denominator) would
    # under-report noise for the small rep counts used here.
    stdev = statistics.stdev(ordered) if n > 1 else 0.0
    cv = stdev / mean if mean > 0 else 0.0

    def _quantile(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    iqr = _quantile(0.75) - _quantile(0.25)
    trim = int(n * 0.1)
    trimmed = ordered[trim : n - trim] if n - 2 * trim > 0 else ordered
    trimmed_mean = statistics.fmean(trimmed)
    return {
        "median_ms": float(median),
        "mean_ms": float(mean),
        "cv": float(cv),
        "iqr_ms": float(iqr),
        "trimmed_mean_ms": float(trimmed_mean),
    }


def _bootstrap_speedup_ci(
    eager_times: list[float],
    kernel_times: list[float],
    *,
    b: int = 1000,
    seed: int = 1234,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the speedup ratio (eager / kernel).

    Pure Python (no numpy) to match the bootstrap style already used in
    eval_protocol.hierarchical_bootstrap_ci. Eager and kernel reps are measured
    sequentially on the same GPU, so their measurement noise (thermal/driver/L2
    state) is positively correlated. When the two arrays are equal-length (the
    bench_source case -- both use ``reps``) we treat them as paired and bootstrap
    the per-rep ratio eager[i]/kernel[i]; this preserves the pairing instead of
    breaking it with independent resampling (which would inflate the interval).
    When lengths differ we fall back to independent resampling of each array.
    Returns the (p05, p95) percentiles of the bootstrapped mean ratio; a wide
    interval signals that a headline speedup is not statistically robust.
    """

    if not eager_times or not kernel_times:
        return (0.0, 0.0)
    rng = random.Random(seed)
    ratios: list[float] = []
    if len(eager_times) == len(kernel_times):
        # Paired bootstrap: resample rep indices jointly so correlated noise
        # cancels in the per-rep ratio (matched-pairs design).
        per_rep = [e / max(k, 1e-9) for e, k in zip(eager_times, kernel_times)]
        n = len(per_rep)
        for _ in range(b):
            total = 0.0
            for _ in range(n):
                total += per_rep[rng.randrange(n)]
            ratios.append(total / n)
    else:
        ne = len(eager_times)
        nk = len(kernel_times)
        for _ in range(b):
            e = sorted(eager_times[rng.randrange(ne)] for _ in range(ne))
            k = sorted(kernel_times[rng.randrange(nk)] for _ in range(nk))
            e_med = e[ne // 2] if ne % 2 else 0.5 * (e[ne // 2 - 1] + e[ne // 2])
            k_med = k[nk // 2] if nk % 2 else 0.5 * (k[nk // 2 - 1] + k[nk // 2])
            ratios.append(e_med / max(k_med, 1e-9))
    ratios.sort()

    def _pct(p: float) -> float:
        idx = min(b - 1, max(0, int(round(p * (b - 1)))))
        return ratios[idx]

    return (_pct(0.05), _pct(0.95))


def _time_cuda_graph_raw(fn: Callable[..., Any], args: tuple[Any, ...], reps: int, warmup: int) -> list[float] | None:
    """CUDA-graph-captured replay timing for sub-10us kernels.

    For very fast kernels, per-launch CPU dispatch overhead dominates the CUDA
    event window and inflates the measured time. Capturing the call in a CUDA
    graph and timing graph *replay* removes that overhead (do_bench_cudagraph
    pattern). Returns the raw per-rep samples, or ``None`` when CUDA graphs are
    unavailable or capture fails -- callers then fall back to event timing with
    no behavior change. This path requires CUDA; on CPU it returns None.
    """

    _require_torch()
    assert torch is not None
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"):
        return None
    try:
        # Warm up on a side stream so capture sees a clean state.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(max(warmup, 3)):
                fn(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*args)
        torch.cuda.synchronize()

        times: list[float] = []
        for i in range(warmup + reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(start.elapsed_time(end))
        return times
    except Exception:
        return None


def _is_strict_tensor(out: Any) -> bool:
    """True only if ``out`` is *exactly* torch.Tensor, not a subclass.

    Uses ``type(x) is torch.Tensor`` (identity), NOT isinstance, because a
    FakeTensor / meta tensor / custom tensor subclass passes isinstance but is
    never executed on the GPU -- it can fabricate a shape/dtype-matching object
    that defeats correctness and timing without doing real work. Returning a
    non-tensor (tuple, ndarray, Python scalar) is likewise rejected.
    Source: SOL-ExecBench arXiv:2603.19173 Table 3 (FakeTensor family);
    IMPROVEMENT_RESEARCH.md item #4.
    """
    tensor_cls = getattr(torch, "Tensor", None)
    if tensor_cls is None:
        # torch is unavailable or a test double without a real Tensor class; the
        # identity gate cannot run, so do not reject (bench_source's CUDA path,
        # which is where this matters, always has the real torch module).
        return True
    return type(out) is tensor_cls


def _check_correct_multi_init(
    solution: Callable[..., Any],
    eager_fn: Callable[..., Any],
    *,
    n: int,
    spec: OpSpec,
    seeds: tuple[int, ...],
) -> bool:
    """Robust-kbench-style multi-init correctness: require allclose on every one
    of ``seeds`` independent random inputs (not a single fixed seed).

    A kernel that hardcodes / caches outputs for one specific input
    distribution passes a single-seed check but fails here. Returns True only if
    dtype, shape, and values match the eager reference for all seeds.
    """

    assert torch is not None
    for s in seeds:
        inputs = make_inputs(n, spec.dtype, seed=s, op=spec.name)
        out = solution(*inputs)
        ref = eager_fn(*inputs)
        if not _is_strict_tensor(out):
            return False
        if out.dtype != ref.dtype:
            return False
        if tuple(out.shape) != tuple(ref.shape):
            return False
        if not torch.allclose(out, ref, rtol=spec.rtol, atol=spec.atol):
            return False
    return True


def _compute_ratio_from_events(events: Any) -> float:
    triton_us = 0.0
    total_us = 0.0
    for event in events:
        duration = getattr(event, "self_cuda_time_total", None)
        if duration is None:
            duration = getattr(event, "cuda_time_total", 0.0)
        if duration is None or duration <= 0:
            continue
        total_us += float(duration)
        key = (getattr(event, "key", "") or "").lower()
        if "triton" in key or key.startswith("kernel_") or "jit" in key:
            triton_us += float(duration)
    if total_us <= 0:
        return 0.0
    return max(0.0, min(triton_us / total_us, 1.0))


def measure_pr_frac(solution: Callable[..., Any], *, n: int, spec: OpSpec, iters: int = 10) -> float:
    """Estimate Triton GPU time / total GPU time for the candidate."""

    assert torch is not None
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
) -> dict[str, Any]:
    _require_torch()
    assert torch is not None
    module = load_solution(src)
    solution = module.solution
    eager_fn = eager_fn_for_op(spec.name)

    def _non_tensor_result() -> dict[str, Any]:
        """Synthetic bench dict for a candidate that did not return a real
        torch.Tensor (FakeTensor / meta / subclass / non-tensor). We return
        early WITHOUT timing it: a fake tensor must never enter the timing loop
        where it could fabricate a speedup."""
        return {
            "op": spec.name,
            "shape": n,
            "split": split,
            "correct": False,
            "multi_init_ok": False,
            "dtype_ok": False,
            "shape_ok": False,
            "speedup": 0.0,
            "speedup_ci_p05": 0.0,
            "speedup_ci_p95": 0.0,
            "speedup_graph": None,
            "t_eager_ms": 0.0,
            "t_kernel_ms": 0.0,
            "t_kernel_graph_ms": None,
            "t_eager_cv": 0.0,
            "t_kernel_cv": 0.0,
            "timing_eager": _timing_stats([]),
            "timing_kernel": _timing_stats([]),
            "launches_timed": 0,
            "pr_frac": 0.0,
            "caps": ["non_tensor_output"],
            "caps_advisory": [],
        }

    inputs = make_inputs(n, spec.dtype, seed=42, op=spec.name)
    out_candidate = solution(*inputs)
    # Strict tensor-subclass identity gate: a FakeTensor/meta/subclass output
    # passes isinstance but never ran on the GPU. Reject before any timing.
    if not _is_strict_tensor(out_candidate):
        return _non_tensor_result()
    out_ref = eager_fn(*inputs)
    dtype_ok = out_candidate.dtype == out_ref.dtype
    shape_ok = tuple(out_candidate.shape) == tuple(out_ref.shape)
    correct_pre = False
    if dtype_ok and shape_ok:
        correct_pre = torch.allclose(out_candidate, out_ref, rtol=spec.rtol, atol=spec.atol)

    eager_args = make_inputs(n, spec.dtype, seed=43, op=spec.name)
    candidate_args = tuple(t.clone() for t in eager_args)
    eager_times = _time_cuda_raw(eager_fn, eager_args, reps=reps, warmup=warmup)
    t_eager_ms = statistics.median(eager_times)

    # Count real Triton launches during the candidate's timed window. If the
    # runtime hook surface is available we use the measured count (0 means the
    # @triton.jit kernel was never executed -> hard reward gate via the
    # no_kernel_launched cap). Otherwise we fall back to the source-grep heuristic
    # but surface that the launch count is UNVERIFIED via an advisory cap, rather
    # than silently implying launches occurred. The advisory cap is reported in
    # caps_advisory (not the reward-gating caps), so it does not zero a legit
    # kernel on older Triton -- it only makes the unverifiable case auditable.
    counter = _TritonLaunchCounter()
    with counter:
        kernel_times = _time_cuda_raw(solution, candidate_args, reps=reps, warmup=warmup)
    t_kernel_ms = statistics.median(kernel_times)
    caps: list[str] = []
    caps_advisory: list[str] = []
    if counter.available():
        launches_timed = int(counter.count)
        if launches_timed <= 0:
            caps.append("no_kernel_launched")
    else:
        launches_timed = reps if contains_triton_jit(src) else 0
        caps_advisory.append("launch_count_unverified")

    post_inputs = make_inputs(n, spec.dtype, seed=44, op=spec.name)
    out_post = solution(*post_inputs)
    # Re-apply the strict identity gate on the post-trial output: a candidate
    # could return a real tensor on the first call and a FakeTensor later.
    if not _is_strict_tensor(out_post):
        return _non_tensor_result()
    ref_post = eager_fn(*post_inputs)
    correct_post = False
    if tuple(out_post.shape) == tuple(ref_post.shape) and out_post.dtype == ref_post.dtype:
        correct_post = torch.allclose(out_post, ref_post, rtol=spec.rtol, atol=spec.atol)

    # Robust-kbench-style multi-init correctness: require the candidate to match
    # the eager reference across several additional independent random seeds.
    # This catches kernels that hardcode / cache outputs for the fixed 42/44
    # inputs. Surfaced as the non-gating advisory cap multi_init_fail so the
    # primary `correct` verdict (pre+post) keeps its existing semantics while the
    # stricter signal is available to downstream eval.
    multi_init_ok = False
    if correct_pre and correct_post:
        try:
            multi_init_ok = _check_correct_multi_init(solution, eager_fn, n=n, spec=spec, seeds=(101, 202, 303))
        except Exception:
            multi_init_ok = False
        if not multi_init_ok:
            caps_advisory.append("multi_init_fail")
    correct = bool(correct_pre and correct_post)
    speedup = t_eager_ms / max(t_kernel_ms, 1e-9)
    try:
        pr_frac = measure_pr_frac(solution, n=n, spec=spec, iters=min(reps, 10))
    except Exception:
        pr_frac = 0.0

    # Distribution-aware timing metadata (CV/IQR/trimmed-mean) and a percentile
    # bootstrap CI on the speedup ratio. All additive: downstream reward uses the
    # median-based `speedup` exactly as before.
    eager_stats = _timing_stats(eager_times)
    kernel_stats = _timing_stats(kernel_times)
    speedup_ci_p05, speedup_ci_p95 = _bootstrap_speedup_ci(eager_times, kernel_times)

    # CUDA-graph replay timing for sub-launch-overhead kernels (None when CUDA
    # graphs are unavailable or capture fails -> reported as None, no fallback
    # effect on the headline speedup).
    t_kernel_graph_ms = None
    speedup_graph = None
    # Rebuild FRESH inputs for graph capture: candidate_args was already consumed
    # by _time_cuda_raw above, and in-place Triton kernels overwrite their output
    # buffers -- capturing the graph over those mutated tensors would time a no-op
    # re-execution of already-computed outputs, not a genuine replay. seed=45 is
    # distinct from the correctness (42/44) and timing (43) seeds.
    graph_args = make_inputs(n, spec.dtype, seed=45, op=spec.name)
    graph_times = _time_cuda_graph_raw(solution, graph_args, reps=reps, warmup=warmup)
    if graph_times:
        t_kernel_graph_ms = statistics.median(graph_times)
        speedup_graph = t_eager_ms / max(t_kernel_graph_ms, 1e-9)

    return {
        "op": spec.name,
        "shape": n,
        "split": split,
        "correct": correct,
        "multi_init_ok": multi_init_ok,
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "speedup": speedup,
        "speedup_ci_p05": speedup_ci_p05,
        "speedup_ci_p95": speedup_ci_p95,
        "speedup_graph": speedup_graph,
        "t_eager_ms": t_eager_ms,
        "t_kernel_ms": t_kernel_ms,
        "t_kernel_graph_ms": t_kernel_graph_ms,
        "t_eager_cv": eager_stats["cv"],
        "t_kernel_cv": kernel_stats["cv"],
        "timing_eager": eager_stats,
        "timing_kernel": kernel_stats,
        "launches_timed": launches_timed,
        "pr_frac": pr_frac,
        "caps": caps,
        "caps_advisory": caps_advisory,
    }
