"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# launch_probe.py — patch the STABLE Python boundary, with auto-fallback to profiler
import triton

import triton
import torch

class LaunchCounter:
    def __init__(self):
        self.n = 0
        self._orig_call = None
        self._orig_run = None
        self._orig_compile = None

    def __enter__(self):
        # Hook JITFunction.__call__
        if hasattr(triton.runtime.jit, "JITFunction"):
            self._orig_call = triton.runtime.jit.JITFunction.__call__
            cnt = self
            def patched_call(self_jit, *a, **k):
                cnt.n += 1
                return cnt._orig_call(self_jit, *a, **k)
            triton.runtime.jit.JITFunction.__call__ = patched_call

            # Also hook run if present
            if hasattr(triton.runtime.jit.JITFunction, "run"):
                self._orig_run = triton.runtime.jit.JITFunction.run
                def patched_run(self_jit, *a, **k):
                    cnt.n += 1
                    return cnt._orig_run(self_jit, *a, **k)
                triton.runtime.jit.JITFunction.run = patched_run
            
        # Hook triton.compile if present
        if hasattr(triton, "compile"):
            self._orig_compile = triton.compile
            cnt = self
            def patched_compile(*a, **k):
                cnt.n += 1
                return cnt._orig_compile(*a, **k)
            triton.compile = patched_compile
            
        return self

    def __exit__(self, *exc):
        if self._orig_call is not None:
            triton.runtime.jit.JITFunction.__call__ = self._orig_call
        if self._orig_run is not None:
            triton.runtime.jit.JITFunction.run = self._orig_run
        if self._orig_compile is not None:
            triton.compile = self._orig_compile

def count_launches(fn, *args):
    lc = LaunchCounter()
    with lc:
        try:
            fn(*args)
        except Exception:
            pass
    if lc.n > 0:
        return lc.n
        
    # Fallback to profiling CUDA events if Python hooks didn't register anything
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False
        ) as prof:
            fn(*args)
        n_prof = 0
        for event in prof.key_averages():
            if "triton" in event.key.lower():
                n_prof += event.count
        return n_prof
    except Exception:
        return 0

