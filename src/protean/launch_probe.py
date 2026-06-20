"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# launch_probe.py — patch the STABLE Python boundary, with auto-fallback to profiler
import triton

class LaunchCounter:
    def __init__(self): self.n = 0
    def __enter__(self):
        self._orig = triton.runtime.jit.JITFunction.__call__
        cnt = self
        def patched(self_jit, *a, **k):
            cnt.n += 1
            return cnt._orig(self_jit, *a, **k)
        triton.runtime.jit.JITFunction.__call__ = patched
        return self
    def __exit__(self, *exc):
        triton.runtime.jit.JITFunction.__call__ = self._orig

def count_launches(fn, *args):
    with LaunchCounter() as lc:
        fn(*args)
    return lc.n

# Fallback selector chosen at import: if the in-image smoke test (incl cached path)
# fails, USE_PROFILER=True and count Triton kernel names in torch.profiler trace.
