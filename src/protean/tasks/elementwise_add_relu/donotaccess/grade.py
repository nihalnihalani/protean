"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grade.py  (root:700, baked into image, loaded by grader via spec_from_file_location)
import sys, hashlib, os
sys.path.insert(0, "/donotaccess")
from rewards import compute_reward          # ← the ONE authority, same path the trainer uses

REWARDS_HASH = "<sha256 baked at image build>"

def _preflight():
    h = hashlib.sha256(open("/donotaccess/rewards.py", "rb").read()).hexdigest()
    if h != REWARDS_HASH:
        raise RuntimeError(f"REWARDS_HASH mismatch {h} != {REWARDS_HASH} — HALT")  # never silent-0

def grade_kernel(op, M, N, dtype, kernel_src, step=0):
    _preflight()
    from subprocess_runner import run_bench  # fail-closed subprocess
    bench = run_bench(op, M, N, dtype, kernel_src, timeout_s=90)
    if bench["status"] != "ok":
        return {"reward": 0.0, "caps": [bench["status"]], **bench}
    return compute_reward(
        correct=bench["correct"], speedup=bench["speedup"], pr_frac=bench["pr_frac"],
        launches_timed=bench["launches_timed"], dtype_ok=bench["dtype_ok"],
        shape_ok=bench["shape_ok"], step=step)

def grade(workdir, override, hidden_root):     # verilog-compatible signature
    op, M, N, dtype, src, split, step = _parse(workdir, override, hidden_root)
    return grade_kernel(op, M, N, dtype, src, step=step)
