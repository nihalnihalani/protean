"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grade.py  (root:700, baked into image, loaded by grader via spec_from_file_location)
import sys, hashlib, os
if os.path.exists("/donotaccess"):
    sys.path.insert(0, "/donotaccess")
else:
    # Fallback to local package directory for development
    local_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, local_dir)

from rewards import compute_reward

REWARDS_HASH = "<sha256 baked at image build>"

def _preflight():
    rewards_path = "/donotaccess/rewards.py"
    if REWARDS_HASH == "<sha256 baked at image build>":
        print("[protean] WARNING: REWARDS_HASH not baked — running in local dev mode")
        return
    if not os.path.exists(rewards_path):
        raise RuntimeError(f"REWARDS_HASH is baked but {rewards_path} does not exist — HALT")
    h = hashlib.sha256(open(rewards_path, "rb").read()).hexdigest()
    if h != REWARDS_HASH:
        raise RuntimeError(f"REWARDS_HASH mismatch {h} != {REWARDS_HASH} — HALT")

def _parse(workdir, override, hidden_root):
    import json
    metadata_path = os.path.join(workdir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        op = meta["op_name"]
        split = meta["split"]
        seed = meta["seed"]
    else:
        op = "elementwise_add_relu"
        split = "train"
        seed = 0

    # Determistically sample shapes
    from shape_sampler import sample_shape
    M, N = sample_shape(op, split, seed)
    
    if override:
        src = override
    else:
        sol_path = os.path.join(workdir, "solution.py")
        if os.path.exists(sol_path):
            with open(sol_path, "r", encoding="utf-8") as f:
                src = f.read()
        else:
            src = ""
            
    dtype = "fp16"
    step = 0
    return op, M, N, dtype, src, split, step

def grade_kernel(op, M, N, dtype, kernel_src, step=0):
    _preflight()
    from protean.subprocess_runner import run_bench  # fail-closed subprocess
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
