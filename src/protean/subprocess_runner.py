"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# subprocess_runner.py — fail-closed, restricted globals
import sys
import os
import subprocess
import json
from .anti_hack import ast_clean, delegates_to_matrix_unit

def run_bench(op, M, N, dtype, src, timeout_s=90):
    # Static checks
    ok, why = ast_clean(src)
    if not ok:
        return {"status": why}
    if delegates_to_matrix_unit(src, op):
        return {"status": "delegating_wrapper"}
        
    # Get src path of protean package (so child can import it)
    protean_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Child script to run
    child_script = """import sys, json, os
sys.path.insert(0, sys.argv[1])

import torch
import triton
import triton.language as tl
import importlib

from protean.launch_probe import LaunchCounter
from protean.bench_core import bench_kernel
from protean.task_catalog import OPS_BY_NAME

op_name = sys.argv[2]
M = int(sys.argv[3])
N = int(sys.argv[4])
dtype_str = sys.argv[5]

# Load reference implementation dynamically
spec = OPS_BY_NAME[op_name]
ref_mod = importlib.import_module(spec.reference_module)
eager_fn = ref_mod.eager_fn
make_inputs = ref_mod.make_inputs

# Safe exec environment
user_src = sys.stdin.read()
safe_globals = {
    "triton": triton,
    "tl": tl,
    "torch": torch,
}
import builtins
safe_builtins = {}
for name in dir(builtins):
    if name not in ("__import__", "eval", "exec", "open", "input", "globals", "locals"):
        safe_builtins[name] = getattr(builtins, name)
safe_globals["__builtins__"] = safe_builtins

try:
    exec(user_src, safe_globals)
except Exception as e:
    print(json.dumps({"status": f"compile_error: {type(e).__name__}: {str(e)}"}))
    sys.exit(0)

if "solution" not in safe_globals:
    print(json.dumps({"status": "compile_error: solution() not defined"}))
    sys.exit(0)

solution_fn = safe_globals["solution"]

try:
    with LaunchCounter() as lc:
        res = bench_kernel(
            eager_fn,
            solution_fn,
            make_inputs,
            shape=(M, N),
            dtype=torch.float16 if dtype_str == "fp16" else getattr(torch, dtype_str),
            reps=50,
            warmup=10
        )
    res["launches_timed"] = lc.n
    res["status"] = "ok"
    print(json.dumps(res))
except Exception as e:
    print(json.dumps({"status": f"runtime_error: {type(e).__name__}: {str(e)}"}))
"""
    
    # Inherits TRITON_CACHE_DIR so child uses the warmed /triton-cache.
    env = os.environ.copy()
    env["PYTHONPATH"] = protean_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    # Spawn subprocess
    creationflags = 0
    if sys.platform == "win32":
        creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        
    p = subprocess.Popen(
        [sys.executable, "-c", child_script, protean_dir, op, str(M), str(N), dtype],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        env=env
    )
    
    try:
        stdout, stderr = p.communicate(input=src, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            p.kill()
        else:
            import signal
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
        return {"status": "timeout"}
        
    if p.returncode != 0:
        return {"status": "crash", "detail": stderr}
        
    try:
        return json.loads(stdout.strip())
    except Exception as e:
        return {"status": "crash", "detail": f"Failed to parse stdout: {stdout}. Error: {e}"}

