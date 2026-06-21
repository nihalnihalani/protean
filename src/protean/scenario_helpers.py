"""Per-episode workspace staging. Writes the agent-facing prompt, a writable
solution.py stub, and a self-serve bench.py into /workdir each episode, and NEVER
copies the hidden reference/grader in. See IMPLEMENTATION_PLAN §4.2.
"""
import os

WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/workdir")


def hidden_dir(op_name: str) -> str:
    """Root:700 dir holding grade.py + reference.py for this op. Unreadable by the agent uid."""
    return os.path.join(os.path.dirname(__file__), "tasks", op_name, "donotaccess")


def _resolve_workspace_root() -> str:
    return WORKSPACE_ROOT


def setup_task(op_name: str, split: str, seed: int, validate_mode: str | None = None) -> dict:
    """Stage the workspace for one episode.

    Creates /workdir, writes prompt.md, stages a template solution.py and bench.py,
    and returns task metadata.
    """
    import json
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    metadata = {
        "op_name": op_name,
        "split": split,
        "seed": seed,
    }
    with open(os.path.join(WORKSPACE_ROOT, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    
    # 1. Load prompt from task folder
    task_dir = os.path.join(os.path.dirname(__file__), "tasks", op_name)
    prompt_path = os.path.join(task_dir, "prompt.md")
    
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    else:
        prompt_content = f"Write a optimized Triton kernel for {op_name}."

    with open(os.path.join(WORKSPACE_ROOT, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt_content)

    # 2. Stage solution.py stub based on the operation type
    if op_name == "elementwise_add_relu":
        solution_stub = """import triton
import triton.language as tl
import torch

@triton.jit
def solution_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    add = x + y
    out = tl.where(add > 0, add, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    solution_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
"""
    else:
        solution_stub = """import triton
import triton.language as tl
import torch

# TODO: write your @triton.jit kernel and wrapper function solution()
def solution(*args, **kwargs):
    pass
"""
    with open(os.path.join(WORKSPACE_ROOT, "solution.py"), "w", encoding="utf-8") as f:
        f.write(solution_stub)

    # 3. Stage bench.py (visible helper to the agent, checking correctness/performance)
    if op_name == "elementwise_add_relu":
        eager_def = "def eager(x, y):\n    return torch.relu(x + y)"
        inputs_def = """def make_inputs(shape):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = torch.randn(shape, dtype=torch.float16, device="cuda")
    return (x, y)"""
    else:
        eager_def = "def eager(*args):\n    raise NotImplementedError()"
        inputs_def = "def make_inputs(shape):\n    return ()"

    bench_code = f"""import torch
import time
import sys

{eager_def}

{inputs_def}

try:
    from solution import solution
except ImportError as e:
    print(f"Error importing solution: {{e}}")
    sys.exit(1)

def run():
    shapes = [(1024, 1024), (2048, 2048)]
    for shape in shapes:
        print(f"Testing shape: {{shape}}")
        x, y = make_inputs(shape)
        try:
            out_k = solution(x, y)
        except Exception as e:
            print(f"Compilation/execution error on shape {{shape}}: {{e}}")
            continue
        
        out_r = eager(x, y)
        if not torch.allclose(out_k, out_r, rtol=1e-2, atol=1e-2):
            print("Incorrect output")
            continue
            
        for _ in range(10):
            solution(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            solution(x, y)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 10.0 # average in ms
        print(f"Shape {{shape}}: Correct. Latency = {{latency_ms:.3f}} ms")

if __name__ == "__main__":
    run()
"""
    with open(os.path.join(WORKSPACE_ROOT, "bench.py"), "w", encoding="utf-8") as f:
        f.write(bench_code)

    # 4. Remove calibration script from visible agent files if present
    calib_path = os.path.join(WORKSPACE_ROOT, "check_calibration.py")
    if os.path.exists(calib_path):
        try:
            os.remove(calib_path)
        except Exception:
            pass

    return {"prompt": prompt_content, "seed": seed, "split": split}

