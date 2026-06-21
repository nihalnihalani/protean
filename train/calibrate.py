"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# calibrate.py — 2-stage gate + 10-step micro-probe + hash preflight
import os
import hashlib
import json
import numpy as np

REWARDS_HASH = "<sha256 baked at image build>"

def _hash(path):
    if not os.path.exists(path):
        return ""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def _load_grade_kernel(op):
    path = f"/donotaccess/{op}/grade.py"
    if not os.path.exists(path):
        # Fallback to local path
        local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(local_dir, "src", "protean", "tasks", op, "donotaccess", "grade.py")
        if not os.path.exists(path):
            path = os.path.join(local_dir, "protean", "tasks", op, "donotaccess", "grade.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"grade_{op}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.grade_kernel

def calibrate(tasks, base_model_runner=None):
    """
    2-stage calibration check.
    base_model_runner: callable prompt -> list[completions]
    """
    rewards_path = "/donotaccess/rewards.py"
    if REWARDS_HASH == "<sha256 baked at image build>":
        print("[protean] WARNING: REWARDS_HASH not baked — running in local dev mode")
    else:
        if not os.path.exists(rewards_path):
            raise RuntimeError(f"REWARDS_HASH is baked but {rewards_path} does not exist — HALT")
        h = _hash(rewards_path)
        if h != REWARDS_HASH:
            raise RuntimeError(f"REWARDS_HASH mismatch {h} != {REWARDS_HASH} — HALT")

    gk = _load_grade_kernel("elementwise_add_relu")
    
    known_good = """import triton
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
    known_bad = """def solution(x, y):
    return x
"""
    
    # 1. Preflight checks
    good_res = gk("elementwise_add_relu", 256, 256, "fp16", known_good)
    bad_res = gk("elementwise_add_relu", 256, 256, "fp16", known_bad)
    
    assert good_res["reward"] > 0.0, "Preflight: known_good kernel scored 0.0"
    assert bad_res["reward"] == 0.0, "Preflight: known_bad kernel scored nonzero"

    if base_model_runner is None:
        print("Stub preflight OK. No base model runner provided, skipping Stage A/B.")
        return "GO"

    # 2. Query model and run Stage A/B evaluation
    rollouts = []
    for task in tasks[:8]:
        op_name = task.get("op_name") or task.get("op")
        completions = base_model_runner(task["prompt"])
        for completion in completions:
            res = gk(op_name, task["M"], task["N"], task["dtype"], completion)
            rollouts.append(res)

    if not rollouts:
        return escalate()

    compiled = [r for r in rollouts if "compile_error" not in r.get("caps", [])]
    compile_rate = len(compiled) / len(rollouts)
    allclose_rate = sum(1 for r in rollouts if r.get("correct", False)) / len(rollouts)
    
    rewards = [r["reward"] for r in rollouts]
    reward_std = np.std(rewards)
    
    prs = [r.get("pr", 0.0) for r in rollouts]
    median_pr = np.median(prs)

    print(f"Calibration metrics: Compile={compile_rate:.2f}, Allclose={allclose_rate:.2f}, Std={reward_std:.3f}, Med_PR={median_pr:.2f}")

    stage_a_passed = (compile_rate >= 0.05) and (allclose_rate >= 0.02) and (reward_std > 0.05)
    
    tasks_with_speedup = set()
    for i, r in enumerate(rollouts):
        if r.get("correct") and r.get("speedup", 0.0) >= 1.0:
            task_idx = i // max(1, len(rollouts) // 8)
            tasks_with_speedup.add(task_idx)
    stage_b_passed = len(tasks_with_speedup) >= 2

    if not stage_a_passed or not stage_b_passed:
        return escalate()
        
    return "GO"

def escalate():
    print("Calibration failed. Escalating configuration...")
    # Option A: We run the trainer as root (unlike the demoted agent), so we can write to /donotaccess/reward_config.json
    config_path = "/donotaccess/reward_config.json"
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg["P_TARGET"] = 1.1
        cfg["CORRECT_FLOOR"] = 0.5
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[protean] Escalated reward targets in {config_path}: P_TARGET=1.1, CORRECT_FLOOR=0.5")
        return "GO_ESCALATED"
    except PermissionError as e:
        raise RuntimeError(
            f"Calibration escalation failed: cannot write to {config_path}. "
            f"This means the trainer cannot lower reward targets when the base model is too weak. "
            f"Check that the trainer process runs as root, or that {config_path} is writable by the trainer uid. "
            f"Original error: {e}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Calibration escalation failed: {config_path} does not exist. "
            f"This means the Dockerfile did not copy reward_config.json to the canonical /donotaccess/ path. "
            f"Re-check step 2 of the build. Original error: {e}"
        ) from e
