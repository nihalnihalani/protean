"""Calibration checks for GRPO training runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import pstdev


REWARDS_HASH = "<sha256 baked at image build>"
CANONICAL_REWARDS = Path("/donotaccess/rewards.py")
CANONICAL_CONFIG = Path("/donotaccess/reward_config.json")
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_rewards_hash() -> bool:
    """Fail loudly if Docker baked a hash and the hidden reward code drifted."""

    if REWARDS_HASH == "<sha256 baked at image build>":
        return False
    if not CANONICAL_REWARDS.exists():
        raise RuntimeError(f"REWARDS_HASH is baked but {CANONICAL_REWARDS} does not exist")
    actual = _sha256(CANONICAL_REWARDS)
    if actual != REWARDS_HASH:
        raise RuntimeError(f"REWARDS_HASH mismatch {actual} != {REWARDS_HASH}")
    return True


def escalate_reward_config(*, p_target: float = 1.1, correct_floor: float = 0.5) -> dict:
    """Deliberately lower reward targets after calibration failure."""

    if not CANONICAL_CONFIG.exists():
        raise RuntimeError(f"reward config missing: {CANONICAL_CONFIG}")
    try:
        data = json.loads(CANONICAL_CONFIG.read_text())
        data["P_TARGET"] = float(p_target)
        data["CORRECT_FLOOR"] = float(correct_floor)
        CANONICAL_CONFIG.write_text(json.dumps(data, indent=2) + "\n")
        return data
    except PermissionError as exc:
        raise RuntimeError(f"reward config is not writable by trainer: {CANONICAL_CONFIG}") from exc


def _task_value(task, key: str, default=None):
    if isinstance(task, dict):
        return task.get(key, default)
    columns = getattr(task, "columns", None)
    if isinstance(columns, dict):
        return columns.get(key, default)
    return getattr(task, key, default)


def _prompt_for_task(task) -> str:
    prompt = _task_value(task, "prompt")
    if prompt:
        return str(prompt)
    op = _task_value(task, "op")
    if not op:
        raise ValueError("calibration task is missing op")
    path = ROOT / "src" / "protean" / "tasks" / str(op) / "prompt.md"
    return path.read_text()


def _shape_for_task(task) -> int:
    shape = _task_value(task, "shape")
    if shape is not None:
        return int(shape)
    if _task_value(task, "M") is not None:
        return int(_task_value(task, "M"))
    return 1024


def calibrate(tasks=None, base_model_runner=None) -> str:
    """Go/no-go preflight plus optional real base-model rollout calibration."""

    if os.environ.get("PROTEAN_SKIP_CALIBRATION") == "1":
        return "SKIPPED"
    verify_rewards_hash()
    if CANONICAL_CONFIG.exists():
        json.loads(CANONICAL_CONFIG.read_text())

    # Preflight KNOWN_GOOD / KNOWN_BAD verification for registered ops
    import torch
    if torch.cuda.is_available():
        from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
        import importlib.util

        def _load_grade_kernel(op_name: str):
            local_path = ROOT / "src" / "protean" / "tasks" / op_name / "donotaccess" / "grade.py"
            if not local_path.exists():
                local_path = Path("/donotaccess") / op_name / "grade.py"
            if local_path.exists():
                spec = importlib.util.spec_from_file_location(f"grade_wrapper_{op_name}", local_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "grade_kernel"):
                        gk = mod.grade_kernel
                        return lambda op, shape_m, shape_n, dtype, src: gk(op, shape_n, dtype, src)
            from protean.grader import grade_source
            return lambda op, shape_m, shape_n, dtype, src: grade_source(src, op=op, shape=shape_n)

        KNOWN_GOOD_KERNELS = {
            "elementwise_add_relu": HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
            "softmax_rows": """import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, stride_row, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * stride_row
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols
    row = tl.load(row_start + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    tl.store(out_ptr + row_idx * stride_row + offsets, row_exp / row_sum, mask=mask)

def solution(x):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, output, M, N, x.stride(0), BLOCK_N=BLOCK_N)
    return output
""",
        }

        KNOWN_BAD = """def solution(*args):
    return args[0]
"""

        for op_name, good_src in KNOWN_GOOD_KERNELS.items():
            gk = _load_grade_kernel(op_name)
            good_res = gk(op_name, 256, 256, "fp16", good_src)
            bad_res = gk(op_name, 256, 256, "fp16", KNOWN_BAD)
            assert good_res["reward"] > 0.0, f"Preflight: {op_name} known_good kernel scored 0.0"
            assert bad_res["reward"] == 0.0, f"Preflight: {op_name} known_bad kernel scored nonzero"

    if base_model_runner is None or not tasks:
        return "GO"

    from protean.grader import grade_source

    rollouts: list[dict] = []
    for task in list(tasks)[:8]:
        op = str(_task_value(task, "op"))
        split = str(_task_value(task, "split", "train"))
        shape = _shape_for_task(task)
        dtype = str(_task_value(task, "dtype", "float16"))
        prompt = _prompt_for_task(task)
        completions = base_model_runner(prompt)
        for completion in completions:
            rollouts.append(grade_source(str(completion), op=op, split=split, shape=shape, dtype=dtype))

    if not rollouts:
        return escalate_reward_config() and "GO_ESCALATED"

    compile_rate = sum(1 for row in rollouts if "compile_error" not in row.get("caps", [])) / len(rollouts)
    allclose_rate = sum(1 for row in rollouts if row.get("correct")) / len(rollouts)
    rewards = [float(row.get("reward", 0.0)) for row in rollouts]
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    tasks_with_speedup = {
        idx
        for idx, row in enumerate(rollouts)
        if row.get("correct") and float(row.get("speedup", 0.0)) >= 1.0
    }

    print(
        "[protean] calibration rollouts: "
        f"compile_rate={compile_rate:.2f}, allclose_rate={allclose_rate:.2f}, reward_std={reward_std:.3f}"
    )
    stage_a = compile_rate >= 0.05 and allclose_rate >= 0.02 and reward_std > 0.05
    stage_b = len(tasks_with_speedup) >= 1
    if not stage_a or not stage_b:
        escalate_reward_config()
        return "GO_ESCALATED"
    return "GO"
