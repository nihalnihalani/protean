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
