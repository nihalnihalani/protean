"""Direct grader used by HUD, scripts, and future training code."""

from __future__ import annotations

import importlib.util
from typing import Any

from protean.anti_hack import ast_clean, contains_triton_jit
from protean.rewards import compute_reward
from protean.splits import Split, shapes_for_split
from protean.task_catalog import get_op


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def grade_source(
    src: str,
    *,
    op: str = "elementwise_add_relu",
    split: Split = "held_out",
    shape: int | None = None,
    reps: int = 50,
    warmup: int = 10,
) -> dict[str, Any]:
    spec = get_op(op)
    shape = shape or shapes_for_split(split)[0]

    ok, reason = ast_clean(src)
    if not ok:
        return compute_reward(
            correct=False,
            speedup=0.0,
            launches_timed=0,
            dtype_ok=False,
            shape_ok=False,
            split=split,
            caps=[reason],
        ) | {"op": op, "shape": shape}

    if not contains_triton_jit(src):
        return compute_reward(
            correct=False,
            speedup=0.0,
            launches_timed=0,
            dtype_ok=False,
            shape_ok=False,
            split=split,
            caps=["no_triton_jit"],
        ) | {"op": op, "shape": shape}

    if not _cuda_available():
        return compute_reward(
            correct=False,
            speedup=0.0,
            launches_timed=0,
            dtype_ok=False,
            shape_ok=False,
            split=split,
            caps=["cuda_unavailable"],
        ) | {"op": op, "shape": shape}

    from protean.bench_core import bench_source

    bench = bench_source(src, n=shape, split=split, spec=spec, reps=reps, warmup=warmup)
    grade = compute_reward(
        correct=bench["correct"],
        speedup=bench["speedup"],
        launches_timed=bench["launches_timed"],
        dtype_ok=bench["dtype_ok"],
        shape_ok=bench["shape_ok"],
        split=split,
        t_eager_ms=bench["t_eager_ms"],
        t_kernel_ms=bench["t_kernel_ms"],
        pr_frac=bench.get("pr_frac", 0.0),
    )
    return grade | {"op": op, "shape": shape}


def to_eval_result(grade_dict: dict):
    raw_reward = float(grade_dict["reward"])
    hud_reward = max(0.0, min(raw_reward / 2.0, 1.0))
    correct = bool(grade_dict.get("correct", False)) and not grade_dict.get("caps")
    speedup_score = max(0.0, min(float(grade_dict.get("speedup_score", 0.0)), 1.0))
    held_out = 1.0 if grade_dict.get("split") == "held_out" and correct else 0.0
    anti_hack = 1.0 if not grade_dict.get("caps") else 0.0
    compile_success = 1.0 if "cuda_unavailable" not in grade_dict.get("caps", []) else 0.0
    info = {
        **grade_dict,
        "protean_reward_raw": raw_reward,
        "hud_reward_normalized": hud_reward,
    }
    try:
        from hud.graders import EvaluationResult, SubScore
    except Exception:  # pragma: no cover - local tests should not require HUD.
        from dataclasses import dataclass

        @dataclass
        class SubScore:
            name: str
            value: float
            weight: float

        @dataclass
        class EvaluationResult:
            reward: float
            subscores: list[SubScore]
            info: dict

    return EvaluationResult(
        reward=hud_reward,
        subscores=[
            SubScore(name="hud_reward", value=hud_reward, weight=1.0),
            SubScore(name="correctness", value=1.0 if correct else 0.0, weight=0.0),
            SubScore(name="speedup", value=speedup_score, weight=0.0),
            SubScore(name="held_out", value=held_out, weight=0.0),
            SubScore(name="anti_hack", value=anti_hack, weight=0.0),
            SubScore(name="compile_success", value=compile_success, weight=0.0),
        ],
        info=info,
    )
