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
    )
    return grade | {"op": op, "shape": shape}


def to_eval_result(grade_dict: dict):
    from hud.graders import EvaluationResult, SubScore

    reward = grade_dict["reward"]
    return EvaluationResult(
        score=reward,
        subscores=[SubScore(name="reward", value=reward, weight=1.0)],
        info=grade_dict,
    )
