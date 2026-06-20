"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# reward.py — calls the grader; HAS NO FORMULA of its own
import importlib.util, os
from pathlib import Path

def _load_grade(op):
    spec = importlib.util.spec_from_file_location(
        f"grade_{op}", f"/donotaccess/{op}/grade.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def make_reward_fn(op, M, N, dtype, step_ref):
    gm = _load_grade(op)
    def reward_fn(completions, **kw):            # trl reward_funcs signature
        out = []
        for c in completions:
            src = _extract_code(c)
            out.append(gm.grade_kernel(op, M, N, dtype, src, step=step_ref[0])["reward"])
        return out
    return reward_fn
