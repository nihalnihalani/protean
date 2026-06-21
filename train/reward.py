"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# reward.py — calls the grader; HAS NO FORMULA of its own
import importlib.util, os
from pathlib import Path

def _extract_code(completion: str) -> str:
    if "```python" in completion:
        code = completion.split("```python")[1].split("```")[0]
        return code.strip()
    elif "```" in completion:
        code = completion.split("```")[1].split("```")[0]
        return code.strip()
    return completion.strip()

def _load_grade(op):
    path = f"/donotaccess/{op}/grade.py"
    if not os.path.exists(path):
        # Local development fallback
        local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(local_dir, "src", "protean", "tasks", op, "donotaccess", "grade.py")
        if not os.path.exists(path):
            path = os.path.join(local_dir, "protean", "tasks", op, "donotaccess", "grade.py")
    spec = importlib.util.spec_from_file_location(f"grade_{op}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
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
