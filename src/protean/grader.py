"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grader.py — build EvaluationResult BY HAND. NEVER hud.graders.combine() (renormalizes, erases hard cap).
from hud.graders import EvaluationResult, SubScore

import os
import importlib.util
from hud.graders import EvaluationResult, SubScore
from .scenario_helpers import WORKSPACE_ROOT, hidden_dir

def _load_grade_module(op_name, hidden_path):
    grade_file = os.path.join(hidden_path, "grade.py")
    spec = importlib.util.spec_from_file_location(f"grade_{op_name}", grade_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate_kernel(op_name: str, split: str, seed: int) -> EvaluationResult:
    # TWO-PATH DESIGN: hidden_dir() returns the in-package path for module loading by the privileged grader/trainer process.
    # The /donotaccess/ path at the root of the filesystem is the canonical copy used for hash-integrity verification.
    hidden = hidden_dir(op_name)
    grade_mod = _load_grade_module(op_name, hidden)
    try:
        r = grade_mod.grade(WORKSPACE_ROOT, None, hidden)
    except Exception as exc:
        return EvaluationResult(
            score=0.0,
            done=True,
            content=f"{op_name}: ungraded ({exc})",
            info={"hard_caps": ["grader_error"]},
            subscores=[]
        )
    return to_eval_result(r)

def to_eval_result(grade_dict: dict) -> EvaluationResult:
    reward = grade_dict.get("reward", 0.0)
    subs = [SubScore(name="reward", value=reward, weight=1.0)]
    if grade_dict.get("caps") or grade_dict.get("hard_caps"):
        # negative-weight hard-cap reconciliation (verilog grader.py pattern)
        subs.append(SubScore(name="hard_cap_penalty", value=reward, weight=-1.0))
    return EvaluationResult(score=reward, subscores=subs, info=grade_dict)
