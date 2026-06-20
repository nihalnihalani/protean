"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grader.py — build EvaluationResult BY HAND. NEVER hud.graders.combine() (renormalizes, erases hard cap).
from hud.graders import EvaluationResult, SubScore

def to_eval_result(grade_dict: dict) -> EvaluationResult:
    reward = grade_dict["reward"]
    subs = [SubScore(name="reward", value=reward, weight=1.0)]
    if grade_dict.get("caps"):
        # negative-weight hard-cap reconciliation (verilog grader.py pattern)
        subs.append(SubScore(name="hard_cap_penalty", value=reward, weight=-1.0))
    return EvaluationResult(score=reward, subscores=subs, info=grade_dict)
