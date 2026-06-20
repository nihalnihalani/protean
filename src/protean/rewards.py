"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# rewards.py  — baked ONLY to /donotaccess/rewards.py
import json, os, hashlib

_CFG_PATH = os.path.join(os.path.dirname(__file__), "reward_config.json")

def _cfg():
    with open(_CFG_PATH) as f:
        return json.load(f)

def compute_reward(*, correct: bool, speedup: float, pr_frac: float,
                   launches_timed: int, dtype_ok: bool, shape_ok: bool,
                   step: int = 0) -> dict:
    """
    CANONICAL formula. Bounded [0, 2.0] for GRPO stability.
    Imported byte-identically by grade.py AND (via grade_kernel) by the trainer.

    daVinci PR is an ADDITIVE BONUS (matches Eq1 continuity), NOT a multiplicative
    gate during the learning phase. The only hard zeros are the ROBUST gates:
    correctness and launch-count. This prevents zeroing correct-but-slow early-7B
    kernels (the gradient-collapse failure mode).
    """
    c = _cfg()
    P_TARGET      = c["P_TARGET"]        # 1.5
    SPEEDUP_FLOOR = c["SPEEDUP_FLOOR"]   # 1.1  dead-band (anti dtype/noise)
    SPEEDUP_CAP   = c["SPEEDUP_CAP"]     # 20.0 hard backstop vs cache-exploit fake speedups
    CORRECT_FLOOR = c["CORRECT_FLOOR"]   # 0.3  (Kevin) keeps reward dense
    PR_BONUS      = c["PR_BONUS"]        # 0.2
    TAU           = c["TAU"]             # 0.5  PR dominance threshold
    PR_HARD_GATE  = c.get("PR_HARD_GATE_AFTER_STEP", 10**9)  # default off; flip to 50 only if hack observed

    caps = []
    # ---- ROBUST HARD ZEROS (never penalize a correct-but-slow kernel on PR) ----
    if not (correct and dtype_ok and shape_ok):
        return {"reward": 0.0, "caps": ["incorrect"], "speedup": speedup, "pr": pr_frac}
    if launches_timed <= 0:
        return {"reward": 0.0, "caps": ["no_triton_launch"], "speedup": speedup, "pr": pr_frac}

    # ---- OPTIONAL phase-2 PR dominance gate (config-flippable, off by default) ----
    pr_ok = (pr_frac > TAU)
    if step >= PR_HARD_GATE and not pr_ok:
        return {"reward": 0.0, "caps": ["pr_dominance_gate"], "speedup": speedup, "pr": pr_frac}

    # ---- speedup score (dead-band, normalized, capped) ----
    if speedup < SPEEDUP_FLOOR:
        speedup_score = 0.0
    else:
        speedup_score = min(speedup, SPEEDUP_CAP) / P_TARGET

    pr_term = PR_BONUS * max(0.0, min(pr_frac, 1.0))   # additive bonus, never a killer
    reward = CORRECT_FLOOR + speedup_score + pr_term
    reward = max(0.0, min(reward, 2.0))
    return {"reward": reward, "caps": caps, "speedup": speedup, "pr": pr_frac}


# Optional bootstrap credit (anneals out) — ONLY used if calibration is too sparse.
def bootstrap_credit(compiles: bool, imports_triton: bool, step: int, anneal_steps: int = 40) -> float:
    if step >= anneal_steps:
        return 0.0
    bonus = (0.1 if compiles else 0.0) + (0.2 if imports_triton else 0.0)
    return bonus * (1.0 - step / anneal_steps)
