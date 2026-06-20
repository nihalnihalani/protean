"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# calibrate.py — 2-stage gate + 10-step micro-probe + hash preflight
def calibrate(tasks):
    # PREFLIGHT (line 1): print both hashes, assert grade.py reward function path matches
    assert _hash("/donotaccess/rewards.py") == REWARDS_HASH
    assert grade_kernel(known_good)["reward"] > 0 and grade_kernel(known_bad)["reward"] == 0

    rollouts = collect(tasks, n=8)               # base/SFT ckpt
    # STAGE A: compile≥5%, allclose≥2%, group reward std > 0.05, ≥1 group w/ 2 distinct values
    # STAGE A also: median pr_frac over 50 rollouts > 0.1  (else TAU=0.5 gate would nuke all)
    # STAGE B (anti compile-only-variance): ≥1 NONZERO-SPEEDUP rollout on ≥2 of the tasks
    if not stage_A(rollouts): return escalate()  # ladder below
    if not stage_B(rollouts): return escalate()
    return "GO"

def escalate():
    # ladder: P_TARGET 1.5→1.2→1.1 (config edit, no rebuild) → L1-elementwise curriculum
    #         → enable bootstrap_credit (anneals out) → as last resort kick L1-only
    ...
