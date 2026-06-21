"""Statistically-powered held-out evaluation (fixes the n=24 underpowering; OPEN_ISSUES).

Design (docs/TECHNICAL_SPEC.md §5.4):
  * UNIT = held-out task (op, shape), NOT a rollout (no pseudoreplication).
  * SCALE = N_OPS x N_HELDOUT_PER_OP = 5 x 40 = 200 paired tasks over CONTINUOUS off-grid shapes.
  * PAIRED = base and trained graded on the SAME tasks; endpoint = per-task delta D(t)=r_trained(t)-r_base(t).
  * CONTINUOUS endpoint = per-task mean reward in [0,2] over R rollouts (more info than clustered Bernoulli).
  * MAGNITUDE+CI = hierarchical bootstrap (resample ops, then shapes within op).
  * SIGNIFICANCE = across-op SIGN TEST on per-op mean delta (clustering-immune: K ops all positive -> (1/2)^K).

Pure-Python (no numpy/scipy). Rewards must come from the single grader authority; this module only aggregates.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from protean.splits import N_OPS, N_HELDOUT_PER_OP, sample_heldout_shape


# ---- normal quantile (Acklam) for power/MDE math ----
def _norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mde_paired(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable standardized paired effect d_z = E[D]/sd(D)."""
    return (_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) / math.sqrt(n)


def required_n_paired(d_z: float, alpha: float = 0.05, power: float = 0.80) -> int:
    return math.ceil(((_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) / d_z) ** 2)


def _binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k, n + 1))


def across_op_sign_test(per_op_delta: dict) -> dict:
    """One-sided sign test on per-op mean deltas; p independent of within-op correlation."""
    n = len(per_op_delta)
    pos = sum(1 for d in per_op_delta.values() if d > 0)
    return {"ops": n, "positive": pos, "p_one_sided": _binom_tail_ge(pos, n, 0.5)}


def _grand_mean(by_op: dict) -> float:
    flat = [v for d in by_op.values() for v in d]
    return sum(flat) / len(flat)


def _std(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def hierarchical_bootstrap_ci(deltas_by_op: dict, B: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Resample ops with replacement, then shapes within op; percentile CI on the grand mean of D."""
    rng = random.Random(seed)
    ops = list(deltas_by_op)
    point = _grand_mean(deltas_by_op)
    boots = []
    for _ in range(B):
        chosen = [rng.choice(ops) for _ in ops]
        vals = []
        for op in chosen:
            d = deltas_by_op[op]
            vals.extend(d[rng.randrange(len(d))] for _ in range(len(d)))
        boots.append(sum(vals) / len(vals))
    boots.sort()
    lo = boots[int((1 - ci) / 2 * B)]
    hi = boots[int((1 + ci) / 2 * B) - 1]
    return {"gap": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0 or hi < 0)}


def build_eval_set(ops: list, n_per_op: int = N_HELDOUT_PER_OP, seed: int = 0) -> list:
    """Frozen, deterministic held-out task set (paired across policies)."""
    tasks = []
    for op in ops:
        rng = random.Random(int(__import__("hashlib").sha256(f"{op}|{seed}".encode()).hexdigest()[:16], 16))
        for idx in range(n_per_op):
            tasks.append({"op": op, "idx": idx, "shape": sample_heldout_shape(rng),
                          "dtype": "float16", "split": "held_out"})
    return tasks


def evaluate_policy(grade_fn: Callable[[dict], float], tasks: list, rollouts: int = 8) -> dict:
    """Per-task MEAN reward over `rollouts` samples. grade_fn(task)->reward in [0,2]. Key = (op, idx)."""
    out = {}
    for t in tasks:
        rs = [grade_fn(t) for _ in range(rollouts)]
        out[(t["op"], t["idx"])] = sum(rs) / len(rs)
    return out


def paired_report(base: dict, trained: dict, B: int = 10000) -> dict:
    """Money report: paired per-task deltas -> hierarchical bootstrap CI + across-op sign test + power."""
    keys = sorted(set(base) & set(trained))
    by_op = defaultdict(list)
    for (op, idx) in keys:
        by_op[op].append(trained[(op, idx)] - base[(op, idx)])
    per_op = {op: sum(d) / len(d) for op, d in by_op.items()}
    all_d = [v for d in by_op.values() for v in d]
    boot = hierarchical_bootstrap_ci(by_op, B=B)
    sign = across_op_sign_test(per_op)
    n = len(all_d)
    sd = _std(all_d)
    return {
        "n_tasks": n, "n_ops": len(by_op),
        "gap_mean": boot["gap"], "ci95": (boot["ci_lo"], boot["ci_hi"]),
        "gap_ci_excludes_zero": boot["excludes_zero"],
        "per_op_delta": per_op, "sign_test": sign,
        "observed_dz": (sum(all_d) / n) / sd if sd > 0 else float("inf"),
        "mde_dz_at_n": mde_paired(n),
        "powered": boot["excludes_zero"] or sign["p_one_sided"] <= 0.05,
    }


if __name__ == "__main__":
    rng = random.Random(7)
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    print(f"eval set: {len(tasks)} held-out tasks / {N_OPS} ops; sample shapes: {[t['shape'] for t in tasks[:6]]}")
    op_off = {op: rng.gauss(0, 0.15) for op in ops}
    EFFECT = 0.18
    base_g = lambda t: max(0.0, 0.30 + op_off[t["op"]] + rng.gauss(0, 0.25))      # noqa: E731
    train_g = lambda t: max(0.0, 0.30 + EFFECT + op_off[t["op"]] + rng.gauss(0, 0.25))  # noqa: E731
    rep = paired_report(evaluate_policy(base_g, tasks), evaluate_policy(train_g, tasks), B=5000)
    print(f"Gap={rep['gap_mean']:.4f}  95%CI=[{rep['ci95'][0]:.4f},{rep['ci95'][1]:.4f}]  excl0={rep['gap_ci_excludes_zero']}")
    print(f"sign test: {rep['sign_test']['positive']}/{rep['sign_test']['ops']} ops, p={rep['sign_test']['p_one_sided']:.4f}")
    print(f"observed d_z={rep['observed_dz']:.3f}  MDE@n={rep['n_tasks']}={rep['mde_dz_at_n']:.3f}  POWERED={rep['powered']}")
    print(f"MDE @ old n=3 = {mde_paired(3):.3f}  ->  @ n={rep['n_tasks']} = {mde_paired(rep['n_tasks']):.3f}")
