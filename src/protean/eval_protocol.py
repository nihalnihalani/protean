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
from collections.abc import Callable
from typing import TYPE_CHECKING

from protean.splits import N_HELDOUT_PER_OP, N_OPS, REAL_OPS, sample_heldout_shape

if TYPE_CHECKING:
    from pathlib import Path


# ---- normal quantile (Acklam) for power/MDE math ----
def _norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0,1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def mde_paired(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable standardized paired effect d_z = E[D]/sd(D)."""
    return (_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) / math.sqrt(n)


def required_n_paired(d_z: float, alpha: float = 0.05, power: float = 0.80) -> int:
    return math.ceil(((_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) / d_z) ** 2)


def power_at(n: int, d_z: float, alpha: float = 0.05) -> float:
    """Achieved power of a two-sided paired t/z test for standardized effect d_z at sample size n.

    Normal-approx power: 1 - beta = Phi(|d_z|*sqrt(n) - z_{1-alpha/2}) (the negligible lower-tail term is
    dropped). Inverse of mde_paired/required_n_paired and used by power_curve below. Returns power in
    [0,1]. (Standard retrospective-power identity; cf. Hidden Costs of RLVR arxiv:2509.21882 saturation.)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ncp = abs(d_z) * math.sqrt(n)
    return max(0.0, min(1.0, _norm_cdf(ncp - _norm_ppf(1 - alpha / 2))))


def power_curve(d_z: float, ns: list | None = None, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Power vs sample-size curve for a fixed standardized effect d_z (pure CPU; no GPU, no grader).

    Returns {"d_z", "alpha", "target_power", "n_for_target", "curve": [(n, power), ...]} where `curve`
    pairs each n in `ns` with its achieved power, and `n_for_target` is required_n_paired(d_z) — the
    smallest n reaching `power`. `ns` defaults to a spread that brackets the n=3 -> n=200 design jump.
    Lets a caller plot the saturation curve and justify the powered held-out scale-up. (research:
    paired-bootstrap 2511.19794; Hidden Costs of RLVR saturation curves arxiv:2509.21882.)
    """
    if ns is None:
        ns = [3, 5, 10, 25, 50, 100, 150, 200]
    return {
        "d_z": d_z,
        "alpha": alpha,
        "target_power": power,
        "n_for_target": required_n_paired(d_z, alpha=alpha, power=power),
        "curve": [(n, power_at(n, d_z, alpha=alpha)) for n in ns],
    }


def _binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


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


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf (pure stdlib; pairs with _norm_ppf for BCa)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Minimum number of ops required before BCa is allowed: the acceleration term is estimated by an
# op-level jackknife, which is unreliable with very few leave-one-out points. arxiv:2404.12967 (2024)
# found BCa offers little over percentile and can be worse in heavy-tailed / tiny-sample settings, so we
# gate BCa on n_ops >= BCA_MIN_OPS and otherwise fall back to percentile (recorded in the returned dict).
BCA_MIN_OPS = 5


def _lo_index(p: float, B: int) -> int:
    """Lower-tail index into a length-B SORTED list for quantile p; floor(p*B) clamped to [0, B-1].

    Matches the percentile branch's historical lo convention `boots[int(p*B)]` exactly (so the default
    percentile path stays byte-identical) while clamping tiny/extreme p in range.
    """
    return max(0, min(int(p * B), B - 1))


def _hi_index(p: float, B: int) -> int:
    """Upper-tail index into a length-B SORTED list for quantile p; floor(p*B)-1 clamped to [0, B-1].

    Matches the percentile branch's historical hi convention `boots[int(p*B) - 1]` exactly. The `-1`
    brings the interval one rank INSIDE the sorted array so an extreme BCa adjustment (large positive z0
    and/or a_hat pushing a_hi toward 1.0) can never silently return the single most extreme replicate as
    the upper bound: a_hi==1.0 maps to boots[B-1] (the max) and an interior a_hi rounds toward the
    interior, the same rule the lower bound uses. This unifies the two paths so they cannot drift.
    """
    return max(0, min(int(p * B) - 1, B - 1))


def hierarchical_bootstrap_ci(
    deltas_by_op: dict, B: int = 10000, seed: int = 0, ci: float = 0.95, method: str = "percentile"
) -> dict:
    """Hierarchical bootstrap CI on the grand mean of D (resample ops, then shapes within op).

    method:
      * "percentile" (default, unchanged behavior) — plain percentile interval on the bootstrap dist.
      * "bca"        — bias-corrected & accelerated (Efron 1987). Bias z0 from the fraction of bootstrap
                       replicates below the point estimate; acceleration a_hat from an OP-LEVEL jackknife
                       (respects the two-level design). Requires len(deltas_by_op) >= BCA_MIN_OPS; with
                       fewer ops the jackknife a_hat is unreliable, so we fall back to percentile and set
                       ci_method="percentile" in the result. (research: paired-bootstrap 2511.19794;
                       comparative study arxiv:2404.12967; Erik Drysdale BCa recipe.)

    Returned keys are a superset of the prior contract (gap, ci_lo, ci_hi, excludes_zero) plus
    ci_method (the method actually used after any fallback) so callers can record which interval was taken.
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got {method!r}")
    rng = random.Random(seed)
    ops = list(deltas_by_op)
    point = _grand_mean(deltas_by_op)
    boots = []
    for _ in range(B):
        chosen = [rng.choice(ops) for _ in ops]
        vals: list[float] = []
        for op in chosen:
            d = deltas_by_op[op]
            vals.extend(d[rng.randrange(len(d))] for _ in range(len(d)))
        boots.append(sum(vals) / len(vals))
    boots.sort()

    used = method
    if method == "bca" and len(ops) < BCA_MIN_OPS:
        used = "percentile"  # not enough ops for a trustworthy jackknife acceleration

    if used == "bca":
        # bias-correction z0: inverse-normal of the fraction of replicates strictly below the point est.
        n_below = sum(1 for b in boots if b < point)
        frac = n_below / B
        frac = min(max(frac, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))  # clamp off 0/1 so ppf is finite
        z0 = _norm_ppf(frac)
        # acceleration a_hat from an op-level jackknife on the grand mean (leave-one-op-out).
        # NOTE: we use the RAW leave-one-out grand means as the jackknife replicates here, not Efron
        # pseudo-values theta_i = n*theta_hat - (n-1)*theta_(i). In the standard a_hat ratio
        # sum((jbar - j)^3) / [6 * (sum((jbar - j)^2))^1.5] the (n-1)/n scale of the pseudo-value
        # transform appears in both numerator and denominator and cancels, so raw leave-one-out estimates
        # give the same a_hat (this matches scipy.stats.bootstrap / Davison & Hinkley 1997).
        jack = []
        for skip in ops:
            kept = {op: deltas_by_op[op] for op in ops if op != skip}
            jack.append(_grand_mean(kept))
        jbar = sum(jack) / len(jack)
        num = sum((jbar - j) ** 3 for j in jack)
        den = 6.0 * (sum((jbar - j) ** 2 for j in jack)) ** 1.5
        a_hat = num / den if den != 0 else 0.0
        z_lo = _norm_ppf((1 - ci) / 2)
        z_hi = _norm_ppf((1 + ci) / 2)

        def _adj(zq: float) -> float:
            denom = 1.0 - a_hat * (z0 + zq)
            if denom == 0:
                denom = 1e-12
            return _norm_cdf(z0 + (z0 + zq) / denom)

        a_lo = min(max(_adj(z_lo), 0.0), 1.0)
        a_hi = min(max(_adj(z_hi), 0.0), 1.0)
        # Guard against a pathological adjustment inverting the interval (e.g. a_hat or z0 so large the
        # adjusted quantiles cross). With valid inputs a_lo < a_hi always holds.
        assert a_lo <= a_hi, f"BCa adjusted quantiles inverted: a_lo={a_lo} > a_hi={a_hi}"
        # SAME index convention as the percentile branch (via the shared helpers) so the two paths cannot
        # drift: the upper bound is brought one rank inside, never returning the lone extreme replicate.
        lo = boots[_lo_index(a_lo, B)]
        hi = boots[_hi_index(a_hi, B)]
    else:
        lo = boots[_lo_index((1 - ci) / 2, B)]
        hi = boots[_hi_index((1 + ci) / 2, B)]

    return {"gap": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo > 0 or hi < 0), "ci_method": used}


def build_eval_set(ops: list, n_per_op: int = N_HELDOUT_PER_OP, seed: int = 0) -> list:
    """Frozen, deterministic held-out task set (paired across policies)."""
    tasks = []
    for op in ops:
        rng = random.Random(int(__import__("hashlib").sha256(f"{op}|{seed}".encode()).hexdigest()[:16], 16))
        for idx in range(n_per_op):
            tasks.append(
                {"op": op, "idx": idx, "shape": sample_heldout_shape(rng), "dtype": "float16", "split": "held_out"}
            )
    return tasks


def _crn_seed(base_seed: int, op: str, idx: int, k: int) -> int:
    """Deterministic per-(task, rollout-index) seed for common-random-numbers pairing."""
    import hashlib

    h = hashlib.sha256(f"{base_seed}|{op}|{idx}|{k}".encode()).hexdigest()[:16]
    return int(h, 16)


def evaluate_policy(grade_fn: Callable, tasks: list, rollouts: int = 8, crn_seed: int | None = None) -> dict:
    """Per-task MEAN reward over `rollouts` samples. grade_fn(task)->reward in [0,2]. Key = (op, idx).

    Common-Random-Numbers (CRN) variance reduction for paired evaluation: when `crn_seed` is provided,
    the k-th rollout of every task is graded with a deterministic rollout-indexed seed derived from
    (crn_seed, op, idx, k), and `grade_fn` is called as grade_fn(task, rollout_seed=<int>). Pairing the
    k-th base rollout with the k-th trained rollout on the SAME seed cancels shared input noise from the
    per-task delta, lowering the variance of the paired endpoint (Law & Kelton; Sharma arxiv:2512.24145).

    When `crn_seed` is None (default) behavior is unchanged: grade_fn is called as grade_fn(task), so
    existing single-argument graders keep working byte-for-byte. For deterministic graders CRN is a no-op
    on the value but the explicit seeding still pins base/trained to identical inputs, guarding against a
    future stochastic grade_fn silently breaking the paired design.
    """
    out = {}
    for t in tasks:
        if crn_seed is None:
            rs = [grade_fn(t) for _ in range(rollouts)]
        else:
            rs = [grade_fn(t, rollout_seed=_crn_seed(crn_seed, t["op"], t["idx"], k)) for k in range(rollouts)]
        out[(t["op"], t["idx"])] = sum(rs) / len(rs)
    return out


def paired_report(base: dict, trained: dict, B: int = 10000, boot_seed: int = 0, ci_method: str = "percentile") -> dict:
    """Money report: paired per-task deltas -> hierarchical bootstrap CI + across-op sign test + power.

    `boot_seed` controls the bootstrap resampling RNG (default 0 preserves prior behavior). Callers that
    want the CI band to vary with their own seed (e.g. synthetic_powered_report) pass it through here.

    `ci_method` selects the bootstrap interval: "percentile" (default, unchanged) or "bca" (bias-corrected
    & accelerated; auto-falls-back to percentile when n_ops < BCA_MIN_OPS). The method actually used is
    surfaced as the additive `ci_method` field so a caller can tell whether a BCa request fell back.
    """
    keys = sorted(set(base) & set(trained))
    by_op = defaultdict(list)
    for op, idx in keys:
        by_op[op].append(trained[(op, idx)] - base[(op, idx)])
    per_op = {op: sum(d) / len(d) for op, d in by_op.items()}
    all_d = [v for d in by_op.values() for v in d]
    boot = hierarchical_bootstrap_ci(by_op, B=B, seed=boot_seed, method=ci_method)
    sign = across_op_sign_test(per_op)
    n = len(all_d)
    sd = _std(all_d)
    return {
        "n_tasks": n,
        "n_ops": len(by_op),
        "gap_mean": boot["gap"],
        "ci95": (boot["ci_lo"], boot["ci_hi"]),
        "gap_ci_excludes_zero": boot["excludes_zero"],
        "ci_method": boot["ci_method"],
        "per_op_delta": per_op,
        "sign_test": sign,
        "observed_dz": (sum(all_d) / n) / sd if sd > 0 else float("inf"),
        "mde_dz_at_n": mde_paired(n),
        "powered": boot["excludes_zero"] or sign["p_one_sided"] <= 0.05,
    }


# ---------------------------------------------------------------------------
# Bridge to the REAL verifier (protean.grader.grade_source). GPU-only at run time;
# imports are local so this module stays import-light for the CPU test suite.
# A fixed kernel source is deterministic per shape, so rollouts=1 (the policy is the
# source; base vs trained = two fixed sources graded on the SAME held-out shapes).
# ---------------------------------------------------------------------------
def grade_reward(source: str, op: str, shape: int, reps: int = 50, warmup: int = 10) -> tuple:
    """Return (reward, caps) from the real grader. reward=0.0 on cuda_unavailable / any cap."""
    from protean.grader import grade_source

    g = grade_source(source, op=op, split="held_out", shape=shape, reps=reps, warmup=warmup)
    return float(g.get("reward", 0.0)), list(g.get("caps", []))


def evaluate_sources(
    sources_by_op: dict, n_per_op: int = N_HELDOUT_PER_OP, reps: int = 50, warmup: int = 10, seed: int = 0
) -> tuple:
    """Grade one fixed kernel source per op across the powered continuous held-out set.

    sources_by_op: {op_name -> kernel_source}. Returns ({(op,idx)->reward}, tasks, caps_seen).
    """
    ops = list(sources_by_op)
    tasks = build_eval_set(ops, n_per_op, seed)
    rewards, caps_seen = {}, set()
    for t in tasks:
        r, caps = grade_reward(sources_by_op[t["op"]], t["op"], t["shape"], reps, warmup)
        rewards[(t["op"], t["idx"])] = r
        caps_seen.update(caps)
    return rewards, tasks, caps_seen


def paired_sources_report(
    base_sources: dict,
    trained_sources: dict,
    n_per_op: int = N_HELDOUT_PER_OP,
    reps: int = 50,
    warmup: int = 10,
    seed: int = 0,
    B: int = 10000,
) -> dict:
    """End-to-end powered comparison of two fixed kernels-per-op via the real grader.

    Use on a GPU box: e.g. base = seed kernels, trained = optimizer best kernels. With only the 2 real
    ops (elementwise_add_relu, rmsnorm) the across-op sign test is auxiliary (K=2); the per-task
    continuous hierarchical bootstrap over n_per_op shapes carries the power.
    """
    base, tasks, base_caps = evaluate_sources(base_sources, n_per_op, reps, warmup, seed)
    trained, _, trained_caps = evaluate_sources(trained_sources, n_per_op, reps, warmup, seed)
    rep = paired_report(base, trained, B=B)
    rep["synthetic"] = False
    rep["caps_seen"] = sorted(base_caps | trained_caps)
    rep["cuda_unavailable"] = "cuda_unavailable" in rep["caps_seen"]
    rep["sign_test_advisory"] = rep["n_ops"] < N_OPS  # K<5 -> sign test is advisory; CI carries the claim
    # A single-op report cannot support a clustering-immune across-op claim (K=1 sign test p=0.5). The CI
    # may still exclude zero, but we refuse to call a 1-op report "powered_real" because the across-op
    # generalization design needs >=2 ops. powered_real is the field downstream readers should trust.
    if rep["n_ops"] < 2:
        rep["powered_real"] = False
        rep["powered_real_note"] = (
            "only 1 op graded — across-op generalization cannot be supported; "
            "powered reflects the single-op CI only, not the moat claim."
        )
    else:
        rep["powered_real"] = bool(rep["powered"]) and not rep["cuda_unavailable"]
    return rep


# ---------------------------------------------------------------------------
# Optimizer-run bridge: grade SEED-vs-BEST kernels from an optimizer run directory.
# `run_dir` is produced by protean.optimizer.run_optimization (writes best_kernel_<op>.py per op).
# GPU-only at run time (delegates to the real grader); the optimizer can call this for final reporting.
# ---------------------------------------------------------------------------
def powered_eval_from_run_dir(
    run_dir: str | Path,
    ops: list,
    n_per_op: int = N_HELDOUT_PER_OP,
    reps: int = 50,
    warmup: int = 10,
    seed: int = 0,
    B: int = 10000,
) -> dict:
    """Powered base(seed)-vs-trained(best) report using kernels from an optimizer run dir.

    For each op, base = protean.kernels.seed_kernel_for(op); trained = run_dir/best_kernel_<op>.py
    if present, else the seed (so a seed-only dir yields Gap≈0, a valid sanity check). GPU-only.

    `ops` must be a subset of splits.REAL_OPS (the implemented ops). This is validated up-front so a
    fictional op raises a clear ValueError here rather than deep inside the grader.
    """
    from pathlib import Path

    from protean.kernels import seed_kernel_for

    unknown = [op for op in ops if op not in REAL_OPS]
    if unknown:
        raise ValueError(f"unknown op(s) for powered eval: {unknown}; implemented ops are {list(REAL_OPS)}")
    run = Path(run_dir)
    base_sources, trained_sources = {}, {}
    for op in ops:
        base_sources[op] = seed_kernel_for(op)
        best = run / f"best_kernel_{op}.py"
        trained_sources[op] = best.read_text() if best.exists() else seed_kernel_for(op)
    rep = paired_sources_report(
        base_sources, trained_sources, n_per_op=n_per_op, reps=reps, warmup=warmup, seed=seed, B=B
    )
    rep["run_dir"] = str(run)
    rep["ops"] = list(ops)
    return rep


# ---------------------------------------------------------------------------
# CPU-safe SYNTHETIC powered report. Lets the demo artifact (demo/powered-eval-200.json)
# and TECHNICAL_SPEC §10 exist WITHOUT a GPU, while being unmistakably labeled synthetic=True.
# Deterministic given `seed`; uses the SAME paired_report machinery as the real path, so the
# statistical shape (n=200, hierarchical bootstrap CI, across-op sign test) is identical.
# ---------------------------------------------------------------------------
def synthetic_powered_report(
    ops: list | None = None,
    effect: float = 0.22,
    base_level: float = 0.45,
    op_spread: float = 0.10,
    noise: float = 0.20,
    rollouts: int = 8,
    seed: int = 1234,
    B: int = 10000,
) -> dict:
    """Build a fully-shaped, deterministic powered report from a synthetic effect (NO grader / NO GPU).

    Models a plausible base-vs-trained improvement (mean per-task delta ~ `effect` reward units) with
    per-op offsets and within-task noise, then runs it through the real paired_report. The result is
    labeled synthetic=True so it can never be mistaken for measured GPU numbers.

    Defaults to the ops that ACTUALLY EXIST (splits.REAL_OPS = 3 ops). It deliberately does NOT invent
    ops like layernorm/gelu: the GPU regen command (scripts/run_powered_eval.py --run-dir ...) grades the
    same op list through the real grader, which would raise ValueError('unknown op: ...') on a fictional
    op. With K=3 real ops the across-op sign test is auxiliary (best one-sided p=0.125 > 0.05, so it can
    never by itself make powered=True); the per-task continuous hierarchical bootstrap CI over the 200
    held-out tasks carries the power. `sign_test_advisory` is therefore True for the real-op artifact.

    Honest labeling of fields:
      * synthetic=True            -> these are NOT measured numbers.
      * powered_real=False        -> a synthetic report is NEVER a real powered result, regardless of CI.
      * cuda_unavailable=None     -> inapplicable; no CUDA path was exercised (vs. False which would
                                     wrongly imply a GPU was present and used).
      * powered                   -> the honest statistical verdict on the SYNTHETIC data (CI excludes 0
                                     or sign-test p<=0.05); paired with powered_real=False it can never be
                                     misread as a real claim.
    The seed threads through to the bootstrap CI (boot_seed=seed), so different seeds give different CI
    bands; calling twice with the same seed is byte-for-byte reproducible.
    """
    if ops is None:
        ops = list(REAL_OPS)
    rng = random.Random(seed)
    tasks = build_eval_set(ops, seed=0)
    off = {op: rng.gauss(0, op_spread) for op in ops}
    base = evaluate_policy(
        lambda t: max(0.0, base_level + off[t["op"]] + rng.gauss(0, noise)), tasks, rollouts=rollouts
    )
    trained = evaluate_policy(
        lambda t: max(0.0, base_level + effect + off[t["op"]] + rng.gauss(0, noise)), tasks, rollouts=rollouts
    )
    rep = paired_report(base, trained, B=B, boot_seed=seed)
    rep["synthetic"] = True
    rep["powered_real"] = False  # a synthetic report is never a real powered result
    rep["synthetic_params"] = {
        "effect": effect,
        "base_level": base_level,
        "op_spread": op_spread,
        "noise": noise,
        "rollouts": rollouts,
        "seed": seed,
    }
    rep["ops"] = list(ops)
    rep["caps_seen"] = []
    rep["cuda_unavailable"] = None  # inapplicable for a synthetic (no-GPU) report
    rep["sign_test_advisory"] = rep["n_ops"] < N_OPS
    rep["note"] = (
        "SYNTHETIC artifact for plumbing/demo only — NOT measured GPU results "
        "(powered_real=false). Reproduce real numbers with scripts/run_powered_eval.py "
        "--run-dir <optimizer-run> on a GPU box; it grades the SAME real ops via the grader."
    )
    return rep


if __name__ == "__main__":
    rng = random.Random(7)
    ops = [f"op{i}" for i in range(N_OPS)]
    tasks = build_eval_set(ops)
    print(f"eval set: {len(tasks)} held-out tasks / {N_OPS} ops; sample shapes: {[t['shape'] for t in tasks[:6]]}")
    op_off = {op: rng.gauss(0, 0.15) for op in ops}
    EFFECT = 0.18
    base_g = lambda t: max(0.0, 0.30 + op_off[t["op"]] + rng.gauss(0, 0.25))  # noqa: E731
    train_g = lambda t: max(0.0, 0.30 + EFFECT + op_off[t["op"]] + rng.gauss(0, 0.25))  # noqa: E731
    rep = paired_report(evaluate_policy(base_g, tasks), evaluate_policy(train_g, tasks), B=5000)
    ci, st = rep["ci95"], rep["sign_test"]
    print(f"Gap={rep['gap_mean']:.4f}  95%CI=[{ci[0]:.4f},{ci[1]:.4f}]  excl0={rep['gap_ci_excludes_zero']}")
    print(f"sign test: {st['positive']}/{st['ops']} ops, p={st['p_one_sided']:.4f}")
    n, dz, mde = rep["n_tasks"], rep["observed_dz"], rep["mde_dz_at_n"]
    print(f"observed d_z={dz:.3f}  MDE@n={n}={mde:.3f}  POWERED={rep['powered']}")
    print(f"MDE @ old n=3 = {mde_paired(3):.3f}  ->  @ n={rep['n_tasks']} = {mde_paired(rep['n_tasks']):.3f}")
