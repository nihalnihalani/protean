"""Run the statistically-powered held-out evaluation (TECHNICAL_SPEC §5.4 / §10).

Three modes:
  1. --synthetic  : CPU-safe. Builds demo/powered-eval-200.json from a labeled synthetic effect
                    (synthetic=true). No GPU, no grader. Used so the demo artifact exists everywhere.
  2. --run-dir    : GPU. base=seed kernels vs trained=best_kernel_<op>.py from an optimizer run dir.
  3. (default)    : GPU. base=seed vs --trained-dir best kernels (legacy path).

Compares base vs trained across the powered continuous off-grid held-out set and reports
Gap + hierarchical-bootstrap CI + across-op sign test + power. CPU machines (non-synthetic modes)
report cuda_unavailable and reward 0.

Usage:
  # CPU-safe synthetic demo artifact (writes demo/powered-eval-200.json):
  python scripts/run_powered_eval.py --synthetic --out demo/powered-eval-200.json

  # GPU: seed-vs-seed sanity (Gap≈0):
  python scripts/run_powered_eval.py --ops elementwise_add_relu rmsnorm

  # GPU: base seed vs an optimizer run's best kernels:
  python scripts/run_powered_eval.py --run-dir runs/protean-fireworks-overnight \
      --ops elementwise_add_relu rmsnorm --n-per-op 40 --out demo/powered-eval-200.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protean.eval_protocol import (
    paired_sources_report,
    powered_eval_from_run_dir,
    synthetic_powered_report,
)
from protean.kernels import seed_kernel_for
from protean.splits import REAL_OPS


def _trained_source(op: str, trained_dir: str | None) -> str:
    if trained_dir is None:
        return seed_kernel_for(op)  # no trained dir → seed (sanity: Gap should be ~0)
    p = Path(trained_dir) / f"best_kernel_{op}.py"
    return p.read_text() if p.exists() else seed_kernel_for(op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=None,
                    help="ops to grade; default = all implemented ops (splits.REAL_OPS).")
    ap.add_argument("--synthetic", action="store_true",
                    help="CPU-safe: emit a labeled synthetic=true report (no GPU/grader).")
    ap.add_argument("--run-dir", default=None,
                    help="optimizer run dir with best_kernel_<op>.py (GPU base-seed-vs-best).")
    ap.add_argument("--trained-dir", default=None, help="legacy: dir with best_kernel_<op>.py")
    ap.add_argument("--n-per-op", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--effect", type=float, default=0.22, help="synthetic mean per-task delta (reward units)")
    ap.add_argument("--seed", type=int, default=1234, help="synthetic RNG seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ops = args.ops if args.ops is not None else list(REAL_OPS)

    if args.synthetic:
        # None -> synthetic_powered_report defaults to REAL_OPS; an explicit --ops is honored.
        rep = synthetic_powered_report(ops=args.ops, effect=args.effect, seed=args.seed, B=args.bootstrap)
    elif args.run_dir is not None:
        rep = powered_eval_from_run_dir(args.run_dir, ops, n_per_op=args.n_per_op,
                                        reps=args.reps, warmup=args.warmup, B=args.bootstrap)
    else:
        base_sources = {op: seed_kernel_for(op) for op in ops}
        trained_sources = {op: _trained_source(op, args.trained_dir) for op in ops}
        rep = paired_sources_report(base_sources, trained_sources, n_per_op=args.n_per_op,
                                    reps=args.reps, warmup=args.warmup, B=args.bootstrap)

    print(f"ops={rep.get('ops', ops)}  n_tasks={rep['n_tasks']}  n_ops={rep['n_ops']}  "
          f"synthetic={rep.get('synthetic')}  powered_real={rep.get('powered_real')}")
    if rep.get("cuda_unavailable"):
        print("⚠️  cuda_unavailable — run this on the GPU box; numbers below are placeholders.")
    if rep.get("synthetic"):
        print("⚠️  SYNTHETIC — labeled plumbing/demo artifact, NOT measured GPU results.")
    print(f"Gap (mean Δ) = {rep['gap_mean']:.4f}   95% CI = [{rep['ci95'][0]:.4f}, {rep['ci95'][1]:.4f}]"
          f"   excludes 0: {rep['gap_ci_excludes_zero']}")
    st = rep["sign_test"]
    advisory = "  (advisory: K<5 ops)" if rep.get("sign_test_advisory") else ""
    print(f"across-op sign test: {st['positive']}/{st['ops']} ops positive, "
          f"p_one_sided={st['p_one_sided']:.4f}{advisory}")
    print(f"observed d_z={rep['observed_dz']:.3f}  vs  MDE d_z @ n={rep['n_tasks']} = {rep['mde_dz_at_n']:.3f}")
    print(f"POWERED: {rep['powered']}   caps_seen={rep.get('caps_seen', [])}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
