"""Run the statistically-powered held-out evaluation against the REAL grader (GPU box).

Compares base (seed) kernels vs trained (optimizer best) kernels across the powered continuous
held-out set, and reports Gap + hierarchical-bootstrap CI + across-op sign test + power.
See docs/TECHNICAL_SPEC.md §5.4. CPU machines will report cuda_unavailable and reward 0.

Usage:
  # seed-vs-seed sanity (Gap≈0):
  python scripts/run_powered_eval.py --ops elementwise_add_relu rmsnorm

  # base seed vs an optimizer run's best kernels:
  python scripts/run_powered_eval.py --ops elementwise_add_relu rmsnorm \
      --trained-dir runs/protean-fireworks-overnight --n-per-op 40 --out demo/powered-eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protean.eval_protocol import paired_sources_report
from protean.kernels import seed_kernel_for


def _trained_source(op: str, trained_dir: str | None) -> str:
    if trained_dir is None:
        return seed_kernel_for(op)  # no trained dir → seed (sanity: Gap should be ~0)
    p = Path(trained_dir) / f"best_kernel_{op}.py"
    return p.read_text() if p.exists() else seed_kernel_for(op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=["elementwise_add_relu", "rmsnorm"])
    ap.add_argument("--trained-dir", default=None, help="optimizer run dir with best_kernel_<op>.py")
    ap.add_argument("--n-per-op", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_sources = {op: seed_kernel_for(op) for op in args.ops}
    trained_sources = {op: _trained_source(op, args.trained_dir) for op in args.ops}

    rep = paired_sources_report(base_sources, trained_sources, n_per_op=args.n_per_op,
                                reps=args.reps, warmup=args.warmup, B=args.bootstrap)

    print(f"ops={args.ops}  n_tasks={rep['n_tasks']}  n_ops={rep['n_ops']}")
    if rep["cuda_unavailable"]:
        print("⚠️  cuda_unavailable — run this on the GPU box; numbers below are placeholders.")
    print(f"Gap (mean Δ) = {rep['gap_mean']:.4f}   95% CI = [{rep['ci95'][0]:.4f}, {rep['ci95'][1]:.4f}]"
          f"   excludes 0: {rep['gap_ci_excludes_zero']}")
    st = rep["sign_test"]
    print(f"across-op sign test: {st['positive']}/{st['ops']} ops positive, p_one_sided={st['p_one_sided']:.4f}")
    print(f"observed d_z={rep['observed_dz']:.3f}  vs  MDE d_z @ n={rep['n_tasks']} = {rep['mde_dz_at_n']:.3f}")
    print(f"POWERED: {rep['powered']}   caps_seen={rep['caps_seen']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
