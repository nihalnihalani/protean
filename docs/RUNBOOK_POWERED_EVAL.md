# Runbook — Powered Held-Out Evaluation (GPU)

Produces the money-slide artifact `demo/powered-eval-200.json`: a paired base-vs-trained held-out
generalization report over the continuous off-grid held-out tasks (`n_ops × n_per_op`), with a
hierarchical-bootstrap 95% CI and an across-op sign test. See `docs/TECHNICAL_SPEC.md` §5.4 / §10.

The reporting *capacity* is the >=5-op / 200-task target (`N_OPS=5`, `N_HELDOUT_PER_OP=40`); the artifact
filename keeps the historical `-200` suffix. Today only **3 ops are implemented** (`splits.REAL_OPS =
elementwise_add_relu, rmsnorm, softmax_rows`), so a faithful report covers `3 × 40 = 120` tasks. The
synthetic demo and the GPU path both use only these real ops — they never invent layernorm/gelu, because
the GPU regen command grades the same op list through the real grader and would `ValueError('unknown op')`
on a fictional op.

This closes GAP_ANALYSIS **G1** (the powered off-grid eval was orphaned — built in
`eval_protocol.py` but never callable from the optimizer/demo path).

---

## CPU (no GPU): synthetic plumbing artifact

The committed `demo/powered-eval-200.json` is **synthetic** (`"synthetic": true`) so the artifact and
spec references exist everywhere, including CI. It is **not** measured GPU data. Regenerate it (deterministic):

```bash
python scripts/run_powered_eval.py --synthetic --out demo/powered-eval-200.json
```

It runs the *same* `paired_report` statistics (hierarchical bootstrap, sign test) over a labeled
synthetic effect, so the report's *shape* matches the real path exactly. The synthetic artifact carries
`"synthetic": true`, `"powered_real": false`, and `"cuda_unavailable": null` (no GPU path was exercised).
Do not present synthetic numbers as measured results — check `powered_real`, not `powered`.

---

## GPU: real measured numbers (`gpu_only`)

Requires a CUDA device and the project's `triton` extra (`pip install -e '.[triton]'`). The grader
(`protean.grader.grade_source`) returns `cuda_unavailable` on CPU, so this step is GPU-only.

### 1. Sanity (seed vs seed → Gap ≈ 0)

```bash
python scripts/run_powered_eval.py   # defaults to all real ops (splits.REAL_OPS)
```

Expect `Gap ≈ 0`, `excludes 0: False`. This confirms the harness is unbiased before trusting a real delta.

### 2. Run the optimizer to produce best kernels

```bash
python scripts/run_optimizer.py --op elementwise_add_relu --out-dir runs/protean-overnight --max-rounds 50
python scripts/run_optimizer.py --op rmsnorm             --out-dir runs/protean-overnight --max-rounds 50
python scripts/run_optimizer.py --op softmax_rows        --out-dir runs/protean-overnight --max-rounds 50
```

Each run writes `runs/protean-overnight/best_kernel_<op>.py` (the trained kernel per op).

### 3. Base (seed) vs trained (best) powered eval → the artifact

```bash
python scripts/run_powered_eval.py \
    --run-dir runs/protean-overnight \
    --ops elementwise_add_relu rmsnorm softmax_rows \
    --n-per-op 40 --reps 50 --warmup 10 --bootstrap 10000 \
    --out demo/powered-eval-200.json
```

For each op, base = `protean.kernels.seed_kernel_for(op)`; trained = `runs/.../best_kernel_<op>.py`
(falls back to the seed if missing → Gap ≈ 0 for that op). The GPU artifact carries
`"synthetic": false`, `"cuda_unavailable": false`, and `"powered_real"` reflecting the real verdict.
`--ops` must be a subset of the implemented ops (`splits.REAL_OPS`); an unknown op raises a clear
`ValueError` before any GPU work.

### Alternative: from inside an optimizer run

`run_optimization(..., powered_eval=True)` writes `runs/<dir>/powered-eval.json` as a final reporting
step (GPU-only; off by default so the CPU test suite never touches CUDA).

---

## Reading the report

| field | meaning |
|---|---|
| `gap_mean` | grand mean per-task reward delta (trained − base), reward units in [0,2] |
| `ci95` / `gap_ci_excludes_zero` | hierarchical bootstrap 95% CI; CI excluding 0 ⇒ a real effect |
| `sign_test` | across-op sign test on per-op mean deltas (clustering-immune) |
| `sign_test_advisory` | `true` when `n_ops < 5`: the sign test is advisory and the CI carries the claim |
| `powered` | the raw statistical verdict on THIS data: CI excludes 0 **or** sign-test p ≤ 0.05 |
| `powered_real` | the field to trust: `false` for any synthetic report and for single-op runs; `true` only for a real (GPU) multi-op report that is statistically powered and not `cuda_unavailable` |
| `cuda_unavailable` | `true`/`false` for real runs; `null` for synthetic (no GPU path exercised) |
| `observed_dz` vs `mde_dz_at_n` | observed standardized effect vs the minimum detectable effect at the report's `n_tasks` |

**Sign-test caveat (G10):** with K=2 real ops the sign test is powerless (p=0.25 even if both improve);
at K=3, the best case (all positive) is p=0.125 > 0.05, so the sign test alone can never make
`powered=true`. Until ≥5 ops are wired, lean on the per-task hierarchical bootstrap CI and disclose
`sign_test_advisory`.

**Single-op caveat:** a report with `n_ops < 2` (e.g. the optimizer's per-op `powered_eval=True` path,
which passes `[op]`) cannot support the across-op generalization claim. Such a report sets
`powered_real=false` with a `powered_real_note`, regardless of whether its single-op CI excludes zero.

## Timing-fidelity caveats (do not over-claim)

Real GPU numbers are subject to clock drift, GPU contention on shared cloud workers, and L2-cache
residency. The grader uses CUDA-event timing, median-of-reps, and an L2 flush between eager and
candidate; still report the CI, never a bare speedup. Re-run the sanity step (seed vs seed → Gap ≈ 0)
if a measured Gap looks implausibly large.
