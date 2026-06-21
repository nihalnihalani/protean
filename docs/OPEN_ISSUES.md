# Open Issues

This file tracks current gaps only. Older KERNEL-FORGE adversarial notes live in `docs/strategy/KERNEL_FORGE_AUDIT.md`.

## Blocking For The Next Live Run

### 1. Fireworks overnight run needs fresh results

Status: ready to run.

The Fireworks backend is wired, Spark has the required environment, and optimizer trials can stream to HUD. The remaining gap is a longer overnight run with enough trials to show whether model-generated edits improve over the current best kernels.

Needed:

```bash
export FIREWORKS_API_KEY=...
export HUD_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --stream-hud --hud-job-name protean-fireworks-overnight --out-dir runs/protean-fireworks-overnight
```

Acceptance:

- every candidate appears under `runs/protean-fireworks-overnight/candidates/`
- every trial appears in `trials.jsonl`
- every trial has either `hud_stream.job_url` or `hud_stream_error`
- all streamed trials share one HUD job URL for the overnight session
- model tokens are logged
- compile/runtime failures are rejected with `eval_error`

### 2. HUD generic model still fails to write good kernels in one step

Status: expected behavior, not a verifier failure.

The generic one-step `claude-haiku-4-5` HUD run produced reward `0.0`. The deterministic Protean demo agent now produces speed-sensitive non-zero rewards on all four tasks. This proves the HUD verifier path works, but it also shows that a generic model needs better prompting/tooling before it can solve the tasks.

Next:

- run HUD with a stronger model or a Protean-specific agent
- compare against the deterministic demo agent
- keep the deterministic demo as the verifier proof

## Product Gaps

### 3. Fireworks cost is stored on candidate edits but not fully reported in summaries

The `CandidateEdit` structure has `tokens` and `model_cost_usd`, but Fireworks currently fills tokens and leaves cost at default unless cost calculation is added.

Fix:

- add provider/model pricing config
- compute cost from token usage
- include total cost in `summary_<op>.json`

### 4. Demo benchmark artifact is per-op

`scripts/run_demo_benchmark.py` currently takes one op at a time. The README presents both ops, but the script should optionally generate a combined two-op report.

Fix:

```bash
python scripts/run_demo_benchmark.py --all-ops
```

### 5. Tiny policy head needs a real before/after curve

The 1M learned controller exists, but the README should not claim it improves kernels until a run proves:

- deterministic edit ordering baseline
- learned edit ordering from same trace
- held-out improvement or fewer wasted trials

## Engineering Gaps

### 6. HUD grouped platform eval needs a clean completion run

HUD deployment and taskset sync are complete:

- environment: https://hud.ai/environments/9907b272-ef58-4f57-9cd3-5dbcb37dd51e
- taskset: https://hud.ai/tasksets/6d2feb10-b23c-4928-a1f9-e8b53db364d7
- task ids: `elementwise_add_relu_train`, `elementwise_add_relu_held_out`, `rmsnorm_train`, `rmsnorm_held_out`

The remaining platform check is a grouped remote eval that exits cleanly from the CLI:

```bash
hud eval protean-kernel-optimizer claude --full --group 3 --max-concurrent 4
```

The first attempt loaded all four tasks and started 12 grouped runs, but the CLI produced no progress output for several minutes and was interrupted to avoid leaving an unmanaged long-running process.

### 7. Static anti-hack checks are intentionally minimal

Current checks block obvious passthroughs. They are not a sandbox. Future hardening should add:

- subprocess isolation per candidate
- import allowlist
- stronger AST chain checks for `torch.ops`
- runtime launch instrumentation instead of source-only `@triton.jit` checks

## Resolved Recently

- HUD `Split` type adapter failure fixed by using plain `str` on HUD-facing template args.
- HUD `reward=...` adapter fixed so dashboard shows non-zero reward.
- Optimizer compile/runtime crashes now log as rejected trials.
- HUD deterministic demo agent added and verified.
- Optimizer trial streaming to HUD added and smoke-tested.
- README updated with current job links and Spark numbers.
