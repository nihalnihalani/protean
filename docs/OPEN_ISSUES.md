# Open Issues

This file tracks current gaps only. Older KERNEL-FORGE adversarial notes live in `docs/strategy/KERNEL_FORGE_AUDIT.md`.

## Blocking For The Next Live Run

### 1. Fireworks key missing on Spark

Status: blocked by environment.

The Fireworks backend is wired, but the latest Spark check did not find `FIREWORKS_API_KEY`.

Needed:

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --out-dir runs/protean-fireworks-overnight
```

Acceptance:

- every candidate appears under `runs/protean-fireworks-overnight/candidates/`
- every trial appears in `trials.jsonl`
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

### 6. HUD subscore warning

HUD `SubScore.value` must be `0..1`, while Protean reward can exceed `1.0`. The adapter caps the subscore at `1.0` and preserves Protean reward as the main HUD reward. This works, but HUD may emit a warning because weighted subscores do not sum to the main reward.

Decision:

- acceptable for the demo
- later add separate normalized HUD subscore names, e.g. `correctness`, `speedup_floor`, `metadata_only_reward`

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
- README updated with current job links and Spark numbers.
