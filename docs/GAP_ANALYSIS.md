# Protean — Gap Analysis

> Analysis of `docs/` against the actual code state (not just doc claims). Strategically the docs are now
> coherent (verifier-first optimizer; GRPO/7B correctly demoted to future/stretch). The gaps below are what
> stand between "a verifier that runs" and "a result worth presenting." Prioritized by severity.

## 🔴 Critical — the namesake claim (shape generalization) is not demonstrated

### G1. Dev held-out shapes are ON the tiling grid; the powered off-grid eval is orphaned
- ✅ **(a) FIXED:** `splits.py` dev `HELD_OUT_SHAPES` is now `(1535, 3073, 6143)` — all **off the tiling grid**
  (mod 64 = 63, 1, 63) and prime-adjacent (`3073`, `6143` prime; `1535 = 5·307`). A kernel that hardcodes
  `BLOCK=64/128/256/512` can no longer tile them evenly, so passing them now requires real boundary masking.
  An import-time invariant (`_assert_powered_split`) and a unit test enforce the off-grid property.
- ~~`splits.py` dev `HELD_OUT_SHAPES = (1536, 3072, 5632)` — **all multiples of 512 (mod 64 = 0)**. A kernel with a
  hardcoded `BLOCK=512` tiles them evenly, so passing them does **not** prove shape generalization — the entire thesis.~~
- `sample_heldout_shape` (continuous, mod 64 ≠ 0) + `eval_protocol` (200 paired tasks, hierarchical bootstrap,
  across-op sign test) were built to fix this but are **used nowhere** except `eval_protocol.py` / `run_powered_eval.py`
  (verified by grep). The optimizer, HUD, and demo never call them.
- `TECHNICAL_SPEC` §10 references `demo/powered-eval-200.json` — the file now exists but only as a synthetic CPU
  demo (`synthetic=true`); the real GPU artifact is still to be produced (see G3).
- **Required:** ~~(a) make dev held-out shapes genuinely off-grid~~ (done); (b) wire `eval_protocol` into a runnable
  path and produce the real (`synthetic=false`) `demo/powered-eval-200.json` on GPU; make it the money slide.

### G2. No evidence any model improves a kernel
- The core value prop (model edits beat the seed kernel) is unproven: Fireworks overnight (OPEN_ISSUES #1) and the
  tiny-policy before/after curve (OPEN_ISSUES #5) are both unrun. No base-vs-trained delta exists.
- **Required:** an overnight optimizer run on GPU capturing the improvement curve; until then no doc may imply improvement.

## 🟠 High — doc/reality mismatches & overclaims

### G3. Stale / contradictory numbers — DOCS FIXED
- README and BUILD_CHECKLIST claimed `49 passed, 1 skipped`; actual is **73 passed, 3 skipped**. Corrected in
  README "What Is Working" and BUILD_CHECKLIST expected output.
- `TECHNICAL_SPEC` §10 and README "Verified Artifacts" cited `demo/powered-eval-200.json` as if it were a
  verified GPU result; it now exists only as a synthetic CPU demo (`synthetic=true`). All references (README,
  TECHNICAL_SPEC §10, IMPLEMENTATION_PLAN) now disclose `synthetic=true` and point at `scripts/run_powered_eval.py`
  for the real GPU regeneration.
- **Remaining (GPU):** generate the real (`synthetic=false`) artifact on CUDA.

### G4. Overclaim on the learned controller — DOCS FIXED
- README "What Is Working" said *Learned controller — Implemented … Trains from verifier traces*, but OPEN_ISSUES
  #5 says don't claim improvement until proven. Softened in README and TECHNICAL_SPEC §11 to "implemented; not
  yet shown to beat the deterministic baseline (before/after curve unrun)."

### G5. "Verifier you can trust" is weaker than advertised
- Anti-hack is source-only AST + a `@triton.jit` string check; **no subprocess isolation**, and `torch.ops.aten.*`
  dispatch is not in `BANNED_CALLS`. **Required:** ban `torch.ops`/`aten` chains; add runtime launch instrumentation
  (count real Triton launches, not source grep); subprocess isolation if time allows.

## 🟡 Medium
- **G6.** HUD grouped remote eval never completed cleanly (OPEN_ISSUES #6) — end-to-end platform path unverified.
- **G7.** Fireworks cost not reported (OPEN_ISSUES #3) — tokens logged, `model_cost_usd` default; no pricing config.
- **G8.** Demo benchmark is per-op only (OPEN_ISSUES #4); README presents all ops — needs a combined `--all-ops` artifact.
- **G9.** `softmax_rows` is half-integrated — in catalog/ARCHITECTURE/HUD tasks but absent from the headline Money Figure
  with measured numbers. DOCS FIXED: now explicitly marked "not yet benchmarked" in the README Money Figure,
  FIGURES.md Figure 3, and TECHNICAL_SPEC §10. Remaining (GPU): capture measured numbers.

## 🟢 Low / polish
- **G10.** Across-op sign test is weak at 3 ops (3/3 → p=0.125); lean on the per-task hierarchical bootstrap and state the
  K=3 limitation, or add ops.
- **G11.** Python-version friction: project caps `<3.13` but system Python is 3.14 → needs a 3.12 venv; document it and
  reconcile `requirements-train.txt` vs the `[train]` extra.

## Verdict
Docs are coherent; the killer gap is **G1 + G2**: the project's namesake (shape generalization) is neither demonstrated
(on-grid dev shapes) nor wired into the live path (orphaned powered eval), and no model-improvement delta exists. Closing
G1+G2 is what turns Protean into a presentable result.
