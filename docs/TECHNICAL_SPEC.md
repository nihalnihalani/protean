# Protean — Technical Specification

**A reusable HUD RL environment that trains Qwen2.5-Coder-7B (single-turn GRPO) to write Triton GPU kernels that are correct *and* fast on held-out, off-grid tensor shapes.**

> Status: definitive spec. Where the spec deviates from a primary source, the deviation is flagged. Where a number is a derived estimate with no measured benchmark it is tagged `[EST]`. Where a stated value depends on a runtime probe it is tagged `[PROBE]`. Source files are cited by absolute path; equations from papers are cited by arXiv id + equation number, verified against the cloned PDF for daVinci.

---

## 1. Overview & Notation

### 1.1 What Protean is

Protean is a verifier-centric RL environment. The model proposes a Triton kernel for a task `(op, shape)`; a clock-locked, L2-flushed, fresh-input subprocess grades it for **correctness** and **measured wall-clock speedup vs PyTorch eager**; a single canonical reward function maps the grade to a scalar; trl GRPO updates a LoRA adapter on Qwen2.5-Coder-7B. The novelty is **not** the RL recipe (which is standard single-turn GRPO) but the **generalization-graded verifier**: training shapes and evaluation shapes are disjoint *by construction*, so the money metric measures shape generalization, not memorization. No prior kernel-RL work (KernelBench, Dr.Kernel, daVinci, Kevin) evaluates on held-out continuous shapes — they all use fixed discrete shapes.

### 1.2 Pipeline (one rollout)

```
prompt(op, shape)  ──►  πθ (Qwen2.5-Coder-7B + LoRA, vLLM colocate)  ──►  kernel source k
                                                                              │
                          AST anti-hack gate  ◄───────────────────────────────┘
                                                                              │
       subprocess (clock-locked H100): compile → correctness → bench → PR ────┘
                                                                              │
                          compute_reward(...) ∈ [0, 2.0]  ◄───────────────────┘
                                                                              │
                          trl GRPO: group-normalize, policy-gradient step  ◄──┘
```

### 1.3 Notation

| Symbol | Meaning | Source / value |
|---|---|---|
| `op` | fused operator from registry `O` | §9; e.g. `matmul_bias_gelu`, `softmax`, `layernorm`, `elementwise_add_relu` |
| `M` | tensor size parameter (one shape axis) | §9 |
| `S_train` | training shape grid | `{256, 512, 1024, 2048}` (`splits.py`) |
| `S_test` | held-out shape band | `{400, 800, 1600, 383, 769, 3072}` (`splits.py`) |
| `k` | candidate kernel source | model output |
| `C ∈ {0,1}` | binary correctness (allclose on fresh inputs) | `bench_core.py` |
| `T_eager(s)` | median wall-clock of PyTorch eager at shape `s` | `bench_core.py` |
| `T_kernel(k,s)` | median wall-clock of candidate kernel | `bench_core.py` |
| `speedup(k,s)` | `T_eager / max(T_kernel, 10⁻⁶ ms)` | `bench_core.py` |
| `pr_frac` | profiling ratio `T_generated / T_total` | daVinci Eq1; ≈1 at single-op scope |
| `launches_timed` | # Triton JIT dispatches during timed pass | `launch_probe.py` |
| `R = compute_reward(...)` | canonical scalar reward ∈ [0, 2.0] | `rewards.py` |
| `G` | GRPO group size (completions per prompt) | 8 (→4 on OOM) |
| `Âᵢ` | GRPO group-normalized advantage of rollout `i` | DeepSeekMath Eq3 |
| `πθ`, `π_ref` | current / reference (SFT) policy | trl GRPOTrainer |
| `β`, `ε` | KL coefficient, PPO clip radius | β=0, ε=0.2 (trl defaults) |
| `fast_p` | fraction correct AND speedup > p | KernelBench (arXiv:2502.10517) |
| `Gap_p` | `fast_p(πθ, S_test) − fast_p(π_base, S_test)` | money metric (§5) |

### 1.4 Reward / config constants (`reward_config.json`, single authority)

| Constant | Value | Source / rationale |
|---|---|---|
| `CORRECT_FLOOR` | 0.3 | Kevin (arXiv:2507.11948): keeps gradient dense for correct-but-slow kernels |
| `P_TARGET` | 1.5 | speedup normalizer; 1.5× → speedup_score = 1.0 |
| `SPEEDUP_FLOOR` | 1.1 | dead-band; below this speedup_score = 0 (anti dtype/timing-noise) |
| `SPEEDUP_CAP` | **2.25** (corrected, §2.4) | makes the formula honestly bounded; was 20.0 |
| `PR_BONUS` | 0.2 | additive daVinci PR port |
| `TAU` | 0.5 | PR-dominance gate threshold (gate off by default) |
| `PR_HARD_GATE_AFTER_STEP` | 1e9 (off) | flip to a finite step only if a PR-hack is observed |
| `enable_bootstrap` | false | partial-credit warmup (§2.5) |

> **Build note:** `compute_reward` reads `reward_config.json` via `_cfg()`; **this file must exist** or `compute_reward` raises at first call. It is part of the sha256-frozen reward authority (see §2.6).

---

## 2. Reward Function

### 2.1 The true objective vs the proxy

The honest objective the system maximizes per shape is

```
R_true(k, s) = C(k, s) · speedup(k, s),   speedup = T_eager(s) / T_kernel(k, s).
```

The optimizable proxy is `compute_reward` (verified verbatim against `/Users/nihalnihalani/Desktop/Github/protean/src/protean/rewards.py`):

```
                ⎧ 0,                                   if ¬(correct ∧ dtype_ok ∧ shape_ok)
R_proxy(k,s) =  ⎨ 0,                                   if launches_timed ≤ 0
                ⎩ clip( CORRECT_FLOOR + speedup_score + pr_term , 0 , 2.0 ),   otherwise
```

with

```
speedup_score = ⎧ 0,                              if speedup < SPEEDUP_FLOOR (=1.1)
                ⎨
                ⎩ min(speedup, SPEEDUP_CAP) / P_TARGET,   otherwise

pr_term       = PR_BONUS · clip(pr_frac, 0, 1).
```

The two hard zeros are the **robust gates** (correctness + dtype/shape match, and a real Triton launch). They fire *before* any speedup or PR credit, so no incorrect or non-launching kernel can earn positive reward.

### 2.2 Relationship to daVinci Eq1 — honest delta (NOT a port)

daVinci-kernel (arXiv:2606.16497, **Eq1, verified PDF p.3**) is **multiplicative**:

```
R_{i,t} = C(y_{i,t}) · (1 + speedup_{i,t} + PR_{i,t}),   PR_{i,t} = T_generated / T_total.
```

Protean's reward is **algebraically distinct** — a *hybrid* of Kevin (additive correctness floor) and daVinci (PR as a reward component), **not** a port of Eq1. The departures:

1. The binary multiplicative coefficient `C(y)` becomes a **hard pre-zero gate** *plus* a **Kevin additive `CORRECT_FLOOR = 0.3`** (Kevin arXiv:2507.11948 §3.2: `S = 0.3·1{correct} + (T_base/T_kernel)·1{correct}`).
2. Speedup is **P_TARGET-normalized** (`/1.5`); Eq1 has no normalizer.
3. PR enters as a **fixed 0.2-scaled additive bonus**, *not* a unit-commensurate multiplicative term.

> The earlier docstring phrase "matches Eq1 continuity" has no algebraic meaning and is **removed**. The honest claim: *inspired-by Eq1, not a port-of.*

**Why additive, not multiplicative (formal justification).** At Protean's single-op subprocess scope, the subprocess contains exactly one op, so `T_generated ≈ T_total ⇒ pr_frac ≈ 1` for every correct rollout. A multiplicative `(1 + speedup + PR)` would merely rescale the speedup term by a near-constant ≈2, contributing **zero contrast** under GRPO group-normalization (a constant factor shared by all rollouts cancels in `(rᵢ − mean)/std`). The additive form isolates the PR signal so its *only* live role is the anti-passthrough offset. See §2.3 for the gradient argument and §11-Q4 for the consequence.

### 2.3 The PR term contributes ≈ zero GRPO gradient (stated, not hidden)

Because `pr_term ≈ 0.2` is a **group-constant** on the correct subset at single-op scope, under GRPO advantage

```
Âᵢ = (rᵢ − mean_G(r)) / std_G(r)
```

adding the same `0.2` to every correct `rᵢ` shifts `mean_G(r)` by `0.2·(#correct/G)` and leaves the *contrast between correct rollouts unchanged*; ∂L/∂(pr_term) ≈ 0. Moreover any rollout that passes the LaunchCounter gate has `launches_timed > 0 ⇒ T_generated > 0 ⇒ pr_frac > 0`, so the "pr=0 forfeits bonus" branch is **unreachable** for any rollout that survives the harder launch gate. **Net:** the PR term is decorative at single-op scope; its anti-passthrough function is fully subsumed by the LaunchCounter.

**Resolution (chosen for the 24 h build):** keep the term for narrative continuity with daVinci, but state in this section (not buried in a gap list) that it provides ~0 training signal. The genuinely load-bearing daVinci-derived mechanism Protean ships is **execution-verified launch counting + fresh-input reseed**, not the PR reward term. (Alternative, deferred: measure `pr_frac` in a multi-op harness with competing GPU work so `pr_frac < 1` becomes a real discriminator — see §11-Q4.)

### 2.4 Boundedness and the saturation point (corrected)

The pre-clip sum is `CORRECT_FLOOR + speedup_score + pr_term`. With the **original** `SPEEDUP_CAP = 20.0`, `speedup_score` reached `20/1.5 = 13.33`, so the `[0, 2.0]` bound was enforced *only* by the post-hoc `clip` — and every kernel above the saturation speedup mapped to exactly 2.0, producing a **gradient-flat dead zone**.

Derive the saturation speedup `S_sat` (clip binds when the sum ≥ 2.0):

- Max PR (`pr_term = 0.2`): `speedup_score ≥ 1.5 ⇒ speedup/1.5 ≥ 1.5 ⇒ S_sat = 2.25×`.
- Zero PR (`pr_term = 0`): `speedup_score ≥ 1.7 ⇒ S_sat = 2.55×`.

**Correction adopted:** set `SPEEDUP_CAP = P_TARGET·(2.0 − CORRECT_FLOOR − PR_BONUS) = 1.5·1.5 = 2.25`. Now the formula is **honestly bounded by construction**, the `20.0` backstop (which only ever guarded a path the clip already covered) is gone, and the flat region is documented as deliberate. Spec sentence to surface on the slide:

> *All correct kernels faster than `S_sat = 2.25×` (max-PR) saturate the reward at 2.0; above `S_sat` GRPO sees zero contrast. A 7B overnight is not expected to exceed ~2× on held-out shapes, so this is acceptable — but it is stated, not hidden behind the clip.*

(If unbounded reward is later wanted, replace `speedup_score` with the scale-invariant `log(speedup)/log(P_TARGET)` — 1.0 at 1.5×, 2.71 at 3×, no saturation — and raise/drop the clip accordingly. Deferred; the cap fix preserves the existing unit tests.)

### 2.5 Bootstrap partial-credit (warmup, integrated & corrected)

If the SFT warm-start leaves the allclose rate so low that most groups are all-zero (gradient collapse, §3.5), a config-gated bootstrap grants annealing partial credit to **correct-and-compiling** kernels only:

```
bootstrap(step) = bonus · max(0, (anneal_steps − step) / anneal_steps),   bonus = 0.1·1{compiles} + 0.2·1{imports_triton}
```

Three corrections vs the stub `bootstrap_credit`:

1. **Integrated** inside `compute_reward` as a named branch gated on `enable_bootstrap` (config) so it falls under the sha256 freeze; it is added **only after** the correctness gate passes (`credit = bootstrap if correct else 0`). It can never resurrect a hard-zeroed kernel — this prevents the step-0 inversion where an incorrect-but-compiling kernel would otherwise tie a correct-but-slow one at 0.3.
2. The anneal is **true-linear to 0 at the boundary** (the stub jumped `bonus/40 → 0` at `step = 40`); use `(anneal_steps − step)/anneal_steps` which is 0 at `step = anneal_steps`.
3. **Group-std interaction documented:** during `step < anneal_steps` the bootstrap manufactures intra-group variance precisely when correctness is sparse; in this window advantages encode "compiles" rather than "fast." This is intentional and anneals out by `anneal_steps = 40`. `enable_bootstrap = false` by default.

### 2.6 One reward authority (Law #1)

`rewards.py` is the **single** reward authority, baked to `/donotaccess/rewards.py`, imported byte-identically by the offline grader `grade.py` and by the trainer via `grade_kernel`. A sha256 of the file is logged at kick. Any prose pseudocode elsewhere (e.g. earlier multiplicative or `[1,3]`-range drafts in `IMPLEMENTATION_components.md`) is **SUPERSEDED**; only the formula in §2.1 runs. The sha256 guard protects the code, not the docs — so this spec restates the canonical formula verbatim to keep the prose consistent with the executed reward.

### 2.7 Worked numeric examples

Config: `CORRECT_FLOOR=0.3, P_TARGET=1.5, SPEEDUP_FLOOR=1.1, SPEEDUP_CAP=2.25, PR_BONUS=0.2`.

| # | correct | speedup | pr_frac | launches | speedup_score | pr_term | pre-clip | **R** |
|---|---|---|---|---|---|---|---|---|
| 1 | False | 3.0 | 0.9 | 1 | — | — | — | **0.00** (hard zero) |
| 2 | True | 1.05 | 0.5 | 1 | 0 (below floor) | 0.10 | 0.40 | **0.40** |
| 3 | True | 1.00 | 0.6 | 1 | 0 (below floor) | 0.12 | 0.42 | **0.42** |
| 4 | True | 1.50 | 0.6 | 1 | 1.50/1.5 = 1.00 | 0.12 | 1.42 | **1.42** |
| 5 | True | 2.25 | 0.0 | 1 | 2.25/1.5 = 1.50 | 0.00 | 1.80 | **1.80** |
| 6 | True | 3.0 | 0.6 | 1 | min(3,2.25)/1.5 = 1.50 | 0.12 | 1.92 | **1.92** |
| 7 | True | 10.0 | 0.6 | 1 | min(10,2.25)/1.5 = 1.50 | 0.12 | 1.92 | **1.92** |
| 8 | True | 2.0 | 1.0 | 0 | — | — | — | **0.00** (no_triton_launch) |

Rows 6,7 are identical (`R = 1.92`): the cap makes 3× and 10× indistinguishable — the documented flat region. Unit tests `test_rewards.py` must assert rows 2, 4, 5, 6, 7.

---

## 3. Training: Single-Turn GRPO + SFT/LoRA Warm-Start

### 3.1 Why GRPO (not PPO)

GRPO (Shao et al., DeepSeekMath, arXiv:2402.03300) replaces PPO's learned value network with a **group-relative baseline**: sample `G` completions per prompt, score each, standardize within the group. This removes the critic (≈14 GB of weights+AdamW state on a 7B — another model's worth) from an already-tight 80 GB H100 budget. The baseline is free: it is computed from rewards the verifier already returns.

### 3.2 The objective (trl default = token-level / DAPO normalization)

trl GRPOTrainer (verified docs, 2026) defaults to `loss_type='dapo'` (token-level normalization), `β = 0.0` (no KL), `scale_rewards='group'`. The shipped loss:

```
L_GRPO(θ) = − (1 / Σᵢ|oᵢ|) Σᵢ Σ_t [ ρ_{i,t} · Âᵢ − β · D_KL(πθ ‖ π_ref) ]

ρ_{i,t} = πθ(o_{i,t}|q,o_{i,<t}) / [πθ(o_{i,t}|q,o_{i,<t})]_{stop-grad}
```

At `num_iterations = μ = 1` (trl default, Protean's setting) the rollout policy equals the policy being updated, so `ρ_{i,t} = 1` for every token and the PPO clip **never binds** — the objective reduces to token-normalized REINFORCE scaled by the advantage. We set `epsilon = 0.2` explicitly (hygiene; harmless at μ=1).

The original DeepSeekMath sequence-level form (arXiv:2402.03300, Eq3) for reference:

```
J_GRPO(θ) = E[ (1/G) Σᵢ (1/|oᵢ|) Σ_t { min(ρ_{i,t}Âᵢ, clip(ρ_{i,t},1−ε,1+ε)Âᵢ) − β D_KL } ].
```

### 3.3 Advantage — two explicit, orthogonal axes

**(i) trl GRPO (shipped):**

```
Âᵢ = (rᵢ − mean_G(r)) / (std_G(r) + ε_std),     ε_std ≈ 1e-8 (trl).
```

The group **mean baseline includes sample i**, giving a self-inclusion weight `1/G`. At `G = 8` this is a **12.5 %** gradient shrinkage vs an unbiased baseline (Dr.Kernel arXiv:2602.05885 App. A: `E[ĝ_GRPO] = (1 − 1/N)·∇J`). Constant across steps → affects convergence speed, not direction.

**(ii) daVinci TRLOO (deferred, arXiv:2606.16497 Eq2, verified PDF p.3):**

```
A_{i,t} = G_{i,t} − (1/(N_t−1)) Σ_{j≠i} G_{j,t}.
```

The LOO baseline **excludes i** (zero self-inclusion); its residual finite-sample factor scales as `1/(N−1)`.

> **Orthogonality (corrected).** std-normalization is *independent* of the LOO-vs-mean baseline choice. trl GRPO = mean baseline + std-norm. Dr.GRPO (arXiv:2503.20783) = mean baseline + **no** std-norm. TRLOO = LOO baseline (std-norm optional). The spec does **not** equate "LOO" with "no std." Protean ships trl GRPO (mean + std); `scale_rewards='group'` is set explicitly (not left to version default).

### 3.4 Single-turn reduction

Protean is single-turn (`T = 1`). The discounted return `G_{i,t} = Σ_{t'≥t} γ^{t'−t} R_{i,t'}` collapses to `G_{i,1} = R_i`, so **γ is irrelevant** and there is no per-turn credit assignment. Kevin's `γ = 0.4` finding only applies to the 2-turn stretch (§11-Q7).

### 3.5 Gradient-collapse failure mode and guards

If all `G` completions of a prompt score 0 (likely cold-start on Triton), `std_G(r) ≈ 0`, the advantage is `0/0 → NaN`, and AdamW is irreversibly poisoned. Guards:

1. **Pre-kick calibration gate (Stage A):** `std(R) > 0.05` over a probe batch, with ≥1 group having ≥2 distinct reward values. Threshold rationale: with `R ∈ [0,2]` and `G=8`, requiring std > 0.05 ≈ requiring one rollout pair to differ by ~0.1 in raw reward — a **minimum-detectable-contrast heuristic**, necessary (nonzero gradient) but *not* a convergence guarantee; not derived from a power analysis.
2. **Per-step NaN/std detector (mid-run, REQUIRED):** in the trainer loop, if `std(group_rewards) < 0.01` for 3 consecutive steps, checkpoint and **abort** — never optimize on degenerate advantages. trl's `ε_std ≈ 1e-8` does *not* protect against near-degenerate groups, so this explicit detector is mandatory, not optional. Also `assert isfinite(rewards/mean)` each step.
3. **Bootstrap (§2.5)** and the L1-elementwise curriculum fallback manufacture early variance if the SFT rate is too low.

### 3.6 Temperature → reward-contrast guarantee

GRPO's gradient is nonzero iff a group has ≥2 distinct rewards. Sampling temperature `τ_gen` (we use 1.0, the daVinci value) is the mechanism that guarantees textual diversity across the `G` completions. If the post-SFT per-prompt success probability is `a ∈ (0,1)`, the probability a group of `G` contains both a correct and an incorrect rollout (mixed reward) is

```
P(mixed) = 1 − a^G − (1 − a)^G.
```

At `a = 0.3, G = 8`: `P(mixed) = 1 − 0.3⁸ − 0.7⁸ = 1 − 6.6e-5 − 0.0576 ≈ 0.942`. So ~94 % of groups carry signal — *provided* `a` is lifted off 0 by SFT. (Caveat: this bounds reward-contrast via correctness diversity; if SFT makes early boilerplate tokens near-deterministic, branchable positions concentrate in the kernel body, shrinking the effective `L` — the `L ≤ 1024` token budget still holds.)

### 3.7 Expected advantage magnitudes (worked)

Post-SFT, suppose 30 % of `G=8` correct, all at ~1.0× (below floor) so correct → `R = 0.3 + 0 + 0.2·pr ≈ 0.32`, incorrect → 0:

```
mean(R) ≈ 0.30·0.32 = 0.096
std(R)  ≈ sqrt(0.30·(0.32−0.096)² + 0.70·(0−0.096)²) ≈ 0.146
Â(correct)   ≈ (0.32 − 0.096)/0.146 ≈ +1.53
Â(incorrect) ≈ (0    − 0.096)/0.146 ≈ −0.66
```

Tractable magnitudes for a 7B LoRA at `lr = 1e-5`. If `std(R) < 0.05`, advantages collapse → the Stage A gate and the §3.5 detector fire.

### 3.8 SFT / LoRA warm-start

- **Backbone:** Qwen2.5-Coder-7B (28 layers, hidden 3584, 28 attn heads, 4 KV heads/GQA, intermediate 18944).
- **Full FT infeasible:** AdamW mixed-precision = `16·P = 16·7.62e9 = 122 GB > 80 GB`. LoRA is the enabler.
- **LoRA r=16, `target_modules='all-linear'`:** trainable params ≈ **40.5M** `[EST]` (per-layer `r·(d_in+d_out)` summed over q/k/v/o/gate/up/down × 28 layers; k_proj/v_proj use `d_out = 4·128 = 512` under GQA). Optimizer state ≈ `12·40.5M ≈ 0.49 GB` (negligible). **Verify at runtime** with `peft_model.print_trainable_parameters()` `[PROBE]` — the architectural estimate excludes possible bias/norm inclusion.
- **trl recommends `lr = 1e-5` for LoRA** (10× the full-FT 1e-6); Protean uses `1e-5`.
- **SFT source:** `hkust-nlp/drkernel-coldstart-8k`, filtered to rows with `final_speedup ≥ 1.2` (~200 rows `[EST]`), skill-conditioned so the static skill prefix (§7) is load-bearing. **Gated on a Friday schema probe** `[PROBE]`: if the `final_speedup` column is absent/NaN-heavy, the SFT pass is **skipped** and the skill prefix is **disabled** (Ablation-1 benefit, §7, does not transfer without skill-conditioned SFT).

---

## 4. Measurement Protocol

Implemented in `/Users/nihalnihalani/Desktop/Github/protean/src/protean/bench_core.py`.

### 4.1 Measured quantities

For task `(op, shape s, dtype)` with reference `f_ref` and candidate `f_θ`:

- `T_ref(s)` = median wall-clock of `f_ref` over `n_rep` iters.
- `T_kernel(s)` = median wall-clock of `f_θ` over `n_rep` iters.
- `speedup(s) = T_ref(s) / max(T_kernel(s), 10⁻⁶ ms)` (denominator clamp avoids div-by-zero).

### 4.2 Timing methodology (CUDA events, L2 flush, locked clocks)

Per iteration (verified `bench_core.py` lines 23–31), with corrections:

```
flush.zero_()                      # 64 MB int32 L2 flush buffer (> H100 50 MB L2; do_bench parity)
torch.cuda.synchronize()           # CORRECTION: flush MUST complete before the timer starts
s.record(); kernel_fn(*inp); e.record()
torch.cuda.synchronize()
times.append(s.elapsed_time(e))    # CUDA-event ms, ~0.5 µs resolution
```

- **Flush is OUTSIDE the timed window** (the stub places `flush.zero_()` before `s.record()` — correct; the added `synchronize()` removes the residual race where the flush kernel could still be executing when `s.record()` fires).
- **Buffer:** 64 MB (16M int32) > H100 SXM5 L2 (50 MB) → full eviction. Bump to 80 MB for associativity margin (optional).
- **Clocks (`@modal.enter`):** `nvidia-smi -lgc 1980` pins the SM clock to 1980 MHz. **Correction (verify, do not assume):** after the call, run `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader` and assert within 5 % of 1980; on failure set `CLOCK_UNLOCKED=1`, raise `reps` 100→200, and annotate the reward dict (`timing_confidence='reduced'`). Modal containers may deny `-lgc`; `check=False` must not silently hide it.

### 4.3 Iteration counts

Fixed `warmup = 25`, `reps = 100` (flat counts, not do_bench time budgets). **Gap:** for a genuinely cold-JIT novel held-out shape (first compile 30–130 s), warmup may exhaust the 60–90 s subprocess timeout, collapsing `n_rep`. Add a **minimum-`n_rep` safety check**: if fewer than 20 timed iters complete, return `timing_unreliable=True` and reward 0 (do not trust a 1-sample median).

### 4.4 Median vs mean (deviation from KernelBench, disclosed)

`bench_core.py` uses the **median**; KernelBench `fast_p` uses the **mean** (via `do_bench` default). GPU timing is right-skewed (JIT/OS/thermal spikes pull the mean up), so median is more robust but **not directly comparable** to published KernelBench numbers.

**Correction (median bug):** the stub `times[len(times)//2]` returns the *upper* median for even `n` (element [50] of 100 = 51st value), systematically over-estimating `T_kernel` and thus **deflating** speedup. Fix: `statistics.median(times)` (handles even/odd). Add a unit test asserting equality with `statistics.median` on a fixed array. Footnote on the money slide: *"Protean reports median latency; KernelBench reports mean — speedups are not directly comparable to KernelBench baselines."*

### 4.5 Determinism: the 3-seed protocol

- `SEED_CORRECT = 42` — pre-timing correctness (allclose).
- `SEED_TIMED_BASE = 43` — fresh inputs per timed iter (seed `43 + i`), defeats result/scratchpad caching: a kernel that caches iter-0's output fails allclose on iter `i ≥ 1` with probability 1 (distinct random tensors).
- **`SEED_POST` (corrected, must be unpredictable/non-overlapping):** the stub's fixed `44` is gameable by a policy that special-cases seed 44. Replace with a non-overlapping deterministic seed `SEED_POST = SEED_TIMED_BASE + warmup + reps` (never reachable by the timed sweep) **or** an unpredictable per-rollout hash `int(sha256(kernel_src_sha256 + str(shape) + str(step))[:8], 16)`. The post-timing allclose on this seed (inputs materialized *after* the timing loop) closes the precompute exploit structurally rather than relying on "a 7B can't do it."

### 4.6 Correctness

```
correct = allclose(f_θ(x), f_ref(x), rtol=1e-2, atol=1e-2)   on SEED_CORRECT and SEED_POST
dtype_ok = (out_k.dtype == out_r.dtype)        # OUTPUT dtype only
shape_ok = (tuple(out_k.shape) == tuple(out_r.shape))
```

`rtol = atol = 1e-2` matches KernelBench `get_tolerance_for_precision('fp16')`. **Soundness gap:** `dtype_ok` checks only the *output* dtype; a kernel computing internally in bf16 and upcasting passes — the rtol/atol allclose is the only (statistical, not formal) defense (§6, T8).

### 4.7 Speedup confidence interval (matched-pair bootstrap)

The per-kernel error bar uses a **matched-pair bootstrap percentile CI**, *not* the delta method (whose normal approximation undercovers for right-skewed sub-millisecond kernels):

```
for b in 1..B (=1000):
    resample paired ratios { T_ref[i] / T_kernel[i] } with replacement
    s_b = median of the resampled ratios
CI_95 = (percentile(s, 2.5), percentile(s, 97.5)).
```

**Required change:** `bench_kernel` must additionally return the raw `times_kernel` array (currently discarded as a local), and `_cached_eager_time` must return its raw array, so the paired resample is computable. The delta-method formula `Var(S) ≈ S²(CV_ref² + CV_kernel²)` is **retracted** as the reported CI.

### 4.8 Eager-baseline caching

`_cached_eager_time(op_ref, make_inputs, shape, dtype)` is cached per `(op, shape)` to avoid re-timing every rollout. **Soundness requirement:** the cached `T_ref` must be measured with the **identical** L2-flush + warmup + median + device-resident-input protocol, else the ratio compares cold-kernel vs warm-eager. Document the cache's internal protocol in this section.

### 4.9 Input residency (avoid timing H2D)

`make_inputs` runs on CPU before `s.record()` in the stub, so allocation is outside the window — but if it returns CPU tensors, an implicit H2D copy inside `kernel_fn` (≈0.067 ms for a 1024² fp16 tensor over PCIe) lands inside the timed region and contaminates fast kernels. **Fix:** pre-allocate a device-resident input pool *before* the loop, `assert all(t.device.type=='cuda')`, index `inp_pool[i]` inside the loop; ensure `_cached_eager_time` uses the same pool so any residual H2D cancels.

---

## 5. Generalization & Statistics

### 5.1 Shape space and the disjoint partition (the moat)

Verbatim from `splits.py`:

```
S_train = {256, 512, 1024, 2048}             # power-of-two grid
S_test  = {400, 800, 1600, 383, 769, 3072}   # off-grid (×100), prime-adjacent, tiling-boundary
S_all   = S_train ∪ S_test ,   S_train ∩ S_test = ∅.
```

`_assert_split_disjoint()` runs **at import, in CI, and inside `freeze()`** and asserts (a) set disjointness and (b) no test shape equals a train power-of-two. This is **Law #1** — the moat invariant.

**Why these test shapes are structurally different, not "just bigger":**
- `3072 = 3·1024` **crosses a tiling boundary**: `ceil(3072/128) = 24` blocks vs `ceil(2048/128) = 16`. A kernel hardcoded for 16 outer iterations produces wrong/incorrect output at 24 → `C = 0`.
- `383, 769` are **prime-adjacent** (`384−1`, `768+1`): no power-of-two block size divides them, so a hardcoded `BLOCK=256` with wrong masking yields numerical error.
- `400, 800, 1600` are round-but-off-grid (multiples of 100), rare in GPU code.

### 5.2 Covariate shift framing

The shift is **covariate**: `P_train(M) ≠ P_test(M)` but the reward labeling function `P(R | k, M)` is the *same physical timing formula* at every `M`. So held-out evaluation measures true shape generalization, not a different task. The *optimal* kernel changes with `M` (different tiling/autotune configs), so a memorized train-shape solution is *expected* to degrade off-grid — which is what makes the gap meaningful.

> **Honest scoping (Ben-David removed).** The earlier Ben-David et al. 2010 citation is dropped: that supervised-domain-adaptation PAC bound assumes a VC-bounded hypothesis class over labeled input-output pairs, not a token-level LLM policy under a shift over tensor shapes, and we never compute the H-divergence. The defensible statement: **disjoint train/test shape bands are a *necessary* condition for a generalization (not memorization) claim; they are *not sufficient*, and we present results as directional evidence under covariate shift.** No sufficiency claim is made.

### 5.3 The generalization gap (money metric)

```
Gap_p = fast_p(πθ, S_test) − fast_p(π_base, S_test),
fast_p(π, S) = (1/|S|) Σ_{s∈S} 1[ correct(π, s) AND speedup(π, s) > p ].
```

Report `fast_1` (comparable to all published baselines) and `fast_1.2` (Dr.Kernel/daVinci threshold). `fast_p` denominates over **all** tasks, not just correct ones (KernelBench `score.py`).

### 5.4 Sample size, MDD, and the clustering correction

**Naive (independent) two-proportion MDD** at α=0.05 (z=1.96), power=0.80 (z=0.842):

```
MDD = (z_{α/2} + z_β)·sqrt( 2·p·(1−p) / n ) = 2.802·sqrt( 2p(1−p)/n ).
```

At `n = 24` (4 ops × 6 test shapes):

| baseline p | MDD (naive) |
|---|---|
| 0.10 | **0.243** |
| 0.20 | **0.323** |
| 0.50 | **0.404** |

Headline (conservative, p≈0.2): **MDD ≈ 32 pp** naive. The Wilson 95 % CI half-width at p=0.2, n=24 is ≈ **0.16** — a *CI width*, **not** an MDD; the two must not be conflated.

**Clustering correction (the binding number).** The 6 shapes per op are graded by the *same checkpoint on the same op* → positively correlated; treating them as 24 independent Bernoulli trials overstates power. Apply the design effect with `m = 6` shapes/op and intra-op `ρ`:

```
n_eff = n_total / (1 + (m−1)·ρ̂).
```

At `ρ = 0.5`: per-op `n_eff = 6/(1+5·0.5) = 1.71`, total `n_eff = 4·1.71 ≈ 6.9`, so

```
MDD = 2.802·sqrt( 2·0.2·0.8 / 6.9 ) ≈ 2.802·0.215 ≈ 0.60  (≈ 47 pp).
```

**Headline (adopt this):** with clustered `n_eff ≈ 7`, **MDD ≈ 47 pp at 80 % power** assuming `ρ = 0.5`; estimate `ρ̂` from the pilot rather than assuming. Do **not** report the naive n=24 number as if independent.

### 5.5 Reporting and the joint (steps × gap) honesty statement

- Error bars: **Wilson interval** for the binary `fast_p` rate; **matched-pair bootstrap** (§4.7) for the speedup panel. Pin both in SLIDES.
- A plausible 7B-overnight true gap is 10–20 pp, **below** MDD ≈ 47 pp at `n_eff ≈ 7` (and below 32 pp even naively). **Significance is not attainable at this scale regardless of step count.** Frame the money slide explicitly:

> *The held-out gap is **directional, single-run** evidence. With `n_eff ≈ 7` (24 clustered tasks), detecting a 10–20 pp gap at 80 % power needs ~70+ effective tasks or a >32–47 pp true gap. We are underpowered **by design** — a hackathon constraint, not a significance claim.*

- **Recommended expansion:** add 6 off-grid shapes per op (`{192, 576, 960, 1344, 1729, 2561}`, all disjoint, asserted) → `n_test = 48`, `n_eff ≈ 14` at ρ=0.5 → MDD ≈ 33 pp. Costs ~$1 of BLOCK-6 inference, zero impact on the overnight run.

### 5.6 Worked CI example

Trained checkpoint passes `fast_1` on 9 of 24 test tasks → `p̂ = 0.375`. Wilson 95 %:

```
center = (p̂ + z²/2n)/(1 + z²/n) = (0.375 + 1.92/48)/(1 + 3.84/24) = 0.4152/1.16 = 0.358
half   = z·sqrt(p̂(1−p̂)/n + z²/4n²)/(1+z²/n)
       = 1.96·sqrt(0.2344/24 + 3.84/2304)/1.16 = 1.96·0.1027/1.16 = 0.174
CI_95 ≈ [0.184, 0.532].
```

A base rate of 0.20 lies inside this interval → a `0.375 − 0.20 = 17.5 pp` gap is **not** significant at n=24 — exactly the underpowering §5.5 warns about.

---

## 6. Anti-Hack Soundness

Threat model: does a policy `π*` exist that earns high `R_proxy` while `R_true` is low? For each exploit we bound `R_proxy(π*) ≤ θ`, with the conservative threshold `θ = CORRECT_FLOOR = 0.3` (the reward of a correct, 1.0×, no-credit kernel). Implemented across four layers: AST ban (`anti_hack.py`), LaunchCounter (`launch_probe.py`), dtype/shape match (`bench_core.py`), speedup floor+cap (`rewards.py`); plus fresh-input reseed and shape-split disjointness.

### 6.1 Formal frame (Skalse et al. 2022)

A proxy is *unhackable* w.r.t. true reward iff proxy-ordering of policies matches true-ordering (arXiv:2209.13085). Their impossibility theorem: over **all** stochastic policies, two reward functions are unhackable only if one is constant. Non-trivial unhackability therefore requires **restricting the policy class** — which is exactly what each guard does (it maps an exploit subclass to `R ≤ θ`). Soundness here is **per-enumerated-exploit**, not a global guarantee.

### 6.2 Exploit-by-exploit bounds

**E1 — Torch/ATen passthrough.** Calls `torch.matmul`/`F.softmax`/… or raw `torch.ops.aten.mm.default`.
- Guard: Layer-1 AST ban fires **before execution** → `R = 0`.
- **Corrections to `anti_hack.py`:** (a) implement `_dotted()` to walk arbitrary-depth `Attribute`/`Name` chains; (b) **prefix-match** ban `torch.ops`, `torch.ops.aten`, and substring-ban `ops.aten`, `cublas`, `cublasLt`, `cudnn`, `_inductor` (current `BANNED_CALLS` is exact-string only → `torch.ops.aten.mm.default` leaks, Test T15); (c) resolve import aliases for `torch.nn.functional`. Completeness assumption: `Π_pass ⊆ {AST contains a banned node OR banned-prefix chain OR banned substring}`.

**E2 — Never-launched kernel (dead JIT).** Writes `@triton.jit` but runs the op in PyTorch.
- Guard: `LaunchCounter` patches `JITFunction.__call__` (stable Python boundary); `launches_timed ≤ 0 ⇒ R = 0`. The patch intercepts cache hits too (verified by the in-image smoke test). Fallback: count Triton kernel names in a `torch.profiler` trace if a Triton version bypasses `__call__`.

**E3 — dtype downcast.** Returns a narrower output dtype for tensor-core speed.
- Guard: `dtype_ok = (out_k.dtype == out_r.dtype) ⇒ R = 0` if mismatched. **Residual (T8):** internal bf16 with upcast output passes; only rtol/atol allclose defends — statistical, not formal.

**E4 — Fixed-shape overfit.** Hardcodes `M ∈ {256,512,1024,2048}`.
- Guard: held-out eval uses `S_test` only. A 16-iter-hardcoded kernel produces wrong output at `M=3072` (24 blocks) → `C=0 ⇒ R=0`; bad masking at `383/769` → `C=0`. **Caveat:** this proves `R=0` only for kernels that hardcode iteration/masking; a genuinely shape-generic kernel correctly generalizes and is *not* penalized — the verifier cannot distinguish "generalized" from "memorized a continuous family" from the gap alone (→ directional evidence, §5.2).

**E5 — Result/scratchpad caching (timer gaming).** Returns iter-0's output on later iters.
- Guard: fresh inputs per timed iter (seed `43+i`) → allclose fails on iter ≥1; post-timing allclose on the unpredictable `SEED_POST` (§4.5) → `C=0`. `SPEEDUP_CAP=2.25` caps any residual fake speedup at `R=2.0` anyway. Sound iff `make_inputs` is injective on seed (true a.s. for `randn`) and the seed schedule is unpredictable (§4.5 fix).

**E6 — Import laundering / dynamic dispatch.** `getattr(torch,'mat'+'mul')`, `importlib`, `eval`/`exec`.
- Guard: `BANNED_NAMES = {eval, exec, __import__, compile}`, `getattr` rejected, `BANNED_IMPORTS = {importlib}`, sandbox `SAFE_BUILTINS`. Two independent layers (AST + sandbox).

**E7 — Trivial-no-op kernel + PyTorch compute.** Launches a no-op Triton kernel (passes LaunchCounter) but PyTorch does the work.
- Guard: E1 bans the PyTorch op; if the wrapper times the *whole* `solution()` it includes the PyTorch call → `speedup < 1.1 ⇒ speedup_score=0 ⇒ R ≤ 0.3 = θ`. PR is **not** a reliable discriminator here (§2.3).

### 6.3 Critical functional bug — `delegates_to_matrix_unit` (must fix)

The stub blocks matmul-class ops if `'tl.dot' in src` or `'tl.math' in src`. **This is backwards:** `tl.dot` *is* Triton's canonical tile-MMA primitive (compiles to tensor-core WMMA, **not** cuBLAS), and `tl.math.exp/erf` are legitimate fused intrinsics. As written, `matmul_bias_gelu` is **unwinnable** (use `tl.dot` → `R=0`; avoid it → scalar MAC loop slower than eager → `speedup_score=0`), so every matmul rollout scores 0 overnight, biasing the gradient toward elementwise ops.

**Fix:** delete the `tl.dot`/`tl.math` bans. cuBLAS delegation is already covered by (a) `torch.matmul/mm/bmm`+`torch.ops.aten` in `BANNED_CALLS` (E1 fix) and (b) `launches_timed > 0`. Replace with a **positive** requirement for matmul-class ops: source must contain `@triton.jit` **and** ≥1 `tl.load` **and** ≥1 `tl.store` (proves a real tile-reading/writing kernel). This is the single most urgent functional bug.

### 6.4 Red-team matrix

| Test | Exploit | Expected | Guard |
|---|---|---|---|
| T1 | `torch.matmul` passthrough | R=0 | E1 AST |
| T6 | `@triton.jit` never called | R=0 | E2 LaunchCounter |
| T8 | bf16 output downcast | R=0 | E3 dtype (residual: internal bf16) |
| T9/T10 | result-cache / scratchpad | R=0 | E5 fresh-input + post-seed |
| T11/T12 | no-op kernel + torch compute | R≤0.3 | E1 + speedup floor |
| T13 | hardcode M=2048 → eval M=3072 | R=0 | E4 shape disjointness |
| T14 | `getattr(torch,'matmul')` | R=0 | E6 AST getattr ban |
| T15 | `torch.ops.aten.mm.default` | R=0 | E1 prefix-ban (after fix) |
| T16 | legit `tl.dot` tiled matmul | R>0 | §6.3 fix (positive req) |

---

## 7. daVinci-kernel Integration (Eq1–Eq10): KEEP / CUT / LITE

All equations verbatim from arXiv:2606.16497 (verified PDF pp.3–6). Master simplification: `T=1 ⇒ G_{i,1}=R_i`, single Policy Agent, no skill co-evolution at training time.

| Eq | daVinci formula | Protean |
|---|---|---|
| **Eq1** reward | `R_{i,t}=C(y)·(1+speedup+PR)`, `PR=T_gen/T_tot` | **LITE** — additive Kevin-floor hybrid (§2.2), not a port |
| **Eq2** TRLOO | `A_{i,t}=G_{i,t}−(1/(N_t−1))Σ_{j≠i}G_{j,t}` | **CUT** — ship trl GRPO (mean+std); 12.5 % self-inclusion at G=8 accepted |
| **Eq3** MRS | `w_t=exp((1/|T_i|)Σ log(π_train/π_rollout))` | **CUT** — single-turn, on-policy; trl clip handles drift |
| **Eq4** PRS | `p_{i,t}=clip((PR−τ)/s,0,1)` | **CUT** — no-op at single-op scope (PR≈1) |
| **Eq5/6** skill trigger/verify | `R*>α·r_1 ∧ R*>β`; `r_verify≥max(β,α·r_1)` | **CUT** — Summary Agent deferred |
| **Eq7** policy LOO | `A^pol=G−(1/((k+1)n−1))Σ G` | **CUT** — no skill schemes |
| **Eq8** selection LOO | `A^sel_i=R̄_i−(1/k)Σ R̄` | **CUT** — no Selection Agent |
| **Eq9** summary LOO | `A^sum_m=R^sum_m−(1/(s−1))Σ R^sum` | **CUT** |
| **Eq10** combined loss | `L=L^pol+w_sel·L^sel+w_sum·L^sum` | **LITE** — only `L^pol` (w_sel=w_sum=0) |

**KEEP (LITE):** (1) PR as additive anti-passthrough bonus (decorative at single-op scope, §2.3); (2) static 5-skill BM25-retrieved prefix with skill-conditioned SFT.

**Skill prefix justification & caveat.** daVinci Ablation 1 (verified PDF p.8, **8B**): removing skill injection drops **Level 2 Fast₁ 44.8 % → 20.6 %** (Level 1 26.1→16.2, Level 3 10.1→2.0) — the largest single ablation drop. **But** that 24.2 pp drop is defined over the full 3-agent co-evolution daVinci *trains with*; Protean's static BM25 prefix (no Summary Agent, ~20 SFT examples) is architecturally closest to **Ablation 4** (policy RL + BM25, no LLM rerank: 8B L2 Fast₁ ≈ 51.2 but collapses to Fast₂ ≈ 4.9 vs full 7.9 — BM25 finds breadth, not the structural bottleneck). **Therefore Protean does not claim Ablation 1's benefit transfers; the static prefix is best-effort prompt engineering of unverified magnitude, and is disabled entirely if SFT is skipped (§3.8).**

**Honest demo language:** *"We integrated two daVinci techniques in LITE form: (1) the profiling-ratio reward term as an additive bonus — which at single-op scope contributes ~0 GRPO gradient and functions only as an anti-passthrough offset; (2) a static skill-injection prefix with skill-conditioned SFT. Full skill co-evolution (Eq5–9, 3-agent TRLOO) is deferred. We use trl GRPO (biased mean/std, 12.5 % self-inclusion at G=8), not daVinci TRLOO. Our novel contribution is the held-out continuous-shape generalization verifier, which daVinci does not have."*

---

## 8. Throughput & Resource Budget (one H100-80GB, colocate)

### 8.1 Why full FT is impossible — the 16P rule

`M_full = (2+4+4+4+2)·P = 16P`. For `P = 7.62e9`: **122 GB > 80 GB**. LoRA is mandatory.

### 8.2 VRAM ledger (itemized, gradient-checkpointing assumption explicit)

| Item | Size | Note |
|---|---|---|
| Base bf16 weights (trainer) | 15.24 GB | `7.62e9·2`, frozen, resident for forward |
| LoRA adapter (bf16) | 0.08 GB | ~40.5M params `[EST/PROBE]` |
| AdamW state (LoRA only) | 0.49 GB | `12·40.5M` — *not* the binding term |
| Activations | **4 GB** (checkpointing ON) | 8–16 GB if OFF |
| vLLM allocation (util=0.45) | 36 GB | own weight copy (~14 GB) + KV (~22 GB) |
| Triton compile scratch + bench subprocess | 4–6 GB | JIT + grader GPU use |
| **Total (ckpt ON)** | **≈ 60.8 GB** | **headroom ≈ 19 GB** |
| **Total (ckpt OFF)** | 65–73 GB | headroom 7–15 GB |

**Decisions:** gradient checkpointing is a **required assumption** (it changes both headroom *and* throughput — trl enables it by default). If margin is tight under concurrent Triton-JIT + vLLM CUDA-graph spikes, drop `vllm_gpu_memory_utilization` 0.45 → 0.38, or `vllm_enable_sleep_mode=True`, or `num_generations` 8→4. **KV math:** Qwen2.5-Coder-7B = `2·2·28·4·128 = 57,344 B/token` ≈ 56 KB; 22 GB KV ≈ 390k tokens ≈ 375 concurrent seqs at L=1024 `[EST]` (theoretical; PagedAttention fragmentation + CUDA-graph reserve reduce it).

### 8.3 Step-count arithmetic

```
t_step = t_gen + t_grade + t_backward
```

- `t_gen` (vLLM, G=8, L=1024): 3–8 s `[EST]`.
- `t_grade` (G=8 × ~10–90 s bench; **dominant term**): 20–80 s warm, up to 720 s cold-JIT.
- `t_backward` (LoRA, grad_accum=4): 5–10 s.

| scenario | t_step | steps in 14 h (50.4k s) |
|---|---|---|
| optimistic (warm, cached eager) | ~28 s | ~1800 |
| realistic | ~98 s | ~514 |
| pessimistic (cold JIT) | ~738 s | ~68 |

**Planning target ≈ 200 steps** (`t_step ≈ 250 s`). The **35× spread** is dominated by Triton subprocess latency, **not** vLLM. No public benchmark exists for trl-GRPO+vLLM-colocate on one H100/7B — all numbers are `[EST]`; the **10-step micro-probe is the go/no-go gate** `[PROBE]`.

### 8.4 Effective batch & cost

`B_eff = per_device(1) · grad_accum(4) · G(8) = 32` (prompt,completion) pairs/update (4 prompts × 8). Small → higher variance, memory-safe. Modal H100 = $3.95/hr; 14 h ≈ $55 + SFT ~$8 `[EST]` ≈ $63 of the $250 credit → ~4× headroom. **Throughput, not cost, is the gate.**

---

## 9. Task & Shape Sampling

### 9.1 Operator registry `O`

A task is `(o, s)`, `o` from a small fixed registry (`task_catalog.py`): `elementwise_add_relu`, `softmax`, `layernorm`, `matmul_bias_gelu`. The registry is intentionally tiny — **the generalization claim rests on shapes, not operator diversity** — so the op axis is fixed while the shape axis supplies effectively-unbounded tasks. Each op exposes `make_inputs(shape, dtype, seed)`, an eager `op_ref`, an op class (elementwise / reduction / matmul), and is roofline-classified (H100 compute-bound threshold `989 TFLOPs / 3.35 TB/s ≈ 295 FLOPs/byte`; elementwise/reduction are memory-bound → fusion is the target, matmul is compute-bound → tensor-core utilization is the target).

### 9.2 Deterministic sampler (sha256, not builtin `hash`)

`sampler.py`:

```
seed(op, idx, split) = int( sha256(f"{op}|{idx}|{split}")[:16], 16 )
rng = random.Random(seed)
M = rng.choice(pool); N = rng.choice(pool)      # pool = TRAIN_M if train else TEST_M
```

sha256 of a canonical string is mandatory for cross-machine/process determinism — Python's builtin `hash()` is randomized per process (`PYTHONHASHSEED`), which would break the disjointness guarantee across machines.

### 9.3 Frozen manifest

`manifest.freeze(ops, n_per_op, path)` calls `_assert_split_disjoint()`, samples `n_per_op` train + `n_per_op` test tasks per op, and `_atomic_write`s versioned JSONL (`.tmp` then `os.replace`). The loop consumes the **frozen file**, never re-samples → fully reproducible.

### 9.4 Complexity

Generation is `O(|O|·n_per_op)` time, `O(|O|·n_per_op)` space; trivially negligible. For matmul-class ops the K dimension needs a third axis with its own band + disjointness (currently unspecified — §11-Q9).

### 9.5 BUILD-BLOCKING import bugs (must fix before any run)

Verified by direct read; **`import protean` currently fails**:

1. **`shape_sampler.py`** imports `TRAIN_BANDS, HELD_OUT_BANDS, assert_disjoint` from `splits` and `sample_shape` from `sampler` — **none exist** (`splits.py` exports `TRAIN_M, TEST_M, _assert_split_disjoint`; `sampler.py` exports `sample_task`). **Fix:** either rename exports, or rewrite the wrapper to `from protean.splits import TRAIN_M, TEST_M, _assert_split_disjoint` and `from protean.sampler import sample_task` (one-file fix, preferred). The moat invariant **never fires** until this is fixed.
2. **`sampler.py`** references `TRAIN_M`/`TEST_M` as bare names inside `sample_task` with **no import** → `NameError`. **Fix:** add `from protean.splits import TRAIN_M, TEST_M`.
3. **`manifest.py`** calls `_assert_split_disjoint`, `sample_task`, `_atomic_write` with **zero imports** and `_atomic_write` is undefined → `NameError` on `freeze()`. **Fix:** add `import json, os; from protean.splits import _assert_split_disjoint; from protean.sampler import sample_task`; implement `_atomic_write(path, rows)` (write `path+'.tmp'` JSONL, `os.replace`).

**CI gate (Law):** `python -c 'import protean.shape_sampler'` and `python -m protean.manifest` must exit 0 before the Friday-night smoke test.

---

## 10. Evaluation Protocol & Ablations

### 10.1 Held-out eval pass (BLOCK 6)

After training, evaluate base `π_base` and trained `πθ` on the **frozen `S_test` manifest** only. For each `(op, s)`: G rollouts (best-of-k optional), grade each via the same `compute_reward` authority, record `correct`, `speedup`, raw `times` arrays. Compute `fast_1`, `fast_1.2`, `Gap_p`, Wilson CIs (rate), bootstrap CIs (speedup). Also report `fast_p@k` (best-of-k) to show test-time scaling.

### 10.2 Primary metrics

- `fast_1(πθ, S_test)`, `fast_1.2(πθ, S_test)` with Wilson 95 % CIs.
- `Gap_p` with the clustered-`n_eff` MDD caption (§5.5).
- Mean held-out speedup with bootstrap CI; correctness rate.
- **Train/test divergence:** `fast_p(πθ, S_train)` vs `fast_p(πθ, S_test)` — a large gap is the memorization signature; a small gap is the generalization signal.

### 10.3 Planned ablations (compute-permitting)

| Ablation | Tests |
|---|---|
| A1: no skill prefix | does the static skill prefix help (conditional on SFT, §7)? |
| A2: no SFT warm-start | does cold-start GRPO move at all (gradient-collapse risk)? |
| A3: `scale_rewards=False` (Dr.GRPO) | does removing std-norm reduce difficulty bias? |
| A4: `SPEEDUP_CAP` 2.25 vs log-speedup | does the flat region hurt? |
| A5: base vs trained on `S_train` | sanity: training shapes should improve more than held-out |

### 10.4 Pre-registered analysis

The money-slide analysis is **pre-registered** as: two-proportion comparison of `fast_1` with Wilson CIs, MDD computed at clustered `n_eff` (ρ̂ from pilot), framed "directional, single run." No post-hoc threshold tuning.

---

## 11. Remaining Open Questions

- **Q1 (saturation policy):** keep `SPEEDUP_CAP = 2.25` (honestly bounded, flat above 2.25×) vs adopt `log(speedup)/log(P_TARGET)` (unbounded, no flat region). Cap chosen for the 24 h build; log-form deferred. No ablation at 7B/<200 steps.
- **Q2 (std-norm debate):** `scale_rewards='group'` vs Dr.GRPO's no-std vs `'batch'`. Unsettled for high-variance kernel rewards; the 0.05 calibration threshold is a heuristic, not a power result.
- **Q3 (KL):** `β = 0` (trl default). Whether a small `β > 0` against the SFT checkpoint prevents non-kernel mode collapse over 300 steps is untested for narrow-domain kernel GRPO.
- **Q4 (PR signal):** at single-op scope PR contributes ~0 gradient (§2.3). A multi-op harness with competing GPU work would make `pr_frac < 1` a real discriminator — deferred. Option: **remove** PR entirely and reclaim the integration slot.
- **Q5 (μ>1):** `num_iterations=1` makes the clip inactive (REINFORCE). Whether μ=2–3 (clip binds, off-policy drift) improves sample efficiency on the 4-prompt batch is untested. clip-higher (`eps_high=0.28`) is a **no-op at μ=1** and absent from trl's public API — removed as a stretch claim unless μ≥2 + a trl fork.
- **Q6 (pretraining contamination):** whether `383/769/3072` appear in Qwen2.5-Coder pretraining is unknown; "off-grid shapes are rare" is plausible-but-unverified. The tiling-boundary argument for 3072 is stronger than the rarity argument. → directional evidence only.
- **Q7 (2-turn stretch):** if implemented, use Kevin SUM with `γ=0.4`, split the trajectory into 2 single-turn samples; leave TRLOO (12.5 % bias at G=8) for later.
- **Q8 (cold-JIT vs timeout):** no minimum-`n_rep` safety check yet (§4.3); a cold compile on a novel held-out shape can exhaust the subprocess timeout.
- **Q9 (matmul K-axis):** the third (K) dimension for `matmul_bias_gelu` needs its own disjoint band; currently `sample_task` draws only (M, N).
- **Q10 (clock-lock on Modal):** `nvidia-smi -lgc 1980` may be denied in Modal containers; the §4.2 verify-and-degrade path is the mitigation, success rate `[PROBE]`.
- **Q11 (LaunchCounter version-fragility):** patching `JITFunction.__call__` has no API-stability guarantee; the pinned-version smoke test + profiler fallback are the only guards.
