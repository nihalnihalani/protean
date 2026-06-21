# Protean Improvement Research & Roadmap (2026)

A prioritized, research-grounded roadmap for raising Protean's technical depth.
Every entry is anchored to a specific symbol in the **real current code** (read from
this worktree) and to a 2025–2026 source. Honesty markers are used throughout:
`[grounded]` = the change is directly implied by the cited paper and the current code;
`[claim]` = the paper's headline number is cited but not independently verified here;
`[speculative]` = a plausible extension we are proposing, not something the paper did.

Baseline at time of writing: **123 passed, 3 skipped** (126 collected) on
`/Users/nihalnihalani/Desktop/Github/protean-improvements/.venv/bin/python -m pytest -q`
(Python 3.12, CPU-only). Hard constraints from the project brief: keep all 123 green;
do not break public APIs (`grade_source`, `compute_reward`, `splits` public symbols,
`eval_protocol` public fns, `paired_report`); CPU-only here (real CUDA work must be a
flagged `gpu_only` script/runbook, never a unit test); additive over rewrites.

---

## 1. Executive summary

Protean is already a mature verifier-first kernel-RL environment: a hard-gated grader
(`grader.grade_source`), a log-scaled speedup reward with a PR bonus (`rewards.compute_reward`),
a 1M-parameter pure-Python policy head (`model/tiny_policy.TinyPolicyHead`), a real Triton
launch counter and AST anti-hack layer (`bench_core._TritonLaunchCounter`, `anti_hack.ast_clean`),
and a statistically-powered held-out eval (`eval_protocol.paired_report` with hierarchical
bootstrap + across-op sign test). The gaps that 2026 literature most directly addresses are
**not bugs** — they are missing mechanisms that the newest kernel-RL papers identify as the
highest-leverage upgrades over exactly the design choices Protean made.

Four themes dominate the 2026 work and map cleanly onto Protean:

1. **Reward shaping is moving from additive bonuses to bottleneck-aware *gates* and
   *milestones*.** Protean's `pr_bonus * pr_clamped` (a flat 0.2 additive term) is the
   exact pattern Dr. Kernel replaces with a multiplicative `(1 + PR)` factor plus
   profiling-based rejection of "lazy" rollouts. This is a sub-20-line change in
   `rewards.compute_reward` and is fully CPU-verifiable because `pr_frac` is already a
   float in the grader payload.

2. **Search is moving from exhaustive sweeps to bandits / quality-diversity.** Protean's
   `local_kernel_edits` sweeps 5 fixed block sizes every round regardless of history.
   KernelBand (UCB over a config space) and KernelFoundry (MAP-Elites archive) both
   target precisely this. The bandit/archive *bookkeeping* is pure Python and CPU-testable;
   only the kernel evaluation needs a GPU.

3. **Advantage estimation is moving to leave-one-out.** Protean's `TinyPolicyHead.train`
   uses vanilla REINFORCE (`grad_logits[action] -= advantage`). Dr. Kernel / Kevin use
   TRLOO (leave-one-out baseline). This is a pure-math change to a class that has no CUDA
   dependency, so it is fully unit-testable.

4. **Verifier hardening is converging on a documented exploit taxonomy.** SOL-ExecBench
   enumerates concurrency-injection, binary-embedding, state-caching, and FakeTensor
   exploits. Protean already blocks several roots in `anti_hack.BANNED_IMPORT_ROOTS` but
   leaves `threading`/`concurrent`/`base64`/`tempfile`/`pickle` open, and `bench_core`
   uses `isinstance`-style checks plus a lingering `sys.modules` entry. All closures are
   additive AST/identity checks that are CPU-verifiable.

**This PR (CPU-only) will implement the four lowest-risk, highest-leverage, fully
CPU-verifiable items** (PR-multiplicative reward + PR gate, TRLOO advantage, anti-hack
import-root expansion + tensor-subclass + module-tombstone, and per-rep timing
distribution stats with a pure-Python SOL bound). Everything requiring a real GPU
(outlier-input correctness, CUDA-graph timing, full MCTS/K-Search) is documented as a
deferred `gpu_only` runbook item. See §4.

---

## 2. Ranked improvements table

Impact is 1–5 (technical-depth × evaluation-credibility gain). Effort is XS/S/M/L.
`cpu_verifiable` = the *logic* can be unit-tested on CPU (the GPU measurement, if any,
is mocked or deferred). "Touches public API?" = no means strictly additive.

| # | Improvement | Where (real symbol) | Impact | Effort | cpu_verifiable | Public API? | Grounding |
|---|-------------|---------------------|:------:|:------:|:--------------:|:-----------:|-----------|
| 1 | PR-multiplicative reward `speedup*(1+PR)` + PR rejection gate | `rewards.compute_reward` | 5 | S | yes | no (new optional field/cap) | Dr. Kernel 2602.05885; daVinci 2606.16497 |
| 2 | TRLOO leave-one-out advantage in policy head | `tiny_policy.TinyPolicyHead.train`, `advantage_from_delta` | 4 | S | yes | no | Dr. Kernel 2602.05885; Kevin 2507.11948 |
| 3 | Expand `BANNED_IMPORT_ROOTS` (threading/base64/pickle/…) | `anti_hack.BANNED_IMPORT_ROOTS` | 4 | XS | yes | no | SOL-ExecBench 2603.19173 |
| 4 | Strict tensor-subclass identity + `sys.modules` tombstone | `bench_core.bench_source`, `load_solution` | 4 | S | yes (logic) | no | SOL-ExecBench 2603.19173; CPython importlib |
| 5 | Per-rep timing distribution (CV/IQR/trimmed-mean) | `bench_core._time_cuda` → dict | 4 | S | yes (mock times) | no (additive keys) | KernelBench 2502.10517; standardkernel.com |
| 6 | Pure-Python SOL roofline bound + `sol_efficiency` | new helper in `bench_core`; `OpSpec` fields | 4 | M | yes (mock BW) | no (additive keys) | SOL-ExecBench 2603.19173; MANTIS 2603.29010 |
| 7 | UCB bandit over (block_size, num_warps, num_stages) | `optimizer.run_optimization`, `policy.py` | 4 | M | yes | no (new `edit_policy`) | KernelBand 2511.18868 |
| 8 | Staged partial-credit reward (compile/correct milestones) | `rewards.compute_reward` | 3 | S | yes | no (additive) | SecureCodeRL 2601.01184; KernelBenchX 2605.04956 |
| 9 | Grader-feedback self-refine (caps fed into next prompt) | `fireworks_policy.build_kernel_edit_prompt` | 3 | S | yes (prompt str) | no (new optional param) | Self-Refine 2303.17651; STARK 2510.16996 |
| 10 | Best-of-N sampling + higher temperature | `fireworks_policy`, `optimizer` | 3 | S | yes (mock edits) | no | AutoTriton 2507.05687; KernelBench monkey runs |
| 11 | Bootstrap CI on `d_z` + rollout-parity assert in report | `eval_protocol.paired_report` | 3 | S | yes | no (additive keys) | Hidden Costs of RLVR 2509.21882 |
| 12 | MAP-Elites archive sidecar (quality-diversity parents) | `optimizer`, `rl_layer.accept_candidate` | 3 | M | yes (dict logic) | no (sidecar) | KernelFoundry 2603.12440; Kernel-Smith 2603.28342 |
| 13 | Outlier-input third correctness seed (float16 overflow) | `bench_core.bench_source` | 3 | S | logic only; check is GPU | no (new cap) | KernelBenchX 2605.04956; MultiKernelBench 2507.17773 |
| 14 | `sys.addaudithook` runtime import backstop | `bench_core` module init | 3 | M | yes | no | PEP 578; SOL-ExecBench 2603.19173 |
| 15 | Subprocess isolation for `bench_source` | `bench_core`/`grader` | 4 | L | yes (mock) | no | SOL-ExecBench 2603.19173; CodeJail |
| 16 | Kevin γ-discounted multi-turn credit | `optimizer`, `tiny_policy` | 3 | M | yes | no | Kevin 2507.11948 |
| 17 | Shrinkage / per-op normalized GRPO advantage | `train/grpo_loop`, new `shrinkage_baseline.py` | 3 | M | yes | no | Shrinking the Variance 2511.03710; VL Norm 2509.07558 |
| 18 | BCa bootstrap CI (opt-in, n_ops≥5) | `eval_protocol.hierarchical_bootstrap_ci` | 2 | M | yes | no (new mode) | Efron 1987; sim study 2404.12967 |
| 19 | CUDA-graph timing path for sub-10µs kernels | `bench_core` (gpu_only) | 3 | M | skeleton only | no | Triton `do_bench_cudagraph`; 2501.09398 |
| 20 | K-Search / MCTS trajectory search | `optimizer` | 5 | L | tree only; value=GPU | yes (restructure) | K-Search 2602.19128; OptiML 2602.12305 |

**This-PR set (CPU-only, fully verifiable): #1, #2, #3, #4, #5, #6** (see §4). Items #7–#12
are the recommended *next* CPU-verifiable batch. Items #13, #15, #19, #20 are GPU-blocked
or large-restructure and are deferred with runbook stubs.

---

## 3. Per-area deep dives

### 3.1 Reward shaping (`rewards.py`)

**Current code.** `compute_reward` (lines 54–123) computes
`reward = correctness_reward + speedup_reward + (pr_bonus * pr_clamped)`, where
`speedup_reward = speedup_reward_weight * speedup_score` and `speedup_score` is the
log-compressed, clamped ratio `log(s/floor)/log(cap/floor)`. `pr_frac` already arrives as
a float in the payload (computed in `bench_core.measure_pr_frac` via `torch.profiler`).
The PR term is a flat additive `0.2 * pr_clamped` and there is no rejection of low-PR
rollouts.

**Change #1 — PR-multiplicative term + gate `[grounded]`.** Dr. Kernel
(arXiv:2602.05885) uses `R = C(y) * (1 + speedup + PR)` and a Profiling-based Rejection
Sampling step (PRS, τ≈0.3) that drops rollouts whose profiling ratio is too low so the
policy cannot earn speedup on a kernel where the Triton launch is not the bottleneck.
daVinci-kernel (arXiv:2606.16497) adopts the same PR term. Concretely:

- Replace `speedup_reward = config.speedup_reward_weight * speedup_score` with
  `speedup_reward = config.speedup_reward_weight * speedup_score * (1.0 + pr_clamped)`
  and remove `pr_bonus` as a *separate additive* term (keep the field for back-compat /
  load path). To preserve all 123 tests, gate the new multiplicative behavior behind a new
  `RewardConfig` field that defaults to the old behavior, e.g. `pr_multiplicative: bool = False`
  — then flip it to `true` only in the production `reward_config.json`.
- Add `pr_frac_gate: float = 0.0` to `RewardConfig` (default keeps tests green). When
  `pr_frac < config.pr_frac_gate`, emit a new cap `below_pr_gate` and zero the speedup
  reward (the kernel is "lazy" — correct but not doing the work on the timed path).
  Production value: `0.3` (Dr. Kernel τ).

Both are pure arithmetic on values already in the payload — one new test in
`tests/test_rewards.py` covering the multiplicative arithmetic and one covering the gate
suffice. `[claim]` Dr. Kernel reports its 14B reaching Claude-4.5-Sonnet-competitive
results on KernelBench Level-2 with 31.6% of kernels ≥1.2× speedup; daVinci reports a 46%
improvement over Dr. Kernel on KernelBench Level-3 — these headline numbers are cited, not
re-verified here.

**Change #8 — staged partial credit `[grounded]`.** Today the reward jumps from `0.0`
(incorrect / any hard cap) to `correct_floor=0.3` (correct-but-slow). SecureCodeRL
(arXiv:2601.01184) and KernelBenchX (arXiv:2605.04956) both report that decomposing the
reward into syntax → compile → execute → correct milestones gives denser gradients when
the model mostly produces failures. Additive plan: emit a `partial_reward_stage` int in the
payload mapped from the caps that already exist (`ast_clean` fail → 0; `no_triton_jit` → 1;
`cuda_unavailable` with a valid jitted kernel → 2; correct-but-below-floor → 3; above-floor →
4). Stages 0–2 are decided entirely by `anti_hack` checks already run in `grader.grade_source`,
so they are CPU-verifiable. Keep the *scalar* `reward` unchanged by default (a new
`compile_ok_reward: float = 0.0` field, set >0 only in production config) so the test suite is
untouched.

### 3.2 Policy-head training (`model/tiny_policy.py`)

**Current code.** `TinyPolicyHead.train` (lines 144–182) is vanilla REINFORCE:
`grad_logits = [advantage * prob ...]; grad_logits[action] -= advantage`. `advantage_from_delta`
(lines 79–84) maps a per-trial delta dict to a scalar. There is no baseline, no
normalization, no entropy term.

**Change #2 — TRLOO leave-one-out baseline `[grounded]`.** Dr. Kernel (arXiv:2602.05885)
and Kevin (arXiv:2507.11948) both note that the standard GRPO in-batch mean *includes the
current sample*, biasing the baseline. TRLOO subtracts the mean of the *other* samples:
`A_i = G_i - (1/(N-1)) Σ_{j≠i} G_j`. In Protean, group examples by `action` (block-size arm),
then for each example subtract the mean advantage of the *other* examples in its action group
before the gradient step. This is pure Python over `list[TraceExample]` with no CUDA
dependency. A unit test asserts: (a) the leave-one-out baseline of a 3-element group equals
the mean of the other two; (b) a single-element group degrades gracefully to the raw
advantage. Keep `train()`'s public signature; the LOO transform happens inside.

**Companion (low-risk) `[grounded]`.** Add advantage normalization (subtract batch mean,
divide by std) and an optional entropy bonus — standard 2025 stability practice (Scaf-GRPO
2510.19807; "GRPO is Secretly a PRM" 2509.21154). Both are additive; existing training tests
check directional improvement, which normalized advantages preserve.

### 3.3 Search / optimizer (`optimizer.py`, `model/policy.py`, `model/rl_layer.py`)

**Current code.** `run_optimization` (lines 85–306) defaults to `max_rounds=1` and, for the
local policy, calls `local_kernel_edits(best_source)` which yields exactly 5 edits — one per
fixed block size — every round (`policy.py` lines 36–45). Acceptance is strict greedy via
`accept_candidate`. The action space is *only* block size; `num_warps`/`num_stages` are never
touched.

**Change #7 — UCB bandit over a joint config space `[grounded]`.** KernelBand
(arXiv:2511.18868) replaces exhaustive sweeps with a multi-armed bandit selecting
`(cluster, strategy)` arms by `μ̂ + c·sqrt(ln t / N)`. For Protean: arms = the cross product
`{128,256,512,1024,2048} × num_warps{4,8,16} × num_stages{2,3,4}` (≈45 configs). Add a
`BanditSearch` class holding `counts`/`values` dicts keyed by config tuple; replace the
flat enumeration with a UCB-ranked single sample per round behind a new `edit_policy="bandit"`
branch (existing branches untouched). Add `_replace_warps_and_stages` next to the existing
`_replace_block_size` in `policy.py`. The bandit math, config-string editing, and JSONL
serialization are pure Python: `tests/test_bandit_optimizer.py` can assert that after N
mocked-reward rounds the bandit concentrates >80% of pulls on the best arm. The hardware-aware
*mask* (prune arms whose bottleneck is saturated) is GPU-only and deferred. `[claim]`
KernelBand reports >33% average improvement over prior art on TritonBench-G across 3 GPUs.

**Change #12 — MAP-Elites archive sidecar `[speculative→grounded]`.** Today the optimizer
keeps a single `best_source`, discarding good-but-not-best parents. KernelFoundry
(arXiv:2603.12440) and Kernel-Smith (arXiv:2603.28342) keep a quality-diversity archive.
Minimal version: a `dict{(block_bucket, op) -> best_source_seen}` updated alongside the scalar
best; parents for the next round are sampled softmax-over-reward from the archive instead of
always mutating the single best. The archive is a pure dict (CPU-testable: assert two distinct
block sizes that each win on different shapes both persist). The `best_kernel_<op>.py` and
JSONL APIs are unchanged; the archive is an optional sidecar file. The *behavioral descriptors*
KernelFoundry uses (d_mem/d_algo/d_sync from AST analysis) can reuse `anti_hack`'s AST walk —
`[speculative]` as applied to these specific axes.

**Change #16 — Kevin γ-discounted multi-turn credit `[grounded]`.** Kevin (arXiv:2507.11948)
shows serial refinement beats parallel sampling and scores each turn with a γ-discounted
look-ahead return (γ≈0.4). Protean's `trials.jsonl` already records per-round deltas; with
`max_rounds>1`, `advantage_from_delta` can be extended to weight earlier rounds by
`γ^(T−t)` when they lead to better later kernels. Pure-Python arithmetic; benefit is clearest
once `max_rounds>1` is the default in overnight runs.

**Change #20 — K-Search / MCTS trajectory search `[grounded, deferred]`.** K-Search
(arXiv:2602.19128) and OptiML (arXiv:2602.12305) replace the flat trial-and-keep loop with a
tree/priority-queue that allows temporary regressions (MCTS backprop) and decouples high-level
*intent* from low-level implementation. The tree/UCT *data structure and selection policy* are
CPU-verifiable, but the node value function calls the grader (GPU). This is a large
restructure of `run_optimization` and is deferred; only a CPU-testable tree+UCT skeleton is
in scope later.

### 3.4 LLM policy (`model/fireworks_policy.py`)

**Current code.** `build_kernel_edit_prompt` (lines 42–68) is static; the optimizer calls
`fireworks_kernel_edit` once per round with `temperature=0.2`, `reasoning_effort="low"`, and
ignores the previous trial's rejection reasons even though `optimizer.run_optimization` already
logs `eval_error`, `caps`, and `delta_vs_best`.

**Change #9 — grader-feedback self-refine `[grounded]`.** Self-Refine (arXiv:2303.17651) and
STARK (arXiv:2510.16996) show iterative refinement with execution feedback beats single-pass.
Add an optional `last_rejection: dict | None = None` to `build_kernel_edit_prompt` and inject a
"Previous attempt was rejected because: {caps}; speedup delta was {…}. Avoid these failure
modes." block; thread the last trial's metadata from the optimizer loop. The prompt builder is
pure Python — unit-test that passing `caps=['below_speedup_floor']` makes the string contain
the reason. `[claim]` STARK reports up to 16× faster kernels vs baseline on KernelBench.

**Change #10 — Best-of-N + temperature `[grounded]`.** AutoTriton (arXiv:2507.05687) and the
well-known KernelBench "monkey" result (sampling DeepSeek-V3 100× raised fast_1 from ~4% to
~37% on Level-2) show diverse sampling + a verifier oracle is cheap and effective. Raise the
generation temperature (0.2 → ~0.8) and request N candidates per round (loop N calls or use the
endpoint `n`), then keep the best by the existing `score()`. The selection logic is
CPU-testable by mocking `fireworks_kernel_edit` to return N fake edits and asserting the
highest-`score` one wins.

**Change (companion) — concise reasoning prompt `[grounded]`.** ConCuR/KernelCoder
(arXiv:2510.07356) find shorter reasoning correlates with higher speedup for kernels; the
prompt can explicitly request "brief, direct reasoning" and cap the token budget.

### 3.5 Verifier security (`anti_hack.py`, `bench_core.py`)

**Current code.** `BANNED_IMPORT_ROOTS` (lines 28–40) blocks `importlib/subprocess/os/sys/
pathlib/ctypes/cffi/cupy/pycuda/cuda/numpy` but **not** `threading`, `concurrent`,
`multiprocessing`, `_thread`, `base64`, `binascii`, `tempfile`, `shutil`, `zipfile`, `tarfile`,
`pickle`, `shelve`. `bench_core.bench_source` (lines 245–252, 279–285) compares outputs with
`torch.allclose` and uses attribute access (`out_candidate.dtype`) without a strict
`type(out) is torch.Tensor` identity check. `load_solution` (lines 96–112) registers the
candidate in `sys.modules[name]` and never removes it, so globals/monkey-patches can survive
across candidates.

**Change #3 — expand `BANNED_IMPORT_ROOTS` `[grounded]`.** SOL-ExecBench (arXiv:2603.19173)
documents a concurrency-exploit family (hide GPU work on background threads / `torch.jit.fork`,
the latter already prefix-banned) and a binary-embedding family (decode a base64 cubin to a
tempfile and load it). Adding the 12 roots above closes both vectors with pure AST logic. Each
new root gets one fixture in `tests/test_hardening.py` mirroring the existing import-ban tests.

**Change #4 — strict tensor-subclass identity + module tombstone `[grounded]`.** SOL-ExecBench
documents a state-caching / FakeTensor family where a candidate returns a `torch.Tensor`
*subclass* that passes `isinstance`/`allclose` but carries no real data. Add
`type(out_candidate) is torch.Tensor` immediately after `solution(*inputs)` (emit a new cap,
e.g. `non_tensor_output`). Wrap `bench_source` (or `load_solution`) in a `try/finally` that
pops `sys.modules[name]` and unlinks the temp file, preventing cross-candidate state bleed
(CPython importlib docs; the lingering-module class of issue). The cap definition, the
`type() is` predicate, and the `sys.modules.pop` are all CPU-testable with a mock module /
mock tensor subclass; the actual `solution()` call path remains GPU-only.

**Change #14 — `sys.addaudithook` runtime backstop `[grounded]`.** An adversary who obfuscates
a banned name past static AST (e.g. building it via string concat and `__import__`, though
`__import__` is name-banned) could still import at runtime. PEP 578's audit hook can raise on
any `import` audit event whose module matches `BANNED_IMPORT_ROOTS`, installed before
`exec_module`. PEP 578 explicitly notes hooks are defense-in-depth, not a sandbox — documented
as such.

**Change #15 — subprocess isolation `[grounded, GPU-blocked logic]`.** SOL-ExecBench runs each
solution in a dedicated subprocess so OOM/SIGKILL/monkey-patches cannot affect later evals.
Because `bench_source` returns a JSON-serializable dict, the boundary is clean. The subprocess
*wrapper* is CPU-testable with a mock `bench_source`; the real run needs a GPU. Medium-to-large
effort; deferred.

### 3.6 Measurement rigor (`bench_core.py`, `eval_protocol.py`)

**Current code.** `_time_cuda` (lines 173–191) returns a single `statistics.median(times)`
float with an L2 flush already in place; per-rep timings are collected then discarded.
`bench_source` returns a fixed dict. `eval_protocol.paired_report` (lines 117–141) already
computes `observed_dz` and a hierarchical bootstrap CI on the grand-mean delta, but reports no
CI on `d_z` and does not assert rollout parity.

**Change #5 — per-rep timing distribution `[grounded]`.** KernelBench (arXiv:2502.10517) and
standardkernel.com recommend reporting variability (CV, trimmed mean) so noisy measurements
don't silently contaminate the speedup signal. Change `_time_cuda` to return a dict
`{median_ms, mean_ms, cv, iqr_ms, trimmed_mean_ms}` (trimmed = drop top/bottom 10%) and forward
the extra keys through `bench_source`. `compute_reward`/`grade_source` keep consuming the
scalar `t_kernel_ms`/`t_eager_ms` (unchanged signatures); the new keys are additive metadata.
CPU-testable by feeding `_time_cuda` a mock list of times via a tiny seam and asserting the
CV/IQR/trimmed-mean math.

**Change #6 — pure-Python SOL roofline bound `[grounded]`.** SOL-ExecBench (arXiv:2603.19173)
and MANTIS (arXiv:2603.29010) anchor the score to a hardware Speed-of-Light bound
`T_SOL = max(FLOPs/peak_compute, bytes/peak_bw)`. All three Protean ops are memory-bound, so
`bytes` follows from shape and dtype alone. Add `analytic_bytes_per_element` /
`analytic_flops_per_element` to `OpSpec` (or a helper keyed by op name), compute `sol_bound_ms`
(pure math — `peak_bw` mocked in tests), and report `sol_efficiency = t_kernel_ms /
sol_bound_ms`. This is fully CPU-verifiable (the arithmetic) and gives a hardware-grounded
stopping criterion. A `below_sol_bound` integrity cap (a kernel timed *faster* than the SOL
bound is physically impossible → timing game) is a natural anti-hack sentinel, but the *check*
needs the real `t_kernel_ms` so it lives in the GPU path.

**Change #11 — bootstrap CI on `d_z` + rollout-parity assert `[grounded]`.** "Hidden Costs of
RLVR" (arXiv:2509.21882) recommends budget parity and effect-size CIs. `paired_report` can add
three additive keys: `d_z_ci95` (reuse the existing B=10000 bootstrap loop with `d_z` as the
statistic), `power_at_observed_dz` (one `_norm_ppf` call), and a runtime assert that base and
trained were evaluated with equal rollout counts. Pure Python; new keys only.

**Change #13 — outlier-input third correctness seed `[grounded, GPU check]`.** KernelBenchX
(arXiv:2605.04956) and MultiKernelBench (arXiv:2507.17773) add an outlier mode (≈0.1% of
elements scaled ×50) that exposes float16 overflow in softmax/RMSnorm that N(0,1) inputs miss.
Add a third check at seed=45 with injected outliers, emitting `incorrect_outlier_input`. The
cap definition and a CPU test *stub* are safe here; the actual numerical check needs CUDA, so
the live check is `gpu_only`.

**Change #18 — opt-in BCa bootstrap `[grounded, low priority]`.** `hierarchical_bootstrap_ci`
uses a percentile CI. BCa (Efron 1987) is more rigorous in principle but the 2024 simulation
study (arXiv:2404.12967) found it no better — and sometimes worse — in heavy-tailed or small-K
settings. With only K=3 `REAL_OPS`, the op-level jackknife for `a_hat` is unreliable. Add BCa
as an *opt-in* `ci_method='bca'` gated by `n_ops>=5`, keeping percentile as default. Honestly
low-value until N_OPS grows.

### 3.7 GRPO path (`train/grpo_loop.py`)

**Change #17 — shrinkage / per-op normalized advantage `[grounded]`.** When a GRPO group all
hits the same cap (all 0.0), group std is 0 and the gradient starves (arXiv:2605.07689). A
shrinkage baseline blending per-prompt leave-one-out mean with the batch mean
(arXiv:2511.03710) and a per-op reward normalization (subtract `mean_reward_by_op` before
advantage; arXiv:2509.07558, 2603.16158) are pure-Python pre-processors that slot into a
reward-fn wrapper before TRL's optimizer step. CPU-testable on a reward batch; the full GRPO
run is GPU.

---

## 4. What we implement in THIS PR vs defer

### In this PR (CPU-only, fully verifiable, additive, no public-API breakage)

- **#1 Reward: PR-multiplicative term + PR rejection gate** (`rewards.py`). New
  `RewardConfig.pr_multiplicative: bool = False` and `pr_frac_gate: float = 0.0` (defaults
  preserve all 123 tests); production `reward_config.json` flips them on. New cap
  `below_pr_gate`. New tests in `tests/test_rewards.py`. Grounding: Dr. Kernel 2602.05885,
  daVinci 2606.16497.
- **#2 Policy: TRLOO leave-one-out advantage** (`tiny_policy.py`). Group by action, subtract
  leave-one-out baseline before the REINFORCE step; optional advantage normalization. New tests
  in `tests/test_tiny_policy.py`. Grounding: Dr. Kernel 2602.05885, Kevin 2507.11948.
- **#3 Anti-hack: expand `BANNED_IMPORT_ROOTS`** with threading/concurrent/multiprocessing/
  _thread/base64/binascii/tempfile/shutil/zipfile/tarfile/pickle/shelve (`anti_hack.py`). New
  fixtures in `tests/test_hardening.py`. Grounding: SOL-ExecBench 2603.19173.
- **#4 Verifier: strict `type(out) is torch.Tensor` cap + `sys.modules` tombstone**
  (`bench_core.py`). New cap `non_tensor_output`; `try/finally` cleanup. CPU tests with a mock
  tensor subclass and `sys.modules` assertion. Grounding: SOL-ExecBench 2603.19173, CPython
  importlib.
- **#5 Measurement: per-rep timing distribution** (`_time_cuda` → dict; additive keys through
  `bench_source`). CPU test via a mock-times seam. Grounding: KernelBench 2502.10517.
- **#6 Measurement: pure-Python SOL bound + `sol_efficiency`** (`bench_core.py`, `OpSpec`).
  CPU test with mocked peak bandwidth. Grounding: SOL-ExecBench 2603.19173, MANTIS 2603.29010.

All six are pure Python, keep `grade_source`/`compute_reward`/`paired_report`/`splits` public
symbols intact (additions only), and are verifiable by extending the existing 123-test suite.

### Recommended next CPU batch (not this PR)

#7 UCB bandit, #8 staged partial credit, #9 self-refine prompt, #10 best-of-N, #11 d_z CI +
rollout-parity, #12 MAP-Elites sidecar, #14 audit-hook backstop, #16 γ-discounted credit,
#17 shrinkage/per-op normalization, #18 opt-in BCa. All CPU-verifiable; deferred only for PR
scope, not capability.

### Deferred to GPU / runbook / large restructure

- **#13 outlier-input correctness** — cap + stub CPU-safe; live check needs CUDA → `gpu_only`.
- **#15 subprocess isolation** — wrapper CPU-testable; real isolation needs the GPU eval loop.
- **#19 CUDA-graph timing** — skeleton returns `None` on CPU; real path is `gpu_only`
  (`triton.testing.do_bench_cudagraph`).
- **#20 K-Search / MCTS** — tree+UCT skeleton CPU-testable; value function is GPU; large
  restructure of `run_optimization`.
- **GPU clock locking** (`nvidia-smi --lock-gpu-clocks`) — runbook only; cannot be a unit test.
- **SFT/LoRA distillation from accepted traces** — data pipeline (`build_sft_dataset` over
  `trials.jsonl` filtering `accepted==True`) is CPU-verifiable; fine-tuning is `gpu_only`.

---

## 5. Bibliography

Sources are cited as the project memory recorded them (2025–2026 kernel-RL literature). arXiv
IDs are reproduced verbatim; treat headline performance numbers as `[claim]` unless
independently reproduced on hardware.

- **Dr. Kernel** — *Reinforcement Learning Done Right for Triton Kernel Generations*,
  arXiv:2602.05885 (Feb 2026). PR-multiplicative reward `R=C(1+speedup+PR)`, Profiling-based
  Rejection Sampling (τ≈0.3), TRLOO advantage. Used for #1, #2, #8.
- **daVinci-kernel**, arXiv:2606.16497 (Jun 2026). Adopts the PR term; evolving skill library;
  three co-trained agents. Used for #1.
- **Kevin** — arXiv:2507.11948 (Cognition, Jul 2025). Multi-turn γ-discounted credit (γ≈0.4),
  Clip-Higher, KL=0, serial > parallel refinement. Used for #2, #16.
- **KernelBand** — *Hardware-Aware Multi-Armed Bandits*, arXiv:2511.18868 (Nov 2025).
  Masked-UCB over (cluster, strategy) arms. Used for #7.
- **KernelFoundry** — *Hardware-Aware Evolutionary GPU Kernel Optimization*, arXiv:2603.12440
  (Mar 2026). MAP-Elites 3D archive + meta-prompt co-evolution. Used for #12.
- **Kernel-Smith** — *A Unified Recipe for Evolutionary Kernel Optimization*,
  arXiv:2603.28342 (Mar 2026). Population archive of executable candidates. Used for #12.
- **K-Search** — arXiv:2602.19128 (Feb 2026). Intent-separated world-model tree search.
  Used for #20.
- **OptiML** — *Program Synthesis and CUDA Kernel Optimization*, arXiv:2602.12305 (Feb 2026).
  MCTS over edit trajectories with regression guardrails. Used for #20.
- **AutoKernel** — arXiv:2603.21331 (Mar 2026). Six-tier optimization playbook, move-on
  criteria, 5-stage correctness harness. Used for #9 context.
- **AutoTriton** — arXiv:2507.05687 (Jul 2025). SFT→GRPO with execution reward; diverse
  sampling. Used for #10, SFT-distillation defer.
- **TritonRL** — arXiv:2510.17891 (Oct 2025). Hierarchical reward decomposition; hidden-stream
  injection exploit. Used for #3, SFT defer.
- **TritonForge** — arXiv:2512.09196 (Dec 2025). Two-stage warm-start benchmark harness.
  Context for anytime harness.
- **ConCuR / KernelCoder** — arXiv:2510.07356 (Oct 2025). ARL difficulty metric; concise
  reasoning correlates with higher speedup. Used for §3.4 companion.
- **STARK** — *Strategic Team of Agents for Refining Kernels*, arXiv:2510.16996 (2025).
  Iterative profiling feedback. Used for #9.
- **Self-Refine** — arXiv:2303.17651 (NeurIPS 2024). Used for #9.
- **SOL-ExecBench** — arXiv:2603.19173 (NVIDIA/Microsoft, Mar 2026). SOL score; exploit
  taxonomy (concurrency, binary embedding, state caching, FakeTensor, env manipulation);
  subprocess isolation; clock locking. Used for #3, #4, #6, #14, #15.
- **MANTIS** — arXiv:2603.29010 (Mar 2026). SOL-guided LLM steering; ROI exponent vs SOL gap.
  Used for #6.
- **KernelBench** — arXiv:2502.10517 (Feb 2025). fast_p metric; CV<3% timing threshold;
  best-of-N "monkey" result. Used for #5, #10.
- **KernelBenchX** — arXiv:2605.04956 (May 2026). Standard + outlier correctness modes; task
  structure explains 3× more variance than method. Used for #8, #13.
- **MultiKernelBench** — arXiv:2507.17773. N=5 seeds, atol/rtol 1e-2. Used for #13.
- **robust-kbench** — arXiv:2509.14279 (Sakana AI, 2025). Diverse init states to prevent
  hardcoding. Context for #13.
- **SecureCodeRL** — arXiv:2601.01184 (Jan 2026). 4-stage partial-credit reward. Used for #8.
- **Shrinking the Variance** — arXiv:2511.03710 (2025). Shrinkage baseline for GRPO. Used for
  #17.
- **VL Norm / unbiased loss aggregation** — arXiv:2509.07558 (2025). Used for #17.
- **Execution-Grounded Credit Assignment for GRPO** — arXiv:2603.16158 (2026). Used for #17.
- **Gradient Starvation in Binary-Reward GRPO** — arXiv:2605.07689. Used for #8, #17.
- **Hidden Costs of RLVR** — arXiv:2509.21882 (2025). Budget parity + saturation curves.
  Used for #11.
- **LLMs Gaming Verifiers / Isomorphic Perturbation Testing** — arXiv:2604.15149 (ICLR 2026
  workshop). Used for §3.5 context (`[speculative]` as applied to kernels).
- **Scaf-GRPO** — arXiv:2510.19807 (2025). Scaffolded group sampling. Used for #2 companion.
- **GRPO is Secretly a PRM** — arXiv:2509.21154 (2025). Advantage variance drives instability.
  Used for #2 companion.
- **BCa bootstrap** — Efron 1987; comparative simulation arXiv:2404.12967 (2024). Used for #18.
- **PEP 578** — Python Runtime Audit Hooks (CPython 3.8+). Used for #14.
- **CUDA Graph batching** — arXiv:2501.09398 (Jan 2025); Triton `do_bench_cudagraph` docs.
  Used for #19.

---

*Honesty note.* The most defensible claims here are the *code-grounded mechanism gaps*
(additive bonus vs multiplicative gate; vanilla REINFORCE vs TRLOO; missing import-root bans;
scalar median vs distribution; no SOL anchor) — these are read directly from the current
source. The *quantitative* paper results (speedup percentages, KernelBench levels) are cited as
the literature reports them and are not reproduced on hardware in this CPU-only worktree.
Items marked `[speculative]` (kernel-specific isomorphic perturbation; AST-derived MAP-Elites
behavioral axes) are proposed extensions, not results from the cited papers.
