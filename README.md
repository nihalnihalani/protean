<div align="center">

# 🧬 Protean

### Kernels that are fast on *any* shape.

**An RL environment + overnight optimizer that trains models to write GPU kernels which stay fast on tensor shapes they have never seen — graded by a verifier you can't fake.**

🌐 **Live dashboard:** [protean-khaki.vercel.app](https://protean-khaki.vercel.app) · 🔁 **HUD env** · 💻 **[GitHub](https://github.com/nihalnihalani/protean)**

*HUD × YC Frontier RL Environments Hackathon · Track: ML Research (GPU)*

</div>

---

## ⚡ TL;DR

Protean turns GPU-kernel optimization into a **HUD reinforcement-learning environment**. A model proposes a Triton
kernel; a hidden, clock-locked verifier grades it for **correctness × measured wall-clock speedup** — on **held-out,
off-grid shapes the kernel never saw**; an optimizer keeps only strict improvements and writes a full audit trail of
every edit, benchmark, failure, token, and decision.

> **The one thing that wins:** every other 2026 kernel-RL system (Kevin, Dr.Kernel, daVinci-kernel, TritonForge)
> trains *and* tests on the **same** shape distribution. KernelBench's own roadmap lists a shape-sweep generalization
> verifier as *unshipped*. **Protean ships exactly that** — so a high reward proves *reasoning*, not *memorization*.

### 💰 Money figure (live, NVIDIA B200 — see the [dashboard](https://protean-khaki.vercel.app))

| Metric | Value |
|---|---|
| **Best speedup** | **4.62×** vs PyTorch eager |
| Best latency | 0.0310 ms |
| Accepted / trials | **17 / 24** |
| Hardware | NVIDIA B200 · Fireworks coding agent + 1M learned controller |

---

## 🎯 The problem

GPU kernels are the bottleneck of the entire inference economy, and **LLMs are bad at writing them**: frontier models
beat PyTorch on <20% of KernelBench tasks, and ~47% of the kernels they *do* get correct are still slower than eager.
Worse, the kernels that pass a benchmark **silently fail on shapes within the same class that weren't in the test set**
(robust-kbench, arXiv:2509.14279) — a hardcoded `BLOCK=512` tiles `1536` evenly but produces wrong output at `3073`.
So "it passed the benchmark" tells you almost nothing about whether the model *understands* kernels.

## 🧬 The idea (and the name)

*Protean* — from Proteus, the sea-god who took any form — means **able to adapt to whatever shape is needed.** That is
the thesis: a kernel-writing policy that generalizes across tensor shapes. We enforce it with a **disjoint-by-construction
shape split**:

- **Train shapes:** the power-of-two grid `{1024, 2048, 4096}`.
- **Held-out shapes:** continuous **off-grid** sizes (`1535, 3073, 6143`, plus a sampler over `mod 64 ≠ 0`,
  prime-adjacent values) — provably absent from training and from every published eval. A memorized fixed-block kernel
  *must* mask correctly to score, so the held-out reward measures generalization, not recall.

---

## 🏗️ How it works

```
            ┌──────────────────────────────────────────────────────────┐
   model    │  HUD env  (src/protean/env.py)  — real @env.template      │
  (Fireworks│   yield prompt(op, held-out shape) ─▶ agent emits kernel  │
   / 1M ctrl│   ◀─ yield EvaluationResult(reward, subscores, content)   │
   / human) └───────────────────────────┬──────────────────────────────┘
                                         │ grade_source()  (single authority)
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
   GPU      │  Verifier  — hidden, root:700, fail-closed                │
  (B200 /   │   AST anti-hack ▸ Triton-launch counter ▸ subclass-identity│
   H100)    │   compile ▸ allclose(fresh inputs) ▸ CUDA-event timing     │
            │   reward = correct × log-speedup (+ profiling-ratio gate)  │
            └───────────────────────────┬──────────────────────────────┘
                                         │ scalar reward + rich trace
                                         ▼
            ┌──────────────────────────────────────────────────────────┐
   loop     │  Optimizer  (deterministic ▸ UCB bandit ▸ Fireworks ▸ 1M) │
            │   keep strict improvements · log every trial · stream→HUD │
            └──────────────────────────────────────────────────────────┘
```

Every path — local optimizer, HUD platform eval, training reward — flows through the **one** `grade_source()`
authority, so a kernel that scores locally scores identically on the dashboard.

---

## 🛡️ The verifier you can trust

A verifier for untrusted, model-written code is only useful if it can't be gamed. Protean closes the known exploit
families (taxonomy from **SOL-ExecBench**, arXiv:2603.19173):

| Exploit | Defense |
|---|---|
| PyTorch passthrough / `torch.ops.aten.*` dispatch | AST ban (imports **and** dotted call-chains) |
| "Defined but never launched" `@triton.jit` | **runtime Triton-launch counter** (0 launches → reward 0) |
| FakeTensor / subclass that fakes `allclose` | **strict `type(x) is torch.Tensor`** identity check |
| Concurrency / binary-embedding / network exfil | expanded import-allowlist + `sys.addaudithook` backstop |
| Input/shape overfit | **fresh random inputs each grade**, on **held-out off-grid shapes** |
| Timer gaming (cache, async) | CUDA events + warmup + L2 flush + post-timing re-check |

The audit hook is **scoped to the candidate-exec window** (it does not impede the host process or torch's own threads),
and the design for true OS-level isolation (nsjail + seccomp + cgroups + GPU MIG, per `docs/SANDBOX_DESIGN.md`) is
specified for the GPU host. The verifier **fails closed** — every malformed/adversarial source scores 0, never crashes
(proven by Hypothesis fuzz tests).

### Reward (exact)
```
reward = 0                                   if not correct (allclose on fresh inputs) or any anti-hack cap
       = CORRECT_FLOOR + log_speedup_credit  otherwise   (+ optional profiling-ratio bottleneck term)
```
Kevin-style correctness floor keeps the gradient dense for weak models; log-scaling prevents outlier inflation
(Dr.Kernel arXiv:2602.05885, Kevin arXiv:2507.11948).

---

## 📊 Evaluation rigor (no hand-waving)

The generalization claim is backed by a **statistically powered, paired** protocol (`src/protean/eval_protocol.py`):

- **Unit = held-out task** `(op, shape)` — not a rollout (no pseudoreplication).
- **Scale:** 5 ops × 40 continuous off-grid shapes = **200 paired tasks** (>170 distinct sizes).
- **Paired** base-vs-trained on the *same* tasks → per-task delta `Δ(t)`.
- **Magnitude + CI:** hierarchical bootstrap (resample ops → shapes); **BCa** option (Efron 1987).
- **Significance:** across-op **sign test** — clustering-immune (`K` ops all-positive → `p = (1/2)^K`).
- **Power:** MDE `d_z = 0.20 @ n=200` vs `1.62 @ n=3` — the old fixed-shape design was underpowered by ~8×.

---

## 🔌 HUD integration

Built as a real **v6 `@env.template` two-yield** environment (the `verilog-template` idiom), not a thin wrapper:

- **Env + reward** via `@env.template` → `EvaluationResult` with `done`, human-readable `content`, and rich subscores.
- **Deploy-clean:** `hud serve protean.env:env`, string-literal template registration (`hud task list --source` lists
  all 6 tasks), no `from __future__ import annotations` deploy-bomb, fail-loud on an empty env.
- **Streaming:** every optimizer trial streams into one HUD job with full step traces (prompt, compile, correctness,
  timing, reward, accept/reject).
- **Integration depth: ~4/5.** Honest remaining work (needs the GPU/platform host): the HUD **training tier**
  (`TrainingClient`, group→GRPO advantages) and a clean grouped remote-eval. See `docs/HUD_INTEGRATION.md`.

---

## ✅ Engineering quality (production-hardened)

| | |
|---|---|
| Tests | **368 passed** (CPU) incl. Hypothesis fuzz tests; runs with **or without** the `hud` extra |
| Types | **mypy: 0 errors** (real fixes, zero blanket ignores) — gated in CI |
| Lint | `ruff` check + format clean — gated in CI |
| Coverage | **73%+** with a ratchet floor — gated in CI |
| Security | `pip-audit`/OSV scan in CI · `SECURITY.md` threat model · windowed import audit hook |
| Reproducibility | SHA-256 + schema-versioned frozen task manifest with drift detection; `uv.lock` pinned |
| CI jobs | lockfile · test (py3.11/3.12) · lint · **typecheck** · **coverage** · **audit** |

---

## 🔬 Grounded in 2026 research
Kevin (2507.11948) · Dr.Kernel (2602.05885) · daVinci-kernel (2606.16497) · KernelFoundry (2603.12440) ·
KernelBand (2511.18868) · robust-kbench (2509.14279) · SOL-ExecBench (2603.19173) · paired-bootstrap (2511.19794).
Full ranked roadmap + citations: `docs/IMPROVEMENT_RESEARCH.md`.

---

## 📁 Repo layout
```
src/protean/
  env.py            # HUD two-yield environment (deploy-clean)
  grader.py         # grade_source() — the single verifier authority
  bench_core.py     # CUDA-event timing, fresh-input correctness, launch counter
  anti_hack.py      # AST bans + windowed runtime audit hook
  rewards.py        # reward math (log-speedup + profiling-ratio)
  splits.py         # train vs off-grid held-out shapes (the moat)
  eval_protocol.py  # paired delta · hierarchical/BCa bootstrap · sign test · power
  optimizer.py      # deterministic ▸ UCB bandit ▸ Fireworks ▸ 1M-controller search
  hud_stream.py     # stream trials into one HUD job (async-safe)
  config.py         # validated settings/secrets
  model/            # policy, fireworks_policy, tiny_policy (1M controller)
  tasks/<op>/       # prompt.md + hidden donotaccess/{grade,reference}.py
train/              # GRPO stretch path (calibrate, grpo_loop, callbacks)
scripts/            # smoke_verifier · run_optimizer · run_powered_eval · run_hud_*
docs/               # TECHNICAL_SPEC · HUD_INTEGRATION(_AUDIT) · SANDBOX_DESIGN · PRODUCTION_READINESS · GAP_ANALYSIS
```

---

## 🚀 Quickstart

```bash
# Python 3.11/3.12 (project caps <3.13)
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"          # CPU dev (no GPU needed)
pytest -q                          # 368 passed
python scripts/check_redteam.py    # anti-hack fails closed

# GPU host (NVIDIA H100/B200) — the real run
pip install -e ".[gpu,test,hud]"
python scripts/run_optimizer.py --all-ops --max-rounds 20   # real speedups
python scripts/run_powered_eval.py --ops elementwise_add_relu rmsnorm softmax_rows --out demo/powered-eval.json

# HUD platform
hud set HUD_API_KEY=...            # YC-RL-HACKATHON
hud serve protean.env:env          # serve the env
hud task list --source src/protean/env.py   # all 6 tasks register
```

**Environment:** secrets only per-feature — `HUD_API_KEY` (HUD), `FIREWORKS_API_KEY` (Fireworks edit policy);
`TRITON_CACHE_DIR` for GPU. Nothing is required to run the tests. See `.env.example`.

---

## 🏆 Track & sponsors
**ML Research (GPU)** — the cleanest non-gameable verifier in the field (measured speedup) on the operator's edge.
Load-bearing sponsors: **HUD** (env/eval/streaming), **Modal** (GPU compute plane), **Fireworks** (edit policy).

## 🗺️ Honest status & roadmap
- ✅ Verifier, optimizer, HUD env, powered eval, security, CI — done & verified (CPU) + real **4.62×** on B200.
- 🔜 Needs the GPU/platform host: HUD **training tier** (GRPO on HUD), grouped remote-eval completion, the
  full nsjail sandbox, and the powered-eval money slide regenerated from B200 grades.

## 📄 License
MIT — see `LICENSE`.
