# Protean

**An RL environment that trains models to write GPU kernels that stay fast on shapes they've never seen.**

> *Kernels that are fast on any shape.*

### 📋 Start here
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** — the build bible (source of truth)
- **[docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)** — mathematical & technical deep-dive (reward math, GRPO objective, timing protocol, generalization statistics, anti-hack soundness)
- **[docs/BUILD_CHECKLIST.md](docs/BUILD_CHECKLIST.md)** — condensed Friday→Sunday actionable checklist
- **[docs/IMPLEMENTATION_components.md](docs/IMPLEMENTATION_components.md)** — per-component specs + devil's-advocate debate

**Build status: 📐 PLAN COMPLETE / ⛔ NOT YET BUILT.** Source files under `src/` and `train/` are stubs with
pseudocode and TODOs extracted from the plan. Implement against `docs/IMPLEMENTATION_PLAN.md` + `docs/TECHNICAL_SPEC.md`.

Protean is a reusable [HUD](https://www.hud.ai) reinforcement-learning environment for the
**HUD × YC Frontier RL Environments Hackathon**. A small model (Qwen2.5-Coder-7B) learns multi-turn
Triton GPU-kernel optimization. Each rollout is graded by a hidden, root-owned verifier:

```
reward = correctness (allclose on fresh random inputs) × measured wall-clock speedup vs PyTorch eager
         — evaluated on HELD-OUT, CONTINUOUS tensor shapes disjoint-by-construction from training shapes
```

## Why this is different (the moat)
Every 2026 kernel-RL system (Kevin, Dr.Kernel, DRTriton, daVinci-kernel, TritonForge) trains **and tests on the
same shape distribution**. KernelBench's own roadmap (#74) lists a shape-sweep generalization verifier as
*unshipped*. Protean ships exactly that: it tests on **off-grid continuous shapes the model has never seen**, so a
high reward proves *reasoning over memorization*. That non-gameable, generalization-graded verifier — packaged as a
forkable HUD env labs can train on — is the contribution.

## The one thing that wins
A **train-shape vs held-out-shape reward curve** (the money slide) showing a real base-vs-trained gap on shapes
absent from any published eval, plus a red-team proof that the verifier can't be gamed — delivered by 1 PM Sunday.

## Architecture
```
            ┌─────────────────────────────────────────────┐
   Qwen2.5  │  HUD two-yield env (env.py)                  │
   -Coder   │   yield prompt ─▶ agent edits solution.py    │
    -7B ───▶│   (multi-turn, /workdir, self-serve bench)   │
            │   ◀─ yield reward (0..1)                      │
            └───────────────┬─────────────────────────────┘
                            │ grader.py → hidden donotaccess/grade.py (root:700)
                            ▼
            ┌─────────────────────────────────────────────┐
   Modal    │  rewards.py  = THE single reward authority   │
   H100     │  bench_core.py  CUDA-event timing            │
            │  anti_hack.py + launch_probe.py (4-layer)    │
            │  splits.py / sampler.py  held-out shapes     │
            └───────────────┬─────────────────────────────┘
                            │ reward signal
                            ▼
            ┌─────────────────────────────────────────────┐
            │  train/grpo_loop.py  (trl GRPO + vLLM)       │
            │  calibrate.py gate → kick by Sun 8 AM        │
            └─────────────────────────────────────────────┘
```

## Repo layout
```
protean/
├── README.md
├── pyproject.toml            # deps: hud-python[agents], torch, triton, trl, vllm
├── Dockerfile.hud            # CUDA+torch+triton, uid-1000 agent, TRITON_CACHE_DIR baked
├── docs/
│   ├── IMPLEMENTATION_PLAN.md          # ⭐ the build bible (full)
│   ├── IMPLEMENTATION_components.md     # per-component specs + devil's-advocate debate
│   ├── BUILD_CHECKLIST.md              # condensed Friday→Sunday actionable checklist
│   ├── papers/davinci-kernel-2606.16497.pdf
│   └── strategy/                       # how we got here (audit, debate, track intel, finalist specs)
├── src/protean/
│   ├── env.py                # HUD two-yield env + uid-wall workspace
│   ├── rewards.py            # THE reward formula (loaded by grader AND trainer)
│   ├── grader.py             # public evaluate_kernel(); loads hidden grade.py
│   ├── bench_core.py         # CUDA-event timing, fresh inputs, median-of-M
│   ├── anti_hack.py          # 4-layer guard (AST ban, launch counter, dtype, floor)
│   ├── launch_probe.py       # Triton-launch instrumentation (+profiler fallback)
│   ├── subprocess_runner.py  # fail-closed isolated execution
│   ├── splits.py             # frozen train/held-out shape-split invariant
│   ├── sampler.py            # deterministic continuous-shape sampler
│   ├── manifest.py           # freeze taskset to versioned JSONL
│   ├── task_catalog.py       # the ~5 fixed ops
│   ├── scenario_helpers.py   # per-episode workspace staging
│   ├── tasks.py              # taskset hud eval / trainer consume
│   └── tasks/<op>/           # prompt.md + donotaccess/{grade.py,reference.py}
├── train/                    # grpo_loop.py, reward.py, calibrate.py, callbacks.py
└── scripts/check_calibration.py   # pre-GRPO go/no-go gate
```

## Quickstart
```bash
# 0. Friday prep (OFF the 24h clock — mandatory)
uv tool install hud-python --python 3.12
hud set HUD_API_KEY=...          # YC-RL-HACKATHON
# build the Modal H100 image (bakes TRITON_CACHE_DIR + warmup JIT), run SFT warm-start

# 1. Dev loop on one op
hud eval src/protean/tasks.py claude     # smoke-test the env against a model
python scripts/check_calibration.py      # base model: ≥5% compilable, ≥2% allclose, reward variance, 20–50% band

# 2. Overnight (kick by Sun 8 AM)
python train/grpo_loop.py --op all --group 8 --abort-step 150

# 3. Sunday: held-out eval → money slide
python train/calibrate.py --eval held_out --base ckpt0 --trained ckptN
```

## daVinci-kernel integration (honest scope)
Integrates **two** techniques from [daVinci-kernel (arXiv 2606.16497)](https://arxiv.org/abs/2606.16497):
the **profiling-ratio reward** (ported as an additive bonus, not the multiplicative Eq1 gate) and a
**static skill-injection prefix** (load-bearing only because the SFT warm-start includes skill-conditioned
examples). The full 3-agent co-evolution (Selection/Summary agents, per-agent LOO advantages, MRS/PRS) is
**explicitly deferred** — we ship `trl` GRPO and say so. See `docs/IMPLEMENTATION_PLAN.md` §5.

## Status
Pre-build. Source files are **stubs with pseudocode** extracted from the plan — implement the TODOs.
Read `docs/IMPLEMENTATION_PLAN.md` first; it is the source of truth.

## License
MIT
