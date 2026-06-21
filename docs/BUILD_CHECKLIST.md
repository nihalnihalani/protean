# Protean — Build Checklist (condensed from IMPLEMENTATION_PLAN.md §6)

## The five non-negotiable laws
1. **ONE reward file, ONE path, ZERO copies.** `rewards.py` lives only at the baked path; grader AND trainer
   `importlib`-load it from there. Divergence is structurally impossible (kills ghost-signal training).
2. **Friday-night prep is OFF the 24h clock and MANDATORY** (image + launch-counter smoke test + SFT checkpoint).
3. **Single-turn GRPO is the only committed path** (multi-turn = stretch).
4. **Held-out continuous-shape split is the frozen moat invariant** — assert disjoint at import; never re-sample at runtime.
5. **Reframe every delta as directional** (Kevin/Dr.Kernel analog), never a hard promise.

## G0 — FRIDAY EVENING (off the clock)
- [ ] `uv tool install hud-python`; `hud set HUD_API_KEY` (YC-RL-HACKATHON)
- [ ] Verify Modal GPU = **H100/A100** (NOT AMD MI300X — Triton tooling crashes)
- [ ] Build Modal image: bake `TRITON_CACHE_DIR=/triton-cache` + a real warmup JIT compile
- [ ] Smoke-test `launch_probe.py` incl. the cached-path case (auto-fallback to profiler if it fails)
- [ ] Check KernelGYM license (MIT/Apache?) → reuse its compile/bench/anti-hack harness
- [ ] SFT warm-start: filter `hkust-nlp/drkernel-coldstart-8k` to `final_speedup≥1.2` (~200 rows), LoRA 1–2 epochs
      → checkpoint on Volume. (Gate behind a schema probe; if columns absent, skip SFT, bootstrap via L1 curriculum.)

## Saturday
- **12:30–13:30 Block 0** — setup, deconflict, confirm fork = hud-blank (ml-template is torchtitan — DO NOT fork it)
- **13:30–17:30 Block 1** — `rewards.py` + env skeleton; ONE op compiles→allclose→times→returns reward on H100
- **17:30–21:00 Block 2** — Modal harness + 4-layer anti-hack + calibration (parallel). Red-team: passthrough /
  never-launched / bf16-downcast kernels must score ~0
- **21:00–21:30 Block 3** — GRPO wiring + run `scripts/check_calibration.py` (the go/no-go gate)
- **21:30 Block 4** — **KICK OVERNIGHT GRPO** (trl + vLLM colocate, group=8, L1 curriculum, step-150 abort rule)

## Sunday
- **overnight Block 5** — monitor; flat at step 150 → switch to pre-recorded curve. Pre-record by ~6 AM regardless.
- **08:00–11:00 Block 6** — held-out eval (base vs trained) → money slide (train-shape vs held-out-shape curve, error bars)
- **11:00–13:00 Block 7** — slides + buffer. Submit by **1 PM**.

## The money slide
Train-shape reward vs **held-out-shape** reward, base vs trained, with error bars on BOTH allclose-rate and
speedup-reward + a red-team proof screenshot + a pre-warmed hero kernel on a held-out shape + a forkable HUD env.

## daVinci KEEP / CUT (total ~1.5h, on the critical path)
| Component | Decision |
|---|---|
| Policy agent | KEEP-FULL |
| PR reward (Eq1) | LITE — additive `+0.2·clip(pr,0,1)` bonus, not the multiplicative gate |
| SFT cold-start | KEEP (Friday, off-clock) |
| Skill library/injection | LITE/contingent on SFT skill-conditioned examples; else CUT |
| Skill Selection + Summary agents, per-agent LOO (Eq7-9), MRS, PRS | CUT — ship `trl` GRPO and say so |
