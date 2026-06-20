# Phase 13 — Final Survivor Specs

Three finalists survive all attacks. **Build #1, stage #2 (shares ~80% of the codebase), keep #3 as insurance/Most-Creative.**

---

## 🥇 #1 — Protean  *(primary · 1st-place play · win-prob 84.8%)*

- **One-line pitch:** *An RL gym where a small model learns to write GPU kernels that beat PyTorch — graded by a stopwatch you can't fake, on tensor shapes it has never seen.*
- **Track:** ML Research (GPU). **Secondary special:** Most Viral (live 2× speedup).
- **Capability:** multi-turn write + optimize a correct, fast Triton kernel for a target op / op-graph.
- **Why RL (not prompting):** the reward (measured speedup) is sparse, verifiable, and only reachable by iterative
  trial against a real GPU — exactly what GRPO over many rollouts optimizes; Kevin-32B proved the curve exists.
- **State:** SSH+GPU sandbox = current kernel source + compile errors + correctness diff + last latency.
- **Actions:** emit/revise Triton kernel (≤K turns); each turn re-compiles, re-checks, re-benchmarks.
- **Reward:** `r = 0 if not allclose(out, eager, N random inputs) else min(speedup_vs_eager / p, 1.0)`, on a **held-out
  continuous shape range**; speedup = median of M CUDA-event-timed runs (warmup + cache flush).
- **Non-gameability proof:** torch-in-kernel → speedup ≤1 ⇒ zero reward; randomized inputs + held-out shapes defeat
  input/shape overfit; CUDA-events + post-timing functional re-check defeat timer gaming. (3 exploit rounds closed.)
- **Scaling:** op-graph sampler composes random PyTorch ops × shapes → unbounded fresh tasks beyond KernelBench's 250;
  curriculum L1→L2→L3.
- **HUD two-yield:**
  ```python
  @env.template()
  async def kernel_task(op_graph, train_shapes, test_shapes):
      kernel = yield build_prompt(op_graph, train_shapes)          # yield 1 → prompt
      ok, spd = await sandbox.compile_check_bench(kernel, test_shapes)
      yield 0.0 if not ok else min(spd / P_TARGET, 1.0)            # yield 2 → reward 0–1
  ```
- **Training plan / Sunday-8AM run:** Sat 12:30 fork `hud init blank` + GPU/ssh capability + harness on 5 tasks →
  **first signal ~3 PM**; op-graph sampler + held-out split by evening; GRPO wiring + smoke train by ~11 PM; **kick
  overnight GRPO by 8 AM Sun** (Qwen2.5-Coder-7B, HUD `Taskset.run(group=16)` → group-relative advantages, curriculum
  L1→L2); base-vs-trained held-out eval → **delta curve by 1 PM**.
- **Budget:** Modal $250 (GPU bench + train) · HUD $200 (env/rollouts) · Fireworks $30 or local — fits at ≤7B.
- **Sponsors (load-bearing):** HUD (env/reward/GRPO) · **Modal** (non-core: parallel compile+bench plane).
- **Demo / WOW:** live hero kernel 4.2→2.0 ms; money slide fast_p **18%→40%** on held-out. Fallbacks: pre-recorded run,
  replay, video.
- **Startup path:** the kernel-optimization RL environment → sell to labs (HUD Series A thesis: envs sold to labs).
- **Risks → rebuttals:** *"KernelBench-in-a-box / Kevin did it"* → the **env** is the contribution (infinite op-graph
  task-gen + a generalization verifier KernelBench lacks, reusable by labs). *"No real RFT overnight"* → weak 7B where
  gains are fast + Kevin precedent + N/CI + pre-recorded fallback.
- **Win prob:** **84.8%.**  Scores (prior loop rubric): verifier 6/6 · feasibility 4/4 · judgefit 5/5 · wow 3/3 · edge 3/3.

---

## 🥈 #2 — TESTBENCH-FORGE v2  *(novelty / Most-Creative + 2nd shot · win-prob 81.6%)*

- **One-line pitch:** *An RL gym that trains agents to write the test suite that catches the most bugs — rewarded by how many hidden, freshly-injected mutants their tests kill.*
- **Track:** Agentic Collaboration (pytest variant) — or Chip Design (Verilog-testbench variant); **pick the software
  variant to present**, mention hardware portability as the moat.
- **Capability:** iteratively write a test suite for a given module that maximizes bug-catching.
- **Why RL:** "write tests that kill unseen bugs" has no supervised target; the agent must learn coverage strategy from
  the mutation-kill signal — a dense, verifiable reward ideal for GRPO.
- **State:** module under test + reference impl + current test suite + coverage/kill feedback.
- **Actions:** add/edit tests over turns; harness runs them against reference + a hidden mutant pool.
- **Reward:** `(#mutants_killed / #mutants) × [suite passes on N equivalent reference variants]`; mutants **hidden +
  freshly generated** each episode; over-specification penalty (must pass refactor-equivalent references).
- **Non-gameability proof:** assert-False → fails reference gate ⇒ 0; mutants never visible ⇒ can't target them;
  over-spec penalty kills brittle snapshot tests. You cannot fake killing a bug you've never seen. (3 rounds closed.)
- **Scaling:** mutation operators auto-generate infinite buggy variants × infinite modules.
- **HUD two-yield:**
  ```python
  @env.template()
  async def testbench_task(module, reference, mutant_seed):
      suite = yield build_prompt(module, reference)               # yield 1 → prompt
      passes = run(suite, reference_variants)                     # gate
      kills  = sum(run_fails(suite, m) for m in mutants(mutant_seed))
      yield (kills/len_mutants) if passes else 0.0                # yield 2 → reward
  ```
- **Training plan:** shares the sandbox + verify harness with #1; the mutation engine is the only new piece (~3h).
  Rides the **same overnight run**. Delta 30%→58% (fast eval = many episodes).
- **Budget:** marginal on top of #1 (same infra). Daytona $100 optional for isolated test sandboxes.
- **Sponsors:** HUD (env) · Daytona (isolated execution) · optionally Fireworks (GRPO).
- **Demo / WOW:** bug-kill meter 30%→58%; "this bug slipped past base, trained caught it" diff. Fallbacks: pre-recorded.
- **Startup path:** verification-as-a-service — produce the test suites labs need to trust their coding agents.
- **Risks → rebuttals:** *"is it an eval?"* → it's an env: state(suite)/actions(add tests)/reward(kill rate)/dist-shift
  (fresh mutants+new modules)/visible overnight delta. *"unfocused dual-track"* → present software only.
- **Win prob:** **81.6%.**

---

## 🥉 #3 — SILICON-FORGE v2  *(insurance / Most-Creative · win-prob 77.0%)*

- **One-line pitch:** *An RL gym where an agent learns to write working silicon — Verilog from a spec, graded by a real EDA flow it can't bluff.*
- **Track:** Chip Design. **Strategic value:** lowest competition + most frontier-resistant.
- **Capability:** write RTL from a parameterized spec over SSH.
- **Why RL:** PPA/functional reward is verifiable and only reachable by iterative synthesis trials.
- **State:** spec + current RTL + sim results + gate-count/timing feedback.
- **Actions:** edit RTL over turns; harness runs hidden testbench + fast synth (Yosys) proxy.
- **Reward:** `testbench_pass(hidden, coverage-gated) × normalized(gate_count vs target)`; full PPA offline.
- **Non-gameability proof:** hidden randomized testbench vectors + coverage gate defeat output-hardcoding; functional
  gate precedes PPA so trivial logic fails. EDA metrics are unarguable. (3 rounds closed.)
- **Scaling:** spec templates parameterized (bit-widths, op sets) → infinite tasks.
- **Training plan / risk:** **delta is the risk** — synth is slow ⇒ fewer episodes/hr. Mitigation: fast functional +
  gate-count proxy for the live curve, full PPA offline. **Only build if #1 shows signal by Sat evening**; else present
  as a slide + pre-synthesized results.
- **Budget:** Modal/GCP for synth runners; HUD env.
- **Sponsors:** HUD · Modal/GCP (synth compute).
- **Demo / WOW:** `timing: MET` + gate-count chart flips green; "an AI learned to write working silicon."
- **Startup path:** the on-ramp to AI chip design; EDA incumbents + silicon startups want it.
- **Risks → rebuttals:** *"delta won't show by 1 PM"* → fast proxy verifier + pre-synthesized fallback; honest "directional
  signal" framing. *"too niche for audience"* → that niche IS the moat (frontier-resistance + Most Creative).
- **Win prob:** **77.0%.**

---

## STOP — conditions satisfied
✅ 3 finalists survive all attacks · ✅ no unpatched verifier exploit (3 forensic rounds each) · ✅ visible training delta
(A/B certain, C proxy-mitigated) · ✅ 4-min demo each with no live dependency · ✅ startup story each · ✅ <30s judge
comprehension each. **Loop converged.**

## Build recommendation (solo, 24h)
Build **Protean**. It and **TESTBENCH-FORGE v2** share ~80% of one codebase (sandbox + verify harness), so
flip on the mutation engine to stage B for the **Most Creative / Most Viral** category — two shots, one overnight run.
Hold **SILICON-FORGE v2** as a slide-only insurance play unless A is green by Saturday night.

**The single thing that wins or loses it:** a real base-vs-trained delta on a *held-out* split by 1 PM Sunday. Start the
harness Saturday 12:30 so first signal lands by 3 PM and the overnight GRPO run is kicked by the 8 AM Sunday deadline.
