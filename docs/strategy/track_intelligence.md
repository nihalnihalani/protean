# Phase 0 — HUD Track Intelligence

> Source: live fetch of https://www.hud.ai/hackathon (June 20–21, 2026 · Y Combinator, SF · 24h).
> Scoring axes 1–5: **Delta** (can an RL run improve in 24h?) · **WOW** (audience gets it instantly?) ·
> **Startup** (can it be a YC company?) · **Operator Edge** (GPU/kernels/agents/retrieval/memory/multi-agent).

## Ground-truth event facts
- **Critical deadline:** Sun **8:00 AM** kick off first training run. **1:00 PM** submissions due. **2:30 PM** top-10 present.
- **Task-tuning guideline (official):** ~10 evals/task, target **20–50% reward with variance** (avoid all-0/all-1).
- **Training routes:** HUD SDK on-policy (roll out taskset → train on trajectories) OR Fireworks GRPO (high-parallel rollouts).
- **Setup:** `pip install hud-python` → `hud set HUD_API_KEY` → `hud init my-env` → `hud eval tasks.py claude` → `hud deploy`.
- **Prizes:** 1st = **YC F26 guaranteed interview** + $10k HUD + RTX 5090 / robot dog + **$10k Modal** + $5k Fireworks.
  Special categories (each $2.5k + AirPods): **Most Utopian · Most Creative · Best Design · Most Viral**.
- **Credits/hacker:** HUD $200 · Modal $250 · Daytona $100 · Exa $50 · Fireworks $30 · MiniMax $30 · GCP $25 · SixtyFour auto.
- **Validating fact (last30days, 2026-06-18, 240 likes):** HUD raised a **Series A** — *"platform for building high-quality
  post-training datasets; 50+ businesses build RL environments and sell them to AI labs."* → the **env IS the company** thesis is real.

---

## Starter repositories (exact fork targets, from hud.ai/hackathon#tracks)
| Track | Template | GitHub repo | Notes |
|---|---|---|---|
| ML Research | ML Research & Training | `hud-evals/ml-template` | **GPU** — Protean fork target |
| ML Research | ML Triage | `hud-evals/ml-triage-tasks` | CPU |
| Chip Design | Verilog | `hud-evals/verilog-template` | SSH + hidden EDA — SILICON-FORGE fork target |
| Robotics | Robotics | `hud-evals/robot-template` | MuJoCo + LIBERO |
| Robotics | Worldsim Robotics | `hud-evals/worldsim-template` | Newton physics · partner AntimLabs |
| Gaming | Video Game Bench | `hud-evals/videogamebench-template` | Game Boy · partner AntimLabs |
| Gaming | ARC-AGI-3 | `hud-evals/arc-agi-3` | rule-learning |
| Agentic | Coding | `hud-evals/coding-template` | hidden pytest — TESTBENCH-FORGE fork target |
| Agentic | Deep Research | `hud-evals/hud-deepresearch` | Exa + SixtyFour |
| Agentic | Browser | `hud-evals/hud-browser` | Chromium CDP/RFB |
| Agentic | Computer Use | `hud-evals/cua-template` | Linux desktop RFB/VNC |
| Auto-Business | Autonomous Business | `hud-evals/autonomous-businesses-template` | clinic triage |
| Auto-Business | GDPval | `hud-evals/gdpval-template` | business evals |
| Bonus | Blank | `hud-evals/hud-blank` | minimal custom env |

Note: `hud-evals/ml-template` already ships GPU scaffolding → faster start than `hud-blank` for Protean.

## Track-by-track analysis

### 1. ML Research  ⭐ TOP PICK
- **Starters:** ML Research & Training (GPU template) · ML Triage (CPU template).
- **Hidden verifier style:** objective + machine-read — *correctness* (unit tests / reference output) **×** *measured wall-clock
  speedup or accuracy*. Closest thing to a perfect RLVR verifier in the whole event (a stopwatch can't be argued with).
- **Likely winning difficulty:** KernelBench L1→L2 (single-op → fused multi-op). Frontier models beat PyTorch in <20% of cases →
  lands naturally in the 20–50% band.
- **Competition density:** MEDIUM-HIGH (popular, but a real barrier — most can't write fast kernels).
- **Sponsor interest:** Modal ($250, biggest credit; GPU plane) + GCP. Strong.
- **Scores:** Delta **5** · WOW **4** · Startup **5** · Operator **5** → **19/20**.

### 2. Chip Design  ⭐ STRATEGIC WHITE-SPACE
- **Starter:** Verilog/SystemVerilog task over SSH, graded by hidden EDA flows.
- **Hidden verifier style:** EDA toolchain — functional sim (testbench pass) + **PPA** (power/perf/area) from synthesis. Extremely
  clean, fully objective, impossible to bluff.
- **Likely winning difficulty:** small RTL modules (ALU, FIFO, FSM) with timing/area targets.
- **Competition density:** **LOW** — almost nobody at an AI hackathon writes Verilog. Biggest moat.
- **Sponsor interest:** indirect (compute), but novelty earns Most Creative.
- **Frontier-resistance:** HIGH — even a top lab researcher can't trivially win this in a weekend without EDA depth.
- **Scores:** Delta **3** (synth is slow, delta harder to move overnight) · WOW **3** · Startup **4** · Operator **3** → **13/20**
  (but strategic value ↑ from low competition + frontier-resistance).

### 3. Robotics
- **Starters:** Robotics (MuJoCo + LIBERO) · Worldsim Robotics (AntimLabs, Newton physics).
- **Verifier:** sim task success (object placed / goal reached) — clean.
- **Difficulty / competition:** medium / medium.
- **Scores:** Delta **3** · WOW **5** (robots = visceral) · Startup **3** · Operator **2** → **13/20**.

### 4. Gaming & Worldsims
- **Starters:** Video Game Bench (Game Boy, AntimLabs) · ARC-AGI-3 (rule-learning games).
- **Verifier:** game score / level completion — clean.
- **Competition:** HIGH (ARC-AGI is a magnet; judges have seen it).
- **Scores:** Delta **3** · WOW **5** · Startup **2** · Operator **2** → **12/20**.

### 5. Agentic Collaboration
- **Starters:** Coding (pytest) · Deep Research (Exa + SixtyFour) · Browser (2048/todo) · Computer Use (Linux desktop RFB).
- **Verifier:** pytest (clean) / deep-research (fuzzy) / game state (clean).
- **Competition:** **VERY HIGH** — the default track; most teams land here. Overcrowded.
- **Sponsor interest:** highest (Exa/SixtyFour/Daytona/Anthropic).
- **Scores:** Delta **4** · WOW **3** · Startup **4** · Operator **4** → **15/20** (discount heavily for competition).

### 6. Autonomous Business
- **Starters:** Autonomous Business (clinic ticket triage) · GDPval (real-world business evals).
- **Verifier:** business value — fuzzier (GDPval graders, partial LLM-judge). Weakest verifier cleanliness.
- **Competition:** medium. **Startup appeal: highest** (investor-legible).
- **Scores:** Delta **3** · WOW **4** · Startup **5** · Operator **3** → **15/20**.

---

## Track ranking & lock
| Rank | Track | Total | Note |
|---|---|---|---|
| 1 | **ML Research (GPU)** | 19 | operator edge + cleanest verifier + fast delta. LOCK PRIMARY. |
| 2 | Agentic Collaboration | 15 | strong but overcrowded → only enter with a non-obvious env. |
| 2 | Autonomous Business | 15 | best startup story, fuzziest verifier. |
| 4 | **Chip Design** | 13 | LOCK SECONDARY (white-space + frontier-resistant insurance). |
| 4 | Robotics | 13 | great WOW, weak operator fit. |
| 6 | Gaming & Worldsims | 12 | crowded, weak startup. |

**LOCKED:** primary = **ML Research (GPU / kernels)**; secondary/insurance = **Chip Design (Verilog)**.
**Special-category target:** **Most Creative** (a novel verifier) and/or **Most Viral** (a legible live speedup).

## The frontier test
*"What wins if OpenAI/Anthropic/DeepMind/Cursor each send a top researcher tomorrow?"* → A polished coding/browser
demo loses to them. What survives is an env where the **verifier + domain depth** are the moat: **kernel speedup**
(measured, un-fakeable) and **chip-design PPA** (EDA-graded). Both chosen tracks pass the frontier test; the
crowded Agentic tracks fail it.
