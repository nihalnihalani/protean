# Protean — Component Specs & Debate Appendix

> Structured output from the kernelforge-impl-plan workflow (companion to IMPLEMENTATION_PLAN.md).


## Round 1 — Component Designs


### HUD environment skeleton for Protean  (~9h)
**Goal:** A reusable HUD RL environment that serves a multi-turn Triton-kernel-optimization task to Qwen2.5-Coder-7B and grades each rollout with a non-gameable, hidden, root-owned verifier whose reward = correctness (allclose on fresh random inputs) x measured wall-clock speedup vs PyTorch eager, evaluated on HELD-OUT continuous shapes. The skeleton must (1) clone cleanly from the verilog-template scaffold (two-yield @env.template + uid-wall Workspace + hidden grader + check_calibration), (2) expose ONE parameterized template that mints train-shape and held-out-shape task variants from ~5 fixed ops via a shape sampler, (3) run on a Modal H100 with TRITON_CACHE_DIR baked, and (4) carry the 4-layer anti-hack guard plus daVinci profiling-ratio gate so the verifier is the moat. This component is ONLY the env skeleton + task plumbing + grader interface; the GRPO trainer and the bench/exec subprocess internals are sibling components it calls into.

**Files:**
- `protean/env.py` — Top-level HUD Environment. Defines `env = Environment(name="protean-v2")` (string literal, NO `from __future__ import annotations`), a uid-wall `_KernelWorkspace(Workspace)` that setpriv-demotes the agent shell to uid 1000, @env.initialize/@env.shutdown that start/stop the workspace and add the shell capability, and the single `@env.template(id="kernel_opt")` two-yield async generator. Mirrors verilog-template/env.py almost line for line.
- `protean/task_catalog.py` — OpSpec dataclass (op_name, eager_fn ref, dtype, ref-impl module path, allclose rtol/atol, P_TARGET, speedup_floor) for the ~5 fixed ops (elementwise_add_relu, rmsnorm, softmax, matmul_bias_gelu, layernorm). Plus the SHAPE SPLIT constants: TRAIN_SHAPES and HELD_OUT_SHAPES samplers. The frozen registry the template binds against.
- `protean/shape_sampler.py` — `sample_shape(op_name, split, seed)` — draws a concrete shape from a CONTINUOUS range per op, with disjoint train vs held-out bands (e.g. train M~U{256..2048 step32}, test M~U{384,768,1536} or a disjoint continuous interval). Returns a fully-resolved shape tuple + the RNG seed used for input generation. This is the generalization moat.
- `protean/tasks.py` — Re-exports `env` + the template, then enumerates the taskset: for each op x split it calls the template to mint a Task, sets .slug and .columns={op,split,shape_class}. This is the module `hud eval tasks.py` serves and the trainer's Taskset.run() consumes.
- `protean/scenario_helpers.py` — Workspace reset/staging: `setup_task(op_name, split, seed, validate_mode)` writes the agent-facing prompt + a writable /workdir/solution.py stub + /workdir/bench.py (the agent's self-serve compile/allclose/latency probe) into the workspace each episode, and NEVER copies the hidden ref/grader in. Resolves WORKSPACE_ROOT (/workdir on image, per-pid tmp locally) and hidden_dir(). Mirrors verilog scenario_helpers.
- `protean/grader.py` — Public `evaluate_kernel(op_name, split, seed) -> EvaluationResult`. Imports the HIDDEN grade module from the root:700 donotaccess dir, runs it in a subprocess, builds an EvaluationResult by hand (NOT via combine()) with the hard-cap-penalty reconciliation subscore so a failed anti-hack gate forces reward to 0. Copied structurally from verilog-template/grader.py.
- `protean/tasks/<op>/donotaccess/grade.py` — The HIDDEN, root-owned verifier (the moat). Sibling-component owned but the skeleton defines its `grade(workdir, op_name, split, seed, hidden_root) -> dict` contract. Imports the reference eager impl, runs the 4-layer anti-hack guard + daVinci profiling-ratio gate in an isolated subprocess, returns {reward, subscores, hard_caps}. Skeleton ships a stub that calls into the sibling exec/bench component.
- `protean/tasks/<op>/donotaccess/reference.py` — Hidden PyTorch-eager reference for the op (the allclose oracle + the eager timing baseline). One per op. Never visible to the agent.
- `protean/tasks/<op>/prompt.md` — Agent-facing task description: the op semantics, the EXACT input signature/dtype/device, the requirement to write a @triton.jit kernel in /workdir/solution.py exposing `solution(*tensors)`, that it will be tested on a withheld shape, and how to use bench.py to self-check. Shape is NOT pinned in the prompt (held-out).
- `protean/scripts/check_calibration.py` — Pre-GRPO go/no-go gate (copied pattern from verilog-template/scripts). Runs the base Qwen2.5-Coder-7B Nx per op via the real grader, asserts >=5% compilable, >=2% allclose, nonzero group reward variance, median group reward in (0,1). Exits nonzero to BLOCK the 8AM kick if calibration fails. Deleted from the agent workspace by setup_task.
- `protean/modal_app.py` — Modal H100 image + runner. Image bakes TRITON_CACHE_DIR=/triton-cache + a warmup pass that JIT-compiles a trivial kernel at build time to kill the 30-120s cold start. Lifts the @app.function(gpu="H100", secrets=[hud-keys]) decorator + run_agent spawn pattern from ml-template/modal_runner.py. Serves the env and runs rollouts/calibration on-GPU.
- `protean/pyproject.toml` — Deps + hatch wheel include list. requires-python >=3.11,<3.13; deps: hud-python[agents]>=0.6.5, torch, triton, pydantic, numpy. only-include env.py/tasks.py/task_catalog.py/shape_sampler.py/grader.py/scenario_helpers.py/tasks.
- `protean/Dockerfile.hud` — CUDA+torch+triton base, creates uid-1000 `agent` user, copies repo to /mcp_server, bakes /triton-cache (chmod 777) + ENV TRITON_CACHE_DIR, runs the warmup JIT compile. Hidden tasks/<op>/donotaccess baked as root:700.

**APIs:** env.py:: env = Environment(name="protean-v2")  # string literal required; hud deploy static-parses the name; env.py:: class _KernelWorkspace(Workspace): def shell_argv(self, command=None, *, cwd=None, env=None) -> list[str]  # setpriv uid-wall, copied from verilog _AgentWorkspace; env.py:: @env.template(id="kernel_opt") async def kernel_opt(op_name: str, split: str = "train", seed: int = 0, validate_mode: str = None)  # two-yield: yield prompt -> yield EvaluationResult; grader.py:: def evaluate_kernel(op_name: str, split: str, seed: int) -> hud.graders.EvaluationResult; grader.py(reused):: EvaluationResult(reward: float, done: bool, content: str, info: dict, subscores: list[SubScore])  # from hud.graders; grader.py(reused):: SubScore(name: str, weight: float, value: float, metadata: dict|None)  # negative weight encodes hard_cap_penalty; tasks/<op>/donotaccess/grade.py:: def grade(workdir: Path, *, op_name: str, split: str, seed: int, hidden_root: Path) -> dict  # returns {reward, hard_caps, subscores}; the HIDDEN verifier contract; scenario_helpers.py:: def setup_task(op_name: str, split: str, seed: int, validate_mode: str|None = None) -> dict  # resets /workdir, stages solution.py stub + bench.py, never copies donotaccess; scenario_helpers.py:: WORKSPACE_ROOT: Path ; def hidden_dir(op_name: str) -> Path; shape_sampler.py:: def sample_shape(op_name: str, split: str, seed: int) -> tuple[int, ...]  # disjoint train/held_out continuous bands; task_catalog.py:: @dataclass(frozen=True) class OpSpec(op_name, prompt, dtype, rtol, atol, P_TARGET, speedup_floor, train_band, held_out_band) ; OPS_BY_NAME: dict[str, OpSpec]; tasks.py:: tasks: list[Task]  # scanned by `hud eval tasks.py` and Taskset.run(); each has .slug and .columns; bench.py (agent-facing, in /workdir):: solution self-check returning {compiled: bool, allclose: bool, max_abs_err: float, latency_ms: float}  # agent observes this as multi-turn state; modal_app.py:: @app.function(image=image, gpu="H100", timeout=86400, secrets=[modal.Secret.from_name("hud-keys")]) async def run_agent(...)  # lifted from ml-template/modal_runner.py; check_calibration.py:: main() -> exits 1 if base-model calibration band fails (blocks GRPO kick)

**daVinci integration:** daVinci-kernel techniques fold into the skeleton at exactly two load-bearing points, both inside the hidden grade.py, and one is deferred. (1) PROFILING-RATIO (PR) REWARD + GATE — daVinci Eq1 R=C(y)*(1+speedup+PR), PR=T_generated/T_total. We compute PR by measuring (via the Triton-launch-instrumented timing pass) the fraction of total GPU kernel time spent inside the agent's @triton.jit kernels vs the whole op runtime, and (a) gate reward to 0 if PR<PR_THRESHOLD (this is the precise countermeasure to the 'Triton fires on a trivial sub-op while cuBLAS does the heavy matmul' exploit the audit flags as residual-2), and (b) fold (1+PR) into the speedup term so kernels that own more of the runtime score higher. This is THE highest-value daVinci import and costs near-zero throughput. (2) EXECUTION-VERIFIED, FAIL-CLOSED grading + allclose-reseed mirrors daVinci's execution verification and anti-lazy-skill ethos: every reward path re-runs with a second random seed after timing, and any guard trip hard-caps to 0 via the negative-weight reconciliation subscore. (3) DEFERRED (slide-only, NOT on the 24h critical path): daVinci's three-agent skill loop (Skill Selection / Policy / Skill Summary sharing one backbone, JSONL skill library, BM25→LLM rerank, Eq5/6 verification, Eq7-9 per-agent LOO advantages, SFT cold start from Dr.Kernel trajectories). The skeleton is designed so a skill library could later be injected as an extra prompt block in setup_task and the Summary Agent's distilled skills written to a versioned JSONL broadcast to workers — but building three co-evolving agents in 24h solo is out of scope; we cite it as the roadmap and reuse only its reward formula + verification discipline. The reward formula itself blends Kevin (0.3*correct + correct*speedup, gamma=0.4 turn discounting handled by the sibling trainer) with daVinci's PR term, which the audit explicitly endorses as the validated starting point.

```python
# ===== env.py (the load-bearing skeleton) =====
# NO `from __future__ import annotations`  (verilog-template warns: it crashes @env.template typed params)
import os, sys
from hud import Environment
from hud.environment import Workspace
from grader import evaluate_kernel
from scenario_helpers import WORKSPACE_ROOT, setup_task
from task_catalog import OPS_BY_NAME

AGENT_UID = int(os.environ.get("AGENT_UID", "1000"))
AGENT_GID = int(os.environ.get("AGENT_GID", "1000"))

class _KernelWorkspace(Workspace):
    # demote agent shell so it cannot read the root:700 hidden grader/reference
    def shell_argv(self, command=None, *, cwd=None, env=None):
        argv = super().shell_argv(command, cwd=cwd, env=env)
        if sys.platform != "win32" and getattr(os, "geteuid", lambda: 1)() == 0:
            argv = ["setpriv","--reuid",str(AGENT_UID),"--regid",str(AGENT_GID),"--clear-groups","--",*argv]
        return argv

env = Environment(name="protean-v2")          # string literal: hud deploy static-parses it
_ws = _KernelWorkspace(WORKSPACE_ROOT, network=False,
                       env={"HOME":"/home/agent","USER":"agent","TRITON_CACHE_DIR":"/triton-cache"})

@env.initialize
async def _up(): await _ws.start(); env.add_capability(_ws.capability("shell"))
@env.shutdown
async def _down(): await _ws.stop()

@env.template(id="kernel_opt", description="Optimize a Triton kernel; graded on a held-out shape.")
async def kernel_opt(op_name: str, split: str = "train", seed: int = 0, validate_mode: str = None):
    # split in {"train","held_out"}; seed drives BOTH the held-out shape draw and the random allclose inputs
    setup_meta = setup_task(op_name, split=split, seed=seed, validate_mode=validate_mode)
    op = OPS_BY_NAME[op_name]
    answer = yield op.prompt          # <-- FIRST YIELD: prompt; agent now multi-turns in /workdir editing solution.py + calling bench.py
    evaluation = evaluate_kernel(op_name=op_name, split=split, seed=seed)   # <-- SECOND YIELD: hidden grade
    info = dict(evaluation.info or {}); info["setup"] = setup_meta; info["final_answer"] = None if answer is None else str(answer)
    evaluation.info = info
    yield evaluation

# ===== shape_sampler.py (the moat) =====
def sample_shape(op_name, split, seed):
    rng = random.Random(hash((op_name, split, seed)) & 0xffffffff)
    band = OPS_BY_NAME[op_name].train_band if split == "train" else OPS_BY_NAME[op_name].held_out_band
    # bands are DISJOINT continuous intervals, e.g. train M in [256,2048], held_out M in (2048,4096] step 32
    M = rng.randrange(band.lo, band.hi, band.step)
    return op_name_to_shape(op_name, M)   # resolves full tuple, e.g. (M, op.N) for rmsnorm

# ===== grader.py (hard-cap reconciliation, copied from verilog grader.py) =====
from hud.graders import EvaluationResult, SubScore
def evaluate_kernel(op_name, split, seed) -> EvaluationResult:
    hidden = hidden_dir(op_name)                       # tasks/<op>/donotaccess  (root:700)
    grade_mod = _load_grade_module(op_name, hidden)    # importlib from the hidden path
    try:
        r = grade_mod.grade(WORKSPACE_ROOT, op_name=op_name, split=split, seed=seed, hidden_root=hidden)
    except Exception as exc:                            # FAIL CLOSED -> reward 0, never error the episode
        return EvaluationResult(reward=0.0, done=True, content=f"{op_name}: ungraded ({exc})",
                                info={"hard_caps":["grader_error"]}, subscores=[])
    subs = _subscores_from_result(r)
    weighted = sum(s.weight*s.value for s in subs); reward = float(r.get("reward",0.0))
    if reward + 1e-9 < weighted:                        # negative-weight subscore reconciles display to capped reward
        subs.append(SubScore(name="hard_cap_penalty", weight=reward-weighted, value=1.0,
                             metadata={"hard_caps": r.get("hard_caps",[])}))
    return EvaluationResult(reward=reward, done=True, content=f"{op_name} graded",
                            info={"op":op_name,"split":split,"seed":seed,"hard_caps":r.get("hard_caps",[])}, subscores=subs)

# ===== tasks/<op>/donotaccess/grade.py (HIDDEN; skeleton defines the contract + calls sibling exec component) =====
# REWARD (Kevin base + daVinci PR + 4-layer guard). Runs in an isolated subprocess on the GPU.
def grade(workdir, op_name, split, seed, hidden_root) -> dict:
    sol = load_user_solution(workdir/"solution.py")          # the agent's @triton.jit + solution() entrypoint
    ref = import_reference(hidden_root/"reference.py")
    shape = sample_shape(op_name, split, seed)               # SAME sampler -> held-out shape the agent never saw
    inputs = make_random_inputs(shape, ref.dtype, seed)     # fresh random tensors

    hard_caps = []
    # --- LAYER 1: AST ban (no torch fallback / passthrough) ---
    if ast_contains_torch_compute(sol_source) or not ast_defines_triton_jit(sol_source): hard_caps.append("ast_ban")
    # --- compile + run with Triton JIT LAUNCH COUNTER instrumented (KernelGYM pattern) ---
    out, launched = run_with_launch_counter(sol.solution, inputs)
    ref_out = ref.eager(*inputs)
    # --- LAYER 2: launch counter (kernel must actually fire in BOTH correctness and timed pass) ---
    if launched == 0: hard_caps.append("no_triton_launch")
    # --- LAYER 3: dtype+shape match BEFORE timing ---
    if out.dtype != ref_out.dtype or out.shape != ref_out.shape: hard_caps.append("dtype_shape")
    # correctness on the natural tolerance for the dtype, re-checked with a 2nd seed AFTER timing
    correct = torch.allclose(out, ref_out, rtol=op.rtol, atol=op.atol)
    if hard_caps or not correct:
        return {"reward":0.0,"hard_caps":hard_caps,"subscores":{"correct":{"raw_score":float(correct),"weight":1.0}}}
    # --- TIMING: L2 flush + do_bench median-of-100, eager baseline same harness ---
    t_kernel, launched_timed = timed_with_launch_counter(sol.solution, inputs)  # L2 flush each iter
    t_eager  = do_bench(lambda: ref.eager(*inputs))
    if launched_timed == 0: return {"reward":0.0,"hard_caps":["no_triton_launch_timed"],"subscores":{}}
    # re-check allclose with 2nd random seed (defeats dtype-downcast/timing-only hacks)
    if not torch.allclose(sol.solution(*make_random_inputs(shape, ref.dtype, seed+1)),
                          ref.eager(*make_random_inputs(shape, ref.dtype, seed+1)), rtol=op.rtol, atol=op.atol):
        return {"reward":0.0,"hard_caps":["allclose_reseed"],"subscores":{}}
    # --- LAYER 4 + daVinci PR: profiling ratio gate ---
    speedup = t_eager / t_kernel                                       # clipped: min(speedup, 3.0)
    PR = t_triton_kernel_time / t_total_gpu_time                       # daVinci Eq1: fraction of GPU time in agent kernel
    if PR < PR_THRESHOLD or speedup < op.speedup_floor:               # floor 1.1x; PR gate kills sub-op cuBLAS hack
        speedup_score = 0.0
    else:
        speedup_score = min(speedup / op.P_TARGET, 1.0)              # P_TARGET=1.5
    # Kevin reward (per-turn shape; trainer sums gamma^t): R = 0.3*correct + correct*speedup_score*(1+PR)
    reward = 0.3 + min(speedup,3.0)/op.P_TARGET * (1.0)              # correct==1 here; PR already gated above
    reward = max(0.0, min(reward, 1.3))
    return {"reward":reward, "hard_caps":[], "subscores":{
        "correct":{"raw_score":1.0,"weight":0.3},
        "speedup":{"raw_score":speedup_score,"weight":0.7,"result":{"speedup":speedup,"PR":PR,"t_kernel_ms":t_kernel,"t_eager_ms":t_eager}}}}

# ===== tasks.py (taskset enumeration consumed by trainer + hud eval) =====
from env import env, kernel_opt   # re-export for `hud eval tasks.py`
tasks = []
for op in TASK_SPECS:
    for split in ("train","held_out"):
        t = kernel_opt(op_name=op.op_name, split=split, seed=0)   # calling template mints a Task
        t.slug = f"{op.op_name}-{split}"
        t.columns = {"op":op.op_name,"split":split,"dtype":op.dtype}
        tasks.append(t)

# ===== check_calibration.py (pre-8AM go/no-go) =====
# for each op: run base model N=10 via evaluate_kernel; gate:
#   assert frac_compilable >= 0.05 and frac_allclose >= 0.02 and group_reward_variance > 0 and 0 < median_reward < 1
#   else sys.exit(1)  -> DO NOT KICK GRPO
```

**Risks:** The hidden grade.py depends on the sibling exec/bench component (Triton launch instrumentation + L2-flush timing + subprocess isolation). If KernelGYM is not reusable (license or API mismatch) this becomes the real critical path, not the skeleton. Mitigation: skeleton ships a grade.py stub with a hard-coded launch-count fallback (sys.settrace on the JITFunction __call__) so the env is testable end-to-end on ONE op before the sibling lands; verify KernelGYM license in hour 0.; hud-python @env.template typed-param crash: adding `from __future__ import annotations` (or a Literal/Optional/Pydantic-typed param) breaks deploy with PydanticUserError -32000 (documented in verilog env.py). Mitigation: keep the future-import OUT, type split/validate_mode as plain `str`/`str|None`, pin the hud SDK commit.; Held-out shapes may already be in the 7B pretraining distribution, weakening the 'generalization not memorization' claim. Mitigation: use CONTINUOUS disjoint bands (train [256,2048], held_out (2048,4096]) so test shapes are provably unseen-as-exact-values; present the train-vs-held-out reward curve as directional, per audit.; Throughput: even with TRITON_CACHE_DIR baked, per-rollout compile+bench is the bottleneck; group=16 x 5 ops x multi-turn may not fit overnight on one H100. Mitigation: warmup-bake the cache at image build, easy-op curriculum start, do_bench median-of-100 only on the timed pass, step-150 abort -> pre-recorded curve.; Reward still gameable if PR_THRESHOLD/P_TARGET are mis-tuned. Mitigation: the Saturday-night red-team checklist (passthrough / never-launched / try-except fallback / bf16-downcast) MUST assert reward~0 on all four before the 8AM kick; this is the env's actual deliverable.; Calibration gate may fail (0% compilable from base 7B). Mitigation: collect 50-100 SFT warmup kernels as insurance and add a +0.1 partial-compile / +0.2 imports-triton credit to manufacture group variance, per audit residual-1.; Local-vs-image path divergence (WORKSPACE_ROOT /workdir on image, per-pid tmp locally; hidden_dir donotaccess) is a known footgun from verilog-template — must mirror its _resolve_workspace_root exactly or graders read the wrong tree.

### Modal H100 compile+bench harness (kernelforge.harness)  (~10h)
**Goal:** A single-GPU Modal H100 service that, given (op_id, shape_dict, agent_kernel_source), compiles the agent's Triton kernel in an isolated subprocess, runs a 4-layer anti-hack-gated correctness check (allclose vs PyTorch eager on TWO random seeds, dtype+shape match, Triton-launch instrumentation, AST ban), measures wall-clock speedup with do_bench-style median-of-N + L2 flush + locked clocks, computes the daVinci turn reward R = C(y)*(1 + speedup + PR), and returns a JSON BenchResult. It is the load-bearing reward oracle for both the HUD env grader (per-turn `bench` tool) and the overnight GRPO trainer (batched reward fn). The same code path serves the live demo stopwatch. Built by lifting the Modal H100 decorator from ml-template/modal_runner.py and the hard-cap EvaluationResult + uid-wall + /donotaccess hidden-grader pattern from verilog-template.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/bench_core.py` — PURE, GPU-only, no-Modal/no-HUD reward oracle. Defines BenchResult dataclass, compile_kernel(), correctness_check(), measure_speedup(), profiling_ratio(), and compute_reward(). The single source of truth for the reward formula. Runs inside the isolated subprocess. ~300 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/anti_hack.py` — 4-layer non-gameability guard: (1) ast_ban() static scan rejecting torch-passthrough/torch.compile/never-jit; (2) LaunchCounter monkeypatch of triton.runtime.driver to count real kernel launches; (3) dtype/shape gate; (4) speedup floor + P_TARGET + daVinci PR gate. Red-team harness asserts 4 known hacks score ~0. ~180 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/subprocess_runner.py` — Spawns bench_core in a fresh `python -c` subprocess with a process-group SIGKILL timeout (port verilog grade.py fail-closed pattern), CUDA_VISIBLE_DEVICES pinned, TRITON_CACHE_DIR set, captures JSON on stdout. Isolates segfaults/OOM/illegal-memory-access from the parent so one bad kernel never kills the rollout. ~120 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/ops.py` — Reference op registry: ~5 fixed ops (fused_elementwise, rmsnorm, softmax, layernorm, matmul_bias_relu) each with a torch-eager reference fn, a make_inputs(shape, seed, dtype) sampler, and natural rtol/atol. The SHAPE sampler (train M in {256,512,1024,2048} vs held-out test {384,768,1536}) lives here. ~200 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/modal_app.py` — Modal app: image (from_dockerfile with TRITON_CACHE_DIR baked + warmup pass), @app.function(gpu='H100', timeout=...) bench_remote(payload)->dict, a deployed always-warm class with @modal.enter() that locks GPU clocks + warms the Triton cache, and a batched bench_batch() for the trainer. Lifted from ml-template/modal_runner.py. ~200 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/Dockerfile.hud` — CUDA+PyTorch+Triton base; ENV TRITON_CACHE_DIR=/triton_cache baked + a build-time warmup that JIT-compiles the 5 reference kernels so the cache layer ships in the image (kills 30-120s cold start). Mirrors verilog Dockerfile structure (uid-wall agent user, /donotaccess hidden grader). ~50 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/env.py` — HUD v6 env: two-yield @env.template that yields the prompt, exposes a persistent shell + a `bench` capability the agent calls each turn (multi-turn state), then on final yield builds the hard-cap EvaluationResult by hand (NOT combine()). Calls into modal_app.bench_remote. Ported from verilog env.py + grader.py.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tests/test_anti_hack.py` — Red-team go/no-go: 4 hack kernels (passthrough, never-launched jit, try/except torch fallback, bf16 downcast) MUST score ~0; a known-good elementwise kernel MUST score >0 with speedup>1. The Hour 4-6 verifier gate. ~120 lines.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tests/check_calibration.py` — Pre-GRPO gate (port verilog check_calibration pattern): run base Qwen2.5-Coder-7B 10x/op over the 5 ops, assert >=5% compilable, >=2% allclose, nonzero group variance, reward band 20-50%. Refuses to print GO unless gates pass. ~100 lines.

**APIs:** bench_core.compile_kernel(source, op_id) -> (module|None, err); bench_core.correctness_check(mod, op, shape, dtype, seeds=(0,1)) -> (bool, diag); bench_core.measure_speedup(fn_k, fn_e, inputs, n=100, warmup=25) -> {t_kernel_ms,t_eager_ms,speedup}; bench_core.profiling_ratio(fn_k, inputs) -> float; bench_core.compute_reward(correct, speedup, pr, launches_c, launches_t, dtype_ok, shape_ok, p_target=1.5, speedup_floor=1.1, pr_min=0.5) -> {reward, cap, ...}; anti_hack.ast_ban(source) -> reason|None; anti_hack.LaunchCounter() (context manager, .count); subprocess_runner.run_bench(payload, timeout_s=90) -> dict; ops.OpSpec(op_id, eager_fn, make_inputs, rtol, atol, kernel_signature); ops.sample_shape(op_id, split, rng) -> dict; modal_app.bench_remote.remote(payload) -> dict; modal_app.bench_batch.remote(payloads) -> list[dict]; env.kernel_task(op_id, shape_split, max_turns=5)  # @env.template two-yield

**daVinci integration:** FOUR daVinci techniques land directly in this harness, all reward-side (the agent/skill-library side is a separate component): (1) TURN REWARD Eq1 R=C(y)*(1+speedup+PR) is implemented verbatim in compute_reward — this is the headline daVinci adoption and replaces a flat Kevin reward; speedup is normalized by P_TARGET=1.5 to cap outliers, the 0.3 correct-but-slow floor is kept from Kevin for GRPO variance. (2) PROFILING RATIO (PR) — daVinci's anti-hack term: profiling_ratio() uses torch.profiler to compute T_triton_kernels/T_total_gpu; a pr_min=0.5 gate zeroes any kernel that delegates most work back to torch (defeats the 'call one tiny triton kernel then torch for the rest' hack). This is daVinci's profiling-aware contribution folded into our 4-layer guard as Layer 4. (3) EXECUTION-VERIFIED launch instrumentation (daVinci anti-hack + KernelGYM): LaunchCounter must see >0 real Triton launches in BOTH the correctness AND the timed pass — defeats never-invoked @triton.jit. (4) PRS PROFILING-AWARE SAMPLING (Eq4 p=clip((PR-tau)/s,0,1)): the trainer can read the per-task PR returned in BenchResult and bias rollout sampling toward kernels that dominate end-to-end runtime — exposed as a field, used by the GRPO component, not the harness itself. NOT adopted in 24h: the three-agent shared backbone, BM25 skill selection, skill summary/verification (Eq5/6/9), TRLOO per-agent advantages — these are the agent+training components; the harness only needs to RETURN the per-turn reward and PR so those can be layered on. The harness deliberately exposes pr and per-turn reward in the JSON so a stretch-goal skill-library agent can consume them without harness changes.

```python
… compute_reward (daVinci Eq1, gated): \n```\ndef compute_reward(correct, speedup, pr, launches_train, launches_timed,\n                   dtype_ok, shape_ok, p_target=1.5, speedup_floor=1.1, pr_min=0.5):\n    # Layer 3+4 hard caps BEFORE crediting anything\n    if not (correct and dtype_ok and shape_ok):       return {reward:0.0, cap:'incorrect'}\n    if launches_train==0 or launches_timed==0:        return {reward:0.0, cap:'no_triton_launch'}\n    if pr < pr_min:                                    return {reward:0.0, cap:'low_profiling_ratio'}\n    if speedup < speedup_floor:                        return {reward:0.3, cap:'correct_but_slow'}  # Kevin 0.3 floor\n    # daVinci Eq1 turn reward: R = C(y)*(1 + speedup_norm + PR)\n    speedup_norm = min(speedup / p_target, 1.0)        # cap runaway/outlier speedups\n    reward = 1.0 * (1.0 + speedup_norm + pr)           # in [1.0, 3.0] for correct fast kernels\n    return {reward: reward, speedup: speedup, pr: pr, cap: None}\n```\n\n… subprocess_runner.run_bench (fail-closed isolation): \n```\ndef run_bench(payload, timeout_s=90):\n    reason = ast_ban(payload['source'])               # Layer 1, static, before any GPU work\n    if reason: return {reward:0.0, cap:'ast_ban', detail:reason}\n    p = Popen(['python','-c',BENCH_ENTRY], env={CUDA_VISIBLE_DEVICES, TRITON_CACHE_DIR}, \n              stdin=json(payload), stdout=PIPE, start_new_session=True)  # own process group\n    try: out = p.communicate(timeout=timeout_s)\n    except TimeoutExpired:\n        os.killpg(p.pid, SIGKILL); return {reward:0.0, cap:'timeout'}   # kills hung/illegal-mem kernel\n    if p.returncode != 0: return {reward:0.0, cap:'crash', detail:stderr_tail}  # segfault/OOM -> 0, not raise\n    return json.loads(out)\n```\n\n… BENCH_ENTRY (in subprocess, the real oracle): \n```\npayload = json.load(stdin); op = OPS[payload['op_id']]; shape = payload['shape']\nmod, err = compile_kernel(payload['source'], op.op_id)\nif mod is None: emit({reward:0.0, cap:'compile_error', detail:err}); exit\ninp0 = op.make_inputs(shape, seed=0, dtype=payload['dtype'])\nwith LaunchCounter() as lc_c:                          # Layer 2 in correctness pass\n    ok, diag = correctness_check(mod, op, shape, dtype)\nfn_k = lambda: mod.kernel(*inp_timed); fn_e = lambda: op.eager_fn(*inp_timed)\nwith LaunchCounter() as lc_t:                          # Layer 2 in timed pass\n    timing = measure_speedup(fn_k, fn_e, inp_timed)\npr = profiling_ratio(fn_k, inp_timed)\nok2,_ = correctness_check(mod, op, shape, dtype, seeds=(1,))   # post-timing re-check, 2nd seed\nemit(compute_reward(ok and ok2, timing.speedup, pr, lc_c.count, lc_t.count, diag.dtype_ok, diag.shape_ok))\n```\n\n… measure_speedup (locked clocks + L2 flush, do_bench style): \n```\ndef measure_speedup(fn_k, fn_e, inputs, n=100, warmup=25):\n    cache = torch.empty(256*1024*1024, dtype=torch.int8, device='cuda')  # L2 flush buffer\n    def timed(fn):\n        for _ in range(warmup): fn()\n        torch.cuda.synchronize(); ts=[]\n        for _ in range(n):\n            cache.zero_()                              # evict L2 so we measure cold mem like real use\n            s,e = cuda.Event(True), cuda.Event(True); s.record(); fn(); e.record()\n            torch.cuda.synchronize(); ts.append(s.elapsed_time(e))\n        return median(ts)\n    tk, te = timed(fn_k), timed(fn_e)\n    return {t_kernel_ms:tk, t_eager_ms:te, speedup: te/tk}\n```\n\n… env.kernel_task (HUD two-yield, multi-turn, hard-cap EvaluationResult): \n```\n@env.template(id='kernel-opt')\nasync def kernel_task(op_id, shape_split='test', dtype='float16', max_turns=5):\n    shape = sample_shape(op_id, shape_split, rng)      # held-out shape: the moat\n    answer = yield make_prompt(op_id, shape)           # agent gets shell + `bench` tool; revises K turns\n    best = max(turn_results, key=lambda r: r['reward']) # reward = best correct kernel across turns\n    # build EvaluationResult BY HAND (verilog pattern) — never combine(), preserves hard cap\n    subs = [SubScore('correct', .5, 1.0 if best.correct else 0.0),\n            SubScore('speedup', .5, min(best.speedup/P_TARGET,1.0) if best.correct else 0.0)]\n    if best.cap: subs.append(SubScore('hard_cap_penalty', best.reward - sum(s.weight*s.value for s in subs), 1.0,\n                                      metadata={'cap':best.cap}))\n    yield EvaluationResult(reward=best.reward, done=True, content=f\"{op_id} {shape}\",\n                           info={'op_id':op_id,'shape':shape,'split':shape_split,'speedup':best.speedup,\n                                 'launches':best.launches,'cap':best.cap}, subscores=subs)\n```\n\n… `bench` tool the agent calls each turn (the per-turn Markov feedback): returns {compiled, allclose, max_abs_err, t_kernel_ms, t_eager_ms, speedup, launch_count, cap} — agent OBSERVES last error + last latency as state before the next revision. Internally just subprocess_runner.run_bench on the agent's current file.
```

**Risks:** TRITON_CACHE_DIR baked into image may MISS the agent's novel kernels (cache only helps recompiling the SAME source); cold-start still bites first-seen kernels 30-120s. Mitigation: warm class + per-op cache warmup covers reference shapes; budget the first-turn latency, keep timeout_s=90.; do_bench median timing variance on a shared/contended H100 can make speedup non-deterministic across calls -> noisy reward -> GRPO instability. Mitigation: lock GPU clocks in @modal.enter(), median-of-100, L2 flush, fixed warmup; for the DEMO never time live (pre-record per audit).; profiling_ratio via torch.profiler adds ~50-200ms overhead per bench and can itself fail/segfault on weird kernels. Mitigation: wrap PR in try/except -> pr=0.0 (which then trips pr_min gate to reward 0, fail-closed); make PR gate optional behind a flag if it proves flaky before 8AM kick.; Subprocess-per-bench spawn cost (~1-2s python+CUDA init) caps trainer throughput; with 5 turns x group=8 it dominates wall-clock. Mitigation: bench_batch() reuses ONE warm container/process where safe; accept subprocess isolation cost as the price of fail-closed (an in-process illegal-memory-access kills the whole rollout otherwise).; LaunchCounter monkeypatch is tied to a specific triton.runtime.driver internal that can change across triton versions -> silently counts 0 (false hard-cap) or always-pass (hole). Mitigation: pin triton version; test_anti_hack.py asserts a KNOWN-GOOD kernel registers >0 launches AND a never-launched one registers 0 — fails build if the hook breaks.; Reward band may not land in 20-50% for the base 7B model -> GRPO stalls (too sparse) or saturates. Mitigation: check_calibration.py is a hard GO/NO-GO gate before 8AM; tunable knobs are speedup_floor, the 0.3 partial-credit floor, and a +0.1 compiles / +0.2 imports-triton bootstrap that anneals out.; Held-out shape sampler could leak: if make_inputs uses shapes that compile to identical Triton configs as train, generalization claim is hollow. Mitigation: document the train/test split (train M in {256,512,1024,2048}, test {384,768,1536}), assert disjoint, and make the train-vs-test reward curve the money slide.

### Reward + Verifier for Protean (correctness x measured wall-clock speedup, daVinci profiling-ratio term, 4-layer programmatic anti-hack, held-out continuous-shape generalization split). A single root:700 hidden grader (mirroring verilog-template/grade.py + the donotaccess pattern), the public-side grader.py mapper, and the HUD two-yield env wrapper. Runs on a Modal H100 and is called by BOTH the pre-GRPO calibration smoke test and the overnight GRPO rollout workers.  (~9.5h)
**Goal:** Emit a non-gameable, low-variance scalar reward in [0, ~3] for an agent-submitted Triton kernel against a hidden PyTorch reference, where the TIMED evaluation runs on HELD-OUT continuous shapes disjoint from the agent's training shapes. Reward = correctness_gate x (Kevin floor 0.3 + measured_speedup x daVinci_PR), with four programmatic gates (AST ban, kernel-launch counter, dtype/shape match, 1.1x speedup floor) that drive passthrough / never-launched / try-except-fallback / bf16-downcast hacks to ~0 BEFORE any speedup credit. The verifier is the project's actual non-gameability deliverable; the train-shape-vs-held-out-shape reward gap is the moat slide.

**Files:**
- `protean/env.py` — HUD Environment + @env.template two-yield generator, lifted from verilog-template/env.py. yield prompt; read agent kernel from /workdir/solution.py; call evaluate_task(); yield a hand-built EvaluationResult (NOT combine(), to preserve the correctness hard cap, exactly as verilog grader.py does). Wraps workspace in a setpriv-demoted _AgentWorkspace.
- `protean/grader.py` — Public-side evaluate_task(task_id, shape_split) -> EvaluationResult. importlib-loads the root:700 hidden grade module (mirrors verilog-template/grader.py), maps result dict {reward, subscores, hard_caps} -> list[SubScore] + EvaluationResult, appends a negative-weight hard_cap_penalty subscore to reconcile displayed subscores to the capped reward, and fails CLOSED to reward=0 on ANY exception.
- `protean/tasks/<op>/donotaccess/grade.py` — THE hidden verifier (root:700, never copied into the agent tree). grade(workdir, kernel_override, hidden_root, shape_split) -> dict. Runs: Layer-1 AST ban -> import agent kernel in a setpriv-demoted subprocess -> Layer-2 launch counter + Layer-3 dtype/shape match -> correctness allclose on held-out shapes (2 seeds, post-timing re-check) -> launch-instrumented do_bench timing -> daVinci PR ratio -> Layer-4 speedup floor -> reward.
- `protean/tasks/<op>/donotaccess/reference.py` — Hidden PyTorch eager reference fn(*tensors) for the op + make_inputs(shape, dtype, seed) factory. The answer key; root:700. ~5 ops: fused elementwise (gelu+mul), softmax-row, layernorm, rmsnorm, matmul-bias-relu.
- `protean/tasks/<op>/shapes.json` — Train/test shape split (the moat). train M in {256,512,1024,2048}; held-out M sampled CONTINUOUS from disjoint band [320,1920]\{train}. grade() picks shapes by shape_split arg so train rollouts and eval rollouts never share shapes.
- `protean/_harness/launch_probe.py` — Triton kernel-launch instrumentation (ports KernelGYM's pattern): monkeypatch triton.runtime JITFunction.run / wrap triton.jit to increment a global launch counter. Imported by grade.py before running the agent kernel; counter must be >0 in BOTH correctness and timed passes.
- `protean/_harness/bench.py` — do_bench(fn, warmup, rep) with L2-cache flush + CUDA-event timing + median-of-100 + locked clocks; also exposes profile_gpu_time(fn) via torch.profiler to compute the PR denominator (total measured GPU kernel time).
- `protean/scripts/check_calibration.py` — Pre-GRPO go/no-go gate (ports verilog check_calibration.py). Asserts golden_kernel->reward~speedup>=1, eager-passthrough->0, never-launched->0, bf16-downcast->0; then runs base Qwen2.5-Coder-7B 10x/op and REQUIRES >=5% compilable, >=2% allclose, nonzero per-group variance, 20-50% band. Exit nonzero blocks the 8AM kick.
- `protean/scripts/redteam.py` — Adversarial probe suite: submits the 4 canonical hacks (passthrough, @triton.jit-never-called, try/except torch fallback, bf16 downcast) and asserts reward < 0.05 for each. Run before 8AM.
- `protean/modal_app.py` — Modal H100 image (lifts ml-template/modal_runner.py decorator) with TRITON_CACHE_DIR baked in + a warmup compile pass to kill 30-120s JIT cold-start. Exposes grade_remote(task_id, kernel_src, shape_split) for fan-out grading.

**APIs:** grade(workdir: Path, kernel_override: str | None, hidden_root: Path, shape_split: str = 'test') -> dict  # returns {reward: float, hard_caps: list[str], subscores: dict, metrics: dict}; evaluate_task(task_id: str, shape_split: str = 'test') -> hud.graders.EvaluationResult; capped(reward: float, caps: list[str]) -> dict; do_bench(fn: Callable, warmup: int = 25, rep: int = 100, flush_l2: bool = True) -> float  # median ms; profile_gpu_time(fn: Callable) -> tuple[float, float]  # (total_gpu_ns, triton_gpu_ns); launch_probe() -> ContextManager  # .launches: int, .triton_gpu_ns: float; monkeypatches triton JITFunction.run; import_agent_kernel_subprocess(src: str, timeout_s: int = 60) -> Callable  # setpriv uid-1000 demoted, returns kfn; load_shapes(hidden_root: Path, split: Literal['train','test']) -> list[tuple[int,...]]; make_inputs(shape: tuple, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, ...]  # per-op, in reference.py; @env.template(id='kernel_task') async def kernel_task(task_id: str, shape_split: str = 'test')  # yields prompt then EvaluationResult; SubScore(name: str, weight: float, value: float, metadata: dict | None)  # from hud.graders; EvaluationResult(reward: float, done: bool, content: str, info: dict, subscores: list[SubScore])  # from hud.graders

**daVinci integration:** Five daVinci-kernel techniques fold directly into the reward path; the rest of daVinci (skill library, 3 agents) is explicitly OUT of scope for the 24h verifier component (it lives in the RL/agent component). (1) TURN REWARD / PR term (Eq1): reward = C(y)*(CORRECT_FLOOR + speed_credit*PR) where PR = triton_gpu_ns / total_gpu_ns from torch.profiler. This is the headline integration — it closes the partial-runtime exploit (Triton fires on a trivial sub-op while torch/cuBLAS does the heavy lifting) that Kevin's plain speedup*correct does NOT. Published June 2026 during the hackathon, so 'we already integrated this weekend's SotA' is a real talking point. (2) PR gate (Eq4/PRS): if PR < 0.5 we cap reward at CORRECT_FLOOR (correctness only, zero speedup credit) instead of rewarding a fake speedup. (3) Lazy-skill filter -> Layer-1 AST ban: daVinci rejects skills whose name/desc/content suggest 'fall back to PyTorch' / 'use torch.compile'; we apply the same rule statically to the kernel SOURCE (ban torch.compile/jit/aten/_inductor, require a real @triton.jit def). (4) Execution verification (Eq6 spirit): every reward is execution-verified by actually launching the kernel and re-running allclose post-timing with a fresh seed — never a static or LLM judge. (5) MRS/importance-ratio filtering (Eq3) is NOT in the grader but is a one-line note for the trainer (drop samples with large pi_train/pi_rollout deviation). De-scoped vs daVinci: we collapse the multi-turn G_{i,t} discounted return into a single terminal eval (reward on best correct kernel across the agent's K<=3 turns) because the two-yield template grades once; the per-turn shaping is a stretch. Skill verification thresholds (alpha=beta=1.2) are NOT used here (no skill library in v1). Kevin's 0.3*correct floor is kept as CORRECT_FLOOR to guarantee GRPO reward density at 7B.

```python
# ===== tasks/<op>/donotaccess/grade.py  (THE hidden verifier) =====
# Mirrors verilog grade.py signature: grade(workdir, override, hidden_root) -> dict
P_TARGET = 1.5          # daVinci-style normalizer; speedup credit saturates near here
SPEEDUP_FLOOR = 1.1     # Layer-4: no speedup credit below this (anti dtype/noise hack)
PR_THRESH = 0.5         # daVinci PR gate: Triton must own >=50% of measured GPU time
CORRECT_FLOOR = 0.3     # Kevin: 0.3*correct baseline keeps reward dense for GRPO

BANNED_AST = {"torch.compile","torch.jit","aten","_inductor"}  # Layer-1 lazy-skill/fallback ban

def grade(workdir, kernel_override, hidden_root, shape_split="test"):
    src = kernel_override or read(workdir/"solution.py")
    # --- Layer 1: static AST ban (daVinci lazy-skill filter, pre-exec) ---
    tree = ast.parse(src)
    if uses_banned_call(tree, BANNED_AST) or not has_triton_jit_def(tree):
        return capped(0.0, ["ast_ban"])           # hard cap
    ref, make_inputs = import_hidden(hidden_root/"reference.py")
    shapes = load_shapes(hidden_root, shape_split) # HELD-OUT continuous shapes for eval
    # run agent kernel in a setpriv-demoted subprocess w/ launch_probe injected
    with launch_probe() as probe:                  # Layer 2 instrumentation armed
        try:
            kfn = import_agent_kernel_subprocess(src)   # demoted uid 1000, walled from key
        except Exception:
            return capped(0.0, ["compile_error"])  # uncompilable -> 0 (fail closed)
    # --- correctness on held-out shapes, 2 seeds ---
    for shape in shapes:
        for seed in (0, 1):
            xs = make_inputs(shape, dtype, seed)
            with launch_probe() as probe:
                out = kfn(*xs)
            ref_out = ref(*[x.clone() for x in xs])
            # Layer 3: dtype + shape match BEFORE allclose (anti bf16-downcast)
            if out.dtype != ref_out.dtype or out.shape != ref_out.shape:
                return capped(0.0, ["dtype_shape_mismatch"])
            if not torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3):
                return capped(0.0, ["incorrect"])
            if probe.launches == 0:                # Layer 2: must fire in correctness pass
                return capped(0.0, ["no_triton_launch_correct"])
    correct = 1.0
    # --- timing on held-out shapes (locked clocks, L2 flush, median-100) ---
    with launch_probe() as probe:
        t_kernel = do_bench(lambda: kfn(*xs))
        gpu_total = profile_gpu_time(lambda: kfn(*xs))   # PR denominator
    if probe.launches == 0:                        # Layer 2: must fire in TIMED pass too
        return capped(0.0, ["no_triton_launch_timed"])
    t_ref = do_bench(lambda: ref(*xs))             # PyTorch eager reference
    # post-timing allclose re-check w/ a 3rd seed (anti time-then-cheat)
    if not torch.allclose(kfn(*make_inputs(shape,dtype,7)), ref(*make_inputs(shape,dtype,7)), 1e-3,1e-3):
        return capped(0.0, ["post_timing_mismatch"])
    speedup = t_ref / t_kernel
    # --- daVinci PR (Eq1): fraction of measured GPU time spent in agent's Triton kernels ---
    PR = probe.triton_gpu_ns / max(gpu_total, 1e-9)
    if PR < PR_THRESH:                             # Triton fired but did trivial work
        return capped(CORRECT_FLOOR, ["low_PR"])   # correctness credit only, no speedup
    # --- Layer 4: speedup floor ---
    speed_credit = 0.0 if speedup < SPEEDUP_FLOOR else min(speedup / P_TARGET, 2.0)
    # daVinci turn reward shape: C(y)*(1 + speedup + PR), de-scoped to single eval turn:
    reward = correct * (CORRECT_FLOOR + speed_credit * PR)
    return {"reward": reward, "hard_caps": [],
            "subscores": {"correct": {"weight":0.3,"raw_score":correct},
                          "speedup": {"weight":0.5,"raw_score":min(speedup/P_TARGET,1)},
                          "PR":      {"weight":0.2,"raw_score":PR}},
            "metrics": {"speedup":speedup,"PR":PR,"t_kernel_ms":t_kernel,"t_ref_ms":t_ref}}

def capped(r, caps): return {"reward": r, "hard_caps": caps, "subscores": {}}

# ===== grader.py (public mapper, mirrors verilog grader.py exactly) =====
def evaluate_task(task_id, shape_split="test") -> EvaluationResult:
    hidden = hidden_dir(task_id)                    # /donotaccess/<id> or tasks/<id>/donotaccess
    gmod = _load_grade_module(task_id, hidden)
    try:
        res = gmod.grade(WORKSPACE_ROOT, None, hidden, shape_split)
    except Exception as e:
        return EvaluationResult(reward=0.0, done=True,         # FAIL CLOSED
            content=f"{task_id}: ungradeable ({type(e).__name__})",
            info={"hard_caps":["grader_error"]}, subscores=[])
    subs = _subscores_from_result(res)             # SubScore(name,weight,value,metadata)
    wsum = sum(s.weight*s.value for s in subs); reward = float(res["reward"])
    if reward + 1e-9 < wsum:                        # negative-weight reconciliation (verilog trick)
        subs.append(SubScore("hard_cap_penalty", reward-wsum, 1.0, {"hard_caps":res["hard_caps"]}))
    return EvaluationResult(reward=reward, done=True, content=f"{task_id} graded",
        info={"task_id":task_id,"shape_split":shape_split,**res.get("metrics",{}),
              "hard_caps":res["hard_caps"]}, subscores=subs)

# ===== env.py (two-yield, lifted from verilog env.py) =====
@env.template(id="kernel_task")
async def kernel_task(task_id, shape_split="test"):
    setup_task(task_id)                             # reset /workdir to op stub + prompt, NO donotaccess
    answer = yield TASK_SPECS[task_id].prompt       # agent writes /workdir/solution.py over K<=3 turns
    ev = evaluate_task(task_id, shape_split)         # held-out shapes by default
    ev.info["final_answer"] = None if answer is None else str(answer)
    yield ev
```

**Risks:** PR term reliability: torch.profiler attribution of GPU ns to Triton vs torch kernels can be noisy/version-fragile on H100. Mitigation: if profiler attribution is flaky by Sat eve, fall back to PR=1.0-only-when-launch-counter-confirms (degrade PR gate to the binary launch gate, which is robust) and present PR as a stretch on the slide.; Held-out shapes may already sit in the 7B pretraining distribution, weakening the 'generalization not memorization' claim. Mitigation: sample test M from a CONTINUOUS band disjoint from train (e.g. 384/768/1536 + random) and frame the result as directional; the train-vs-test reward GAP is the evidence, not absolute numbers.; Reward sparsity / GRPO advantage collapse at 7B: base model may be near-0 allclose -> flat groups. Mitigation: CORRECT_FLOOR=0.3 (Kevin) + curriculum start on fused-elementwise + the calibration gate (>=2% allclose, nonzero variance) BLOCKS the kick if too sparse; add a +0.1 compile-credit shaped term only if the gate fails.; Timing variance inflating/deflating speedup -> reward noise. Mitigation: locked GPU clocks, L2 flush, median-of-100, SPEEDUP_FLOOR=1.1 dead-band, and re-check allclose post-timing; never time live in the demo (pre-record).; Subprocess/setpriv kernel execution adds per-rollout latency and can hang on a bad kernel. Mitigation: 60s subprocess timeout -> compile_error->0; TRITON_CACHE_DIR warm; fan-out grading across Modal workers.; Modal H100 availability / $250 budget: overnight GRPO + grading fan-out can exhaust credit. Mitigation: colocate trainer+grader on one H100, cache compiled kernels, step-150 abort rule.; combine() footgun: routing through hud.graders.combine() renormalizes weights and ERASES the correctness hard cap (verified in verilog grader.py comments). Mitigation: build EvaluationResult by hand with the negative-weight hard_cap_penalty reconciliation — already in the pseudocode.

### Task generation: parameterized shape sampler + deterministic train/test split for Protean  (~5h)
**Goal:** Turn a small bank of ~5 fixed KernelBench-L2-style fused ops into an effectively-unbounded HUD v6 taskset by sampling tensor SHAPES from continuous ranges, with a documented, deterministic, DISJOINT train/test shape split. This is the generalization-verifier moat: train shapes and held-out test shapes never overlap, so the trained-vs-base delta on held-out shapes proves reasoning over memorization. Each minted task is a standard HUD two-yield @env.template (yield prompt; yield EvaluationResult) whose prompt embeds a concrete-shape PyTorch reference module (KernelBench Model/get_inputs/get_init_inputs contract) and whose grader (separate component) checks allclose x measured speedup on that exact shape. The sampler must (a) produce reproducible shape sets from a seed, (b) guarantee train/test disjointness by construction, (c) tag every task with columns for filtering/curriculum, and (d) feed both the offline taskset-sync path and the in-loop GRPO rollout path.

**Files:**
- `kernelforge/ops/__init__.py` — OP_BANK registry: maps op_name -> OpSpec. The ~5 fixed fused ops (the only authored content).
- `kernelforge/ops/specs.py` — OpSpec dataclass + the 5 concrete op definitions (fused_bias_gelu, layernorm, softmax_rowmax, fused_mul_add_relu, rmsnorm). Each carries a reference-module SOURCE STRING template, the shape-param schema (which dims are sampled), dtype, and natural allclose tolerance.
- `kernelforge/sampler.py` — Core: ShapeSampler. Continuous-range shape sampling, deterministic seeded RNG, train/test disjoint split logic, ShapeSample dataclass, prompt rendering from OpSpec + shape.
- `kernelforge/splits.py` — Frozen, documented TRAIN_SHAPE_SPACE / TEST_SHAPE_SPACE range definitions + the disjointness invariant and the assertion that guards it. This is the on-slide 'money split' source of truth.
- `kernelforge/tasks.py` — HUD entrypoint. Builds the public `tasks` list for `hud eval`/`hud sync` by calling the kernel_task @env.template once per sampled ShapeSample; sets .slug and .columns. Re-exports env.
- `kernelforge/env.py` — Environment(name=...), the kernel_task @env.template (two-yield), uid-wall Workspace, Modal-image init. The grader call lives here but is a SEPARATE component; this file only consumes ShapeSample.to_prompt() and passes shape/op metadata to the grader.
- `kernelforge/manifest.py` — CLI: emit/freeze the exact sampled train & test ShapeSamples to JSONL (shapes_train.jsonl / shapes_test.jsonl) so the overnight GRPO run and the demo eval draw from byte-identical, version-pinned sets. Mirrors daVinci skill-library JSONL snapshot pattern.
- `tests/test_sampler.py` — Asserts determinism (same seed -> same shapes), train/test disjointness, in-range membership, and that every rendered prompt imports cleanly + exposes Model/get_inputs/get_init_inputs.

**APIs:** @dataclass(frozen=True) class OpSpec: name:str; level:str; ref_src_template:str; sampled_dims:tuple[str,...]; fixed_kwargs:dict; dtype:str='float16'; rtol:float; atol:float; tags:tuple[str,...]  # ref_src_template is a str.format template with {M},{N},{K} placeholders rendering a KernelBench-style module; @dataclass(frozen=True) class ShapeSample: op_name:str; dims:dict[str,int]; dtype:str; split:str; seed:int; def task_id(self)->str  # stable hash-based id e.g. f'{op}_{M}x{N}_{dtype}_{split}'; ShapeSample.to_prompt(self)->str  # renders OpSpec.ref_src_template with dims -> full PyTorch reference module source + the multi-turn instructions; the agent must emit ModelNew with identical get_inputs/get_init_inputs; ShapeSample.to_columns(self)->dict  # {op, split, M, N, dtype, level, tags} for HUD .columns filtering; class ShapeSampler:
  def __init__(self, seed:int, ops:list[OpSpec]=OP_BANK)
  def sample(self, split:Literal['train','test'], n:int, op_filter:list[str]|None=None, curriculum_level:str|None=None)->list[ShapeSample]
  def _sample_dims(self, op:OpSpec, space:ShapeSpace, rng)->dict[str,int]  # draws each sampled dim from its continuous range, snaps per granularity rule; def is_disjoint(train:list[ShapeSample], test:list[ShapeSample])->bool  # invariant guard, also called in tests + at sync time; kernel_task(op_name:str, dims:dict, dtype:str, split:str, rtol:float, atol:float) -> Task  # the @env.template async generator: setup -> `answer = yield prompt` -> `yield grade(...)`; manifest.freeze(seed:int, n_train:int, n_test:int, out_dir:Path)->None  # writes shapes_{train,test}.jsonl, asserts disjoint before writing; manifest.load(path:Path)->list[ShapeSample]  # rehydrate for GRPO rollout + demo eval (single source of truth)

**daVinci integration:** daVinci-kernel techniques folded into the SAMPLER component (only where they add value within 24h): (1) JSONL versioned snapshots — daVinci stores its skill library as versioned JSONL broadcast to rollout workers; manifest.freeze() applies the same discipline to SHAPE sets so the overnight GRPO rollouts, the calibration gate, and the 1PM demo eval all draw from a byte-identical, seed-pinned distribution (no silent re-sampling drift). (2) Curriculum hook — daVinci/DRTriton advance difficulty by accuracy threshold; sampler.sample(curriculum_level=...) and the OpSpec.level field let the GRPO loop start on the easiest ops (single fused elementwise, guarantees reward density per the calibration gate) and anneal toward layernorm/softmax. (3) Task tagging for per-agent routing — daVinci's Skill-Selection agent reranks by tags; to_columns() emits op/tags so the (optional, stretch) skill-selection path and curriculum filters can subset tasks without re-deriving shapes. (4) Anti-hack ALIGNMENT (enforced in the grader component, but the sampler enables it): the sampler bakes the natural per-op dtype tolerance (OpSpec.rtol/atol) into each task so the grader can reject the precision-downgrade exploit; and the held-out continuous shapes are the sampler's contribution to daVinci's anti-hack philosophy (a kernel hardcoded to a train shape fails on test shapes). EXPLICITLY DEFERRED (cite as prior art, do NOT build): DRTriton's CSP-DAG op-graph composer and daVinci's full three-agent skill-summary loop — the parameterized shape sampler over fixed ops is the de-scoped, buildable moat per the audit.

```python
# ---------- splits.py : the moat, frozen & documented ----------
# Granularity rule: shapes snap to a stride so allclose/timing are stable and dims are realistic.
# DISJOINTNESS BY CONSTRUCTION: train draws from a fixed discrete grid; test draws from the
# complementary continuous range (interpolation) so a test shape can NEVER equal a train shape.
@dataclass(frozen=True)
class ShapeSpace:
    # per-dim: ('train_grid', tuple) for train  OR ('test_range', lo, hi, stride) for test
    dims: dict[str, tuple]

TRAIN_SPACE = ShapeSpace(dims={
    "M": ("grid", (256, 512, 1024, 2048)),     # exact powers/multiples the model sees
    "N": ("grid", (256, 512, 1024, 2048)),
})
# Test = the GAPS between train points, snapped to stride 128, guaranteed not in the grid.
TEST_SPACE  = ShapeSpace(dims={
    "M": ("range", 384, 1792, 128),            # 384,768,1152,1408,1536,1792 ... all NOT in train grid
    "N": ("range", 384, 1792, 128),
})
def _assert_split_disjoint():
    train_pts = set(product over grid dims)
    test_pts  = {snap(x) for x in test range}
    assert train_pts & test_pts == set()       # fails the build if someone edits ranges badly

# ---------- sampler.py ----------
class ShapeSampler:
    def __init__(self, seed, ops=OP_BANK):
        self.seed = seed; self.ops = {o.name: o for o in ops}
    def sample(self, split, n, op_filter=None, curriculum_level=None):
        ops = [o for o in self.ops.values()
               if (not op_filter or o.name in op_filter)
               and (not curriculum_level or o.level == curriculum_level)]
        out, seen = [], set()
        for i in range(n):
            # derive a per-draw deterministic RNG: same (seed,split,i) -> same shape forever
            rng = Random(hash((self.seed, split, i)))
            op  = rng.choice(ops)
            dims = self._sample_dims(op, TRAIN_SPACE if split=="train" else TEST_SPACE, rng)
            s = ShapeSample(op.name, dims, op.dtype, split, self.seed)
            if s.task_id() in seen:    # de-dupe within a split
                continue
            seen.add(s.task_id()); out.append(s)
        return out
    def _sample_dims(self, op, space, rng):
        d = {}
        for dim in op.sampled_dims:                 # e.g. ("M","N")
            kind, *p = space.dims[dim]
            if kind == "grid":   d[dim] = rng.choice(p[0])
            else:                lo,hi,stride = p; d[dim] = lo + stride*rng.randint(0,(hi-lo)//stride)
        return {**op.fixed_kwargs, **d}

# ---------- ShapeSample.to_prompt (KernelBench contract) ----------
def to_prompt(self):
    op = OP_BANK_BY_NAME[self.op_name]
    ref_src = op.ref_src_template.format(**self.dims)   # concrete-shape Model + get_inputs
    return MULTITURN_INSTRUCTIONS + "\n```python\n" + ref_src + "\n```"
    # ref module example (fused_bias_gelu, M,N substituted):
    #   class Model(nn.Module):
    #       def __init__(self): super().__init__(); self.b = nn.Parameter(torch.randn({N}))
    #       def forward(self, x): return F.gelu(x + self.b)
    #   def get_inputs():      return [torch.randn({M},{N}, device='cuda', dtype=torch.float16)]
    #   def get_init_inputs(): return []
    # agent must output ModelNew with a Triton kernel + identical get_inputs/get_init_inputs

# ---------- env.py : two-yield template (REAL HUD v6 shape) ----------
env = Environment(name="kernelforge-v2")
@env.template(id="kernel_opt")
async def kernel_task(op_name, dims, dtype, split, rtol, atol):
    sample = ShapeSample(op_name, dims, dtype, split, seed=GLOBAL_SEED)
    answer = yield sample.to_prompt()                 # 1st yield: prompt (agent runs K turns via shell)
    yield grade_kernel(answer, sample, rtol, atol)    # 2nd yield: EvaluationResult (SEPARATE grader comp)

# ---------- tasks.py : mint the public taskset ----------
sampler = ShapeSampler(seed=GLOBAL_SEED)
_train = sampler.sample("train", n=N_TRAIN)           # feeds GRPO rollouts
_test  = sampler.sample("test",  n=N_TEST)            # held-out generalization eval (the money slide)
assert is_disjoint(_train, _test)
tasks = []
for s in (_train + _test):
    t = kernel_task(op_name=s.op_name, dims=s.dims, dtype=s.dtype,
                    split=s.split, rtol=OP_BANK_BY_NAME[s.op_name].rtol,
                    atol=OP_BANK_BY_NAME[s.op_name].atol)
    t.slug = s.task_id().replace("_","-")
    t.columns = s.to_columns()                        # {op,split,M,N,dtype,level} -> filterable
    tasks.append(t)

# ---------- manifest.py : pin the exact sets (daVinci JSONL-snapshot pattern) ----------
def freeze(seed, n_train, n_test, out_dir):
    sp = ShapeSampler(seed)
    tr, te = sp.sample("train", n_train), sp.sample("test", n_test)
    assert is_disjoint(tr, te)
    write_jsonl(out_dir/"shapes_train.jsonl", [asdict(s) for s in tr])
    write_jsonl(out_dir/"shapes_test.jsonl",  [asdict(s) for s in te])
```

**Risks:** Disjointness regression: a careless edit to TRAIN/TEST ranges could make a test shape coincide with a train shape, silently destroying the generalization claim. MITIGATION: _assert_split_disjoint() runs at import time AND in tests; freeze() re-asserts before writing. This is the single most load-bearing invariant.; 'Held-out' shapes may still be near the model's pretraining distribution (powers-of-two are everywhere in code corpora), weakening the 'reasoning over memorization' claim. MITIGATION: choose test shapes on the 128-stride OFF-grid points (384,768,1152...) that are unusual, and frame the result as DIRECTIONAL train-vs-test delta, not an absolute generalization proof (per audit reframing).; Too few base ops (5) means low task diversity -> GRPO can memorize per-op kernels rather than generalize across shapes. MITIGATION: the shape axis still gives 6x6=36+ shape points per op; emphasize the per-op held-out-shape curve, and keep op-graph composition as the explicitly-labeled stretch.; Rendered reference module may not import/run (bad template -> grader errors -> whole rollout 0). MITIGATION: test_sampler.py compiles every rendered prompt's Model and runs get_inputs() on CPU before any GRPO kick; fail the build if any op template is malformed.; Shape ranges that make a kernel too small (M*N tiny) yield noisy timing and unstable speedup -> reward variance collapse. MITIGATION: floor the ranges (>=384) and snap to stride 128 so every shape is large enough for stable do_bench timing; calibration gate catches residual cases.; Determinism break if hash() salting (PYTHONHASHSEED) differs between control plane and Modal workers -> train/test sets diverge across machines. MITIGATION: use hashlib.sha256 of a canonical string (not builtin hash()) inside _sample_dims RNG seeding, and always consume the FROZEN manifest JSONL in the loop rather than re-sampling.; Scope creep into op-graph/CSP-DAG during the window. MITIGATION: hard rule from audit — shape sampler only; op-graph is a slide, not code.

### GRPO training loop (trl GRPOTrainer + vLLM colocate, calibration gate, step-150 abort) for Protean  (~9h)
**Goal:** Take the already-green Protean HUD env (two-yield: yield prompt, yield EvaluationResult with the 4-layer anti-hack + PR reward) and an SFT/base Qwen2.5-Coder-7B, and run a single-GPU (one H100) multi-turn GRPO loop overnight that produces a measurable base-vs-trained delta on HELD-OUT continuous shapes by Sun 1PM. The loop must (1) NOT kick GRPO until a hard calibration gate passes (group reward variance present, 20-50% in-band), (2) drive the multi-turn Triton kernel env through trl's environment_factory so the env's own grader is the ONLY reward path (no LLM judge), (3) layer in the daVinci-kernel TRLOO/null-baseline and PR reward where they add signal-density within 24h, and (4) auto-abort at step 150 if the reward curve is flat so we fall back to the pre-recorded curve. Everything runs on Modal H100 with TRITON_CACHE_DIR baked into the image; trl+vLLM colocate avoids any Fireworks-preview dependency.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_modal.py` — Modal entrypoint. Lifts ml-template/modal_runner.py's @app.function(image=image, gpu='H100', timeout=86400, secrets=[...]) decorator block. image = from_dockerfile('Dockerfile.train') with TRITON_CACHE_DIR=/triton-cache baked + warmup pass; .add_local_dir('kernelforge'). Defines run_grpo() that calls train.grpo_loop.main(). Also defines calibration_smoke() (gpu='H100', timeout=3600) so the gate runs in the SAME image/GPU type as training.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_loop.py` — Core loop. Builds GRPOConfig (use_vllm=True, vllm_mode='colocate', num_generations=8, max_completion_length, gradient_accumulation_steps), instantiates KernelEnv factory + kernel_reward_func, wraps GRPOTrainer, installs the AbortCallback (step-150 rule) and CalibrationGateCallback (refuses to start if gate not green). main() = load model/LoRA -> run gate -> trainer.train().
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/kernel_env.py` — trl multi-turn environment adapter. Thin OpenEnv-style class wrapping the HUD two-yield env (kernelforge/env.py) for K<=3 turns: reset() samples a (op, shape) task from the curriculum/shape-sampler and returns the prompt; step(action) extracts the ```python triton kernel, calls the env grader (compile->allclose->time->4-layer guard->PR), feeds compile_err/allclose_diff/last_latency back as the next observation, tracks best_correct_reward across turns. Exposes .reward (best across turns) for the reward func.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/reward.py` — kernel_reward_func(environments, **kwargs)->list[float] (trl signature). Returns env.reward per rollout. Also houses compute_turn_reward() = daVinci Eq1 R=C(y)*(1+speedup+PR) with the speedup floor + partial-compile shaping, and the discounted multi-turn return G (gamma=0.4 Kevin / configurable). The HARD anti-hack reward math itself lives in kernelforge/grader.py (reused by env at grade time); reward.py only aggregates turns.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/callbacks.py` — TrainerCallback subclasses: CalibrationGateCallback (on_train_begin: asserts the gate JSON written by calibrate.py is GREEN, else raise SystemExit), AbortCallback (on_log: tracks reward EMA; at global_step>=150 if slope<eps over last 50 steps -> trainer.control.should_training_stop=True and write ABORTED sentinel), RewardCurveLogger (dumps train-shape vs held-out-shape reward to curve.jsonl for the money slide).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — Pre-GRPO calibration gate (standalone, runs via Modal calibration_smoke()). Rolls the BASE model 10x over each of the ~5 narrowest L1 tasks via vLLM offline, scores each with the real grader, computes per-group: %compilable, %allclose, reward std, %in 20-50% band. Writes gate.json {green: bool, per_task_stats}. GREEN requires >=5% compilable AND >=2% allclose AND >=1 task group with std>0.05 AND median reward in (0,1).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/shape_sampler.py` — Parameterized SHAPE sampler + train/test split (the moat). sample_task(split) over ~5 fixed ops; train M in {256,512,1024,2048}, held-out M in {384,768,1536} (disjoint continuous bands). Used by kernel_env.reset() and by RewardCurveLogger to interleave held-out eval rollouts.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/configs.py` — All hyperparams in one dataclass (group/num_generations=8, gamma=0.4, K_turns=3, lr, lora_r, max_completion_length, abort_step=150, abort_window=50, abort_eps, P_TARGET=1.5, speedup_floor=1.1, partial_compile_credit=0.1, davinci_null_baseline=True, pr_weight). Single source of truth shared by calibrate/train/reward.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/Dockerfile.train` — Training image: CUDA + torch + triton + vllm + trl + peft, ENV TRITON_CACHE_DIR=/triton-cache, a build-time RUN that compiles one warmup Triton kernel to populate the cache (kills 30-120s JIT cold-start per rollout). Mirrors the TRITON_CACHE bake called out in the phase14 audit.

**APIs:** GRPOConfig(use_vllm=True, vllm_mode='colocate', num_generations, max_completion_length, gradient_accumulation_steps, learning_rate, max_steps, beta, scale_rewards, loss_type='grpo', log_completions, save_steps); GRPOTrainer(model, reward_funcs=kernel_reward_func, environment_factory=lambda: KernelEnv(...), args, train_dataset, callbacks); reward_funcs signature: kernel_reward_func(environments: list[KernelEnv], **kwargs) -> list[float]; transformers.TrainerCallback.on_train_begin / on_log / on_step_end; control.should_training_stop; vllm.LLM(model).generate(prompts, SamplingParams(temperature=1.0, n=10)) for the offline calibration gate; modal.App.function(image=image, gpu='H100', timeout=86400, secrets=[modal.Secret.from_name('hud-keys', required_keys=['HUD_API_KEY'])]); modal.Image.from_dockerfile('Dockerfile.train').add_local_dir('kernelforge', remote_path='/kernelforge'); HUD env contract reused at grade time: @env.template async gen -> yield prompt -> yield EvaluationResult(reward, done, info, subscores) (from hud.graders); EvaluationResult built by hand, NOT via combine(), to preserve the hard cap (verilog grader.py pattern); torch.cuda.Event(enable_timing=True) elapsed_time for median latency; torch.profiler for PR kernel-time attribution; triton.JITFunction launch instrumentation for layer-2

**daVinci integration:** Integrated selectively for 24h-solo signal density, NOT wholesale (the 3-agent/skill-library system is out of scope for one night). PR REWARD TERM (Eq1 / Eq4): grader.py computes pr = profiled fraction of measured GPU kernel-time spent in the agent's Triton kernels (torch.profiler attribution) and HARD-gates reward=0 below PR_FLOOR. This directly kills the 'Triton fires on a trivial sub-op while cuBLAS does the heavy work' exploit and is the headline 'we integrated this weekend's SotA' line. TURN REWARD shape: compute_turn_reward implements daVinci Eq1 R = C(y)*(1+speedup+PR) as the per-turn signal feeding the multi-turn return (gamma=0.4, Kevin-style), giving denser credit than a single terminal scalar. TRLOO / NULL-BASELINE: trl's GRPO already does group-relative (leave-one-out-like) advantages via scale_rewards over num_generations=8 — that is the buildable analog of daVinci Eq7 LOO. If time allows (stretch, +1h), inject a 'no-skill' null scheme into the group to mirror daVinci's (k+1)*n grouping so trained kernels are advantaged only when they beat the unconditioned baseline. MRS/PRS (Eq3/Eq4): PRS profiling-aware sampling is approximated by curriculum-weighting toward ops where PR is informative; full importance-ratio MRS filtering is explicitly deferred (needs train-vs-rollout logprob deltas trl does not surface cleanly in one night). SKILL LIBRARY / Selection+Summary agents (Eq5/6/8/9): OUT OF SCOPE for the overnight loop — single Policy Agent only. Anti-hack LAZY-SKILL filter maps onto our AST-ban layer (reject torch-fallback). Net: take PR reward + turn-reward shaping + null-baseline (the parts that raise signal density and close the exploit in hours); skip the skill-evolution machinery (the part that needs days and a shared backbone).

```python

# ============ grader.py : the ONLY reward path (4-layer anti-hack + daVinci PR) ============
# (lives in env build, imported by kernel_env.step; mirrors verilog grader.py fail-closed style)
def grade_kernel(kernel_src, op, M, N, dtype):
    info = {"hard_caps": []}
    # LAYER 1 (AST ban): reject torch fallback / @torch.compile / aten passthrough
    if ast_has_banned(kernel_src, {"torch.matmul","F.softmax","torch.compile",...}):
        return {**info, "reward":0.0, "hard_caps":["ast_ban"]}
    try:
        mod = compile_and_import(kernel_src)            # Triton JIT (TRITON_CACHE warm)
    except Exception as e:
        # partial-compile shaping ONLY during training to manufacture variance (annealed out)
        return {**info, "compiled":False, "reward":SETTINGS.partial_compile_credit*0.0}
    x = [rand(shape, dtype, device='cuda') for _ in range(N_INPUTS)]   # random inputs
    ref = REF_FNS[op](*x)                               # PyTorch eager reference
    # LAYER 2 (launch counter): wrap triton.JITFunction to count real GPU launches
    with launch_counter() as lc:
        out = mod.entry(*x)
    if lc.triton_launches == 0:                         # never-launched hack -> 0
        return {**info,"reward":0.0,"hard_caps":["no_triton_launch"]}
    # LAYER 3 (dtype match): bf16-downcast hack -> 0
    if out.dtype != ref.dtype:
        return {**info,"reward":0.0,"hard_caps":["dtype_mismatch"]}
    if not torch.allclose(out, ref, rtol, atol):
        return {**info,"reward":0.0,"hard_caps":["not_allclose"]}
    # timed pass: warmup + L2 flush + CUDA events, median of M runs
    t_kernel = timed_median(lambda: mod.entry(*x))      # re-launch under timer
    t_eager  = timed_median(lambda: REF_FNS[op](*x))
    # re-check allclose AFTER timing (catch async/overwrite tricks)
    if not torch.allclose(mod.entry(*x), ref, rtol, atol):
        return {**info,"reward":0.0,"hard_caps":["post_timing_mismatch"]}
    speedup = t_eager / t_kernel
    # LAYER 4 (daVinci PR gate): fraction of measured GPU kernel-time in the agent's triton kernels
    pr = profiled_triton_fraction(op, x, mod.entry)     # via torch.profiler kernel attribution
    if pr < PR_FLOOR:                                   # cuBLAS-does-the-work exploit -> 0
        return {**info,"reward":0.0,"hard_caps":["low_profiling_ratio"]}
    if speedup < SETTINGS.speedup_floor:               # 1.1x floor
        speedup = 0.0                                   # correct-but-not-faster -> correctness floor only
    # bounded correctness x speedup reward (phase13 form), PR folded in (daVinci Eq1 spirit)
    base = 0.3 + 0.7*min(speedup/SETTINGS.P_TARGET, 1.0)  # 0.3 correctness floor keeps variance alive
    reward = base * (1.0 if pr>=PR_FLOOR else 0.0)
    return {**info,"compiled":True,"allclose":True,"speedup":speedup,"pr":pr,"reward":reward}

# ============ kernel_env.py : multi-turn loop trl drives ============
class KernelEnv:
    def reset(self):
        self.task = sample_task(self.split, self.rng)
        self.turn = 0; self.turn_rewards=[]; self._best=0.0
        return build_prompt(self.task)                  # op + shape + "write a Triton kernel; you have K turns"
    def step(self, action):
        self.turn += 1
        src = extract_code_block(action)
        g = grade_kernel(src, **self.task)              # the ONLY reward path
        r = compute_turn_reward(g, SETTINGS)            # daVinci Eq1: C*(1+speedup+PR)
        self.turn_rewards.append(r)
        if g.get("allclose"): self._best = max(self._best, g["reward"])
        done = (self.turn >= SETTINGS.K_turns) or (g.get("speedup",0) >= SETTINGS.P_TARGET)
        # OBSERVATION = state the agent revises against (last error + last latency + diff)
        obs = format_feedback(compiled=g.get("compiled"), err=g.get("err"),
                              allclose=g.get("allclose"), speedup=g.get("speedup"),
                              pr=g.get("pr"), hard_caps=g["hard_caps"])
        return obs, r, done, g
    @property
    def reward(self):                                   # scalar GRPO sees per rollout
        # best CORRECT reward across turns (Kevin-style "best across turns"),
        # optionally discounted return for denser turn-credit:
        return self._best  # or discounted_return(self.turn_rewards, SETTINGS.gamma)

# ============ calibrate.py : HARD gate, runs BEFORE any GRPO ============
def run_gate(s):
    llm = vllm.LLM(s.model)                             # offline batched gen, base weights
    stats={}
    for task in NARROW_L1_TASKS[:5]:
        outs = llm.generate([task.prompt]*10, temp=1.0)
        rs=[grade_kernel(extract_code_block(o), **task)["reward"] for o in outs]
        comp=[...]; allc=[...]
        stats[task.id]={"pct_compile":mean(comp),"pct_allclose":mean(allc),
                        "std":pstdev(rs),"median":median(rs),
                        "in_band":mean(0.2<=r<=0.5 for r in rs)}
    green = (any(t["pct_compile"]>=0.05 for t in stats.values())
             and any(t["pct_allclose"]>=0.02 for t in stats.values())
             and any(t["std"]>0.05 for t in stats.values())
             and any(0<t["median"]<1 for t in stats.values()))
    write_json("gate.json", {"green":green, "stats":stats})
    return {"green":green,**stats}
# GUARANTEE pass if base too sparse: start curriculum at single elementwise op,
# enable partial_compile_credit (+0.1) and 0.3 correctness floor -> manufactures variance.

# ============ callbacks.py : abort + curve ============
class AbortCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        r = logs.get("reward");  self.hist.append((state.global_step, r))
        if state.global_step >= SETTINGS.abort_step:
            window=[x for x in self.hist if x[0]>=state.global_step-SETTINGS.abort_window]
            if slope(window) < SETTINGS.abort_eps:     # flat
                write_file("ABORTED"); control.should_training_stop=True
                log("step-150 abort: flat reward -> use pre-recorded curve")

# ============ grpo_loop.py : assembly ============
def build_trainer(s):
    cfg = GRPOConfig(use_vllm=True, vllm_mode="colocate",
                     num_generations=s.num_generations,           # group=8
                     max_completion_length=s.max_completion_length,
                     gradient_accumulation_steps=s.grad_accum,
                     learning_rate=s.lr, max_steps=s.max_steps,
                     beta=0.0,                                     # no KL (DAPO-ish), or small
                     scale_rewards=True,                          # group-relative std norm (GRPO LOO-like)
                     loss_type="grpo", log_completions=True, save_steps=10)
    # dataset is just prompts; the env_factory provides the real multi-turn rollout+reward
    ds = Dataset.from_dict({"prompt": [[{"role":"user","content":""}]]*s.max_steps*64})
    return GRPOTrainer(model=load_peft(s.model, s.lora_r),
                       reward_funcs=kernel_reward_func,
                       environment_factory=lambda: KernelEnv(s, split="train"),
                       args=cfg, train_dataset=ds,
                       callbacks=[CalibrationGateCallback(), AbortCallback(), RewardCurveLogger()])
def main(s=None):
    s = s or GRPOSettings()
    if not run_gate(s)["green"]:                        # gate is a HARD precondition
        raise SystemExit("calibration RED: do NOT kick GRPO")
    build_trainer(s).train()                            # overnight; auto-stops at abort or max_steps

```

**Risks:** trl's environment_factory multi-turn API is younger/less stable than single-turn reward_funcs; if it doesn't drive K-turn rollouts cleanly, FALLBACK to single-turn GRPO (yield prompt -> one completion -> grade) which trl supports rock-solid via reward_funcs alone. Multi-turn becomes a stretch, not a blocker. Budget 1h to validate the factory path Sat night; if red by then, ship single-turn.; vLLM colocate + 7B policy + LoRA + Triton-compiling rollouts on ONE 80GB H100 can OOM or thrash; mitigate with LoRA (not full FT), modest max_completion_length, vllm gpu_memory_utilization tuned down, and gradient_accumulation instead of large batch. If still tight, drop to Qwen2.5-Coder-3B or reduce num_generations to 4 (trl 2-GRPO shows small groups still work).; Throughput: each rollout = JIT compile + allclose + timed bench (+profiler for PR). Even with TRITON_CACHE warm, PR profiling adds per-rollout cost; gate PR profiling to the timed pass only and cache per-(op,shape) eager baselines. Realistic ~30-100 GRPO steps overnight — step-150 abort may never trigger simply because we don't reach 150; treat abort as 'flat-at-whatever-step-we-reach' and pre-record the curve by 6AM regardless (audit mandate).; GRPO advantage collapse if every rollout in a group scores 0 (all-zero -> zero gradient). The calibration gate + 0.3 correctness floor + partial_compile_credit + easy-op curriculum start are the explicit defenses; if a group is all-zero at runtime, it simply contributes no gradient (not a crash), but persistent all-zero = no learning. Gate is the go/no-go.; Reward hacking inside the timed pass (async overwrite, cached output) — mitigated by post-timing allclose re-check and launch counter, but a determined exploit could still slip; red-team grader.py with passthrough/never-launched/bf16 kernels BEFORE the kick (asserts reward~0), per the audit's hour 4-6 plan. This is grader work, upstream of this loop, but the loop's integrity depends on it.; Held-out generalization curve (the money slide) requires interleaved eval rollouts on HELDOUT_M shapes during training; if RewardCurveLogger eval rollouts steal too much GPU time, run them sparsely (every 25 steps) or compute the base-vs-trained held-out delta as a separate post-hoc eval pass after training, not inline.

### daVinci skill-library-lite: BM25 retrieve + LLM rerank (select_skills) + Summary-Agent skill distillation with execution verification, persisted as versioned JSONL snapshots and injected into the Policy Agent's first turn. A standalone Python package (skilllib/) the Protean HUD env imports; decoupled from RL so it works whether the policy is trained or base.  (~9h)
**Goal:** Give the Qwen2.5-Coder-7B policy a growing, execution-verified library of Triton optimization techniques that are retrieved (BM25 top-20 -> LLM rerank <=3) and injected into context before each task, and that grows from successful rollouts via a Summary Agent gated by daVinci Eq5/Eq6 verification. The lite scope: selection and summary run as FIXED-PROMPT LLM calls (no joint RL of the selection/summary heads, i.e. skip Eq7-9), so the library co-evolves with the policy OFFLINE between training steps rather than inside the GRPO loss. This is the highest-leverage daVinci technique buildable solo in 24h: it lifts both base and trained deltas, is the generalization 'moat' multiplier (good skills transfer to held-out continuous shapes), and degrades gracefully (empty library == null scheme == plain Protean).

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/__init__.py` — Public API re-exports: Skill, SkillLibrary, BM25Index, select_skills, summarize_rollout, verify_skill, LAZY_PATTERNS. One import surface for env.py and the trainer.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/schema.py` — Skill dataclass (daVinci 3.2 five fields: name/description/scope/tags/content) + (de)serialization to a single JSONL line; validation (snake_case name, non-empty content, scope in enum).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/library.py` — SkillLibrary: load/save versioned JSONL snapshots keyed by training step (snapshots/skills_step{N}.jsonl), staging cache L_cache, atomic flush+broadcast, one-skill-per-task growth rule, dedup by name.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/retrieve.py` — BM25Index over name+description+tags+content (rank_bm25.BM25Okapi); lazy build after load / rebuild after flush; top-N candidates by score with a min-score threshold (empty -> no skills).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/select.py` — Selection Agent: select_skills tool schema + two-stage retrieve(task_desc)->BM25 top-20->LLM rerank to <=3 names. LLM call via vLLM OpenAI-compatible endpoint (same backbone). Falls back to top-k-by-BM25 if LLM/tool-parse fails.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/summarize.py` — Summary Agent: trigger check (Eq5 R*>alpha*r1 and R*>beta), update_skill_library tool, lazy-skill regex filter, then verify_skill (Eq6) by re-running the policy on the original task with the candidate injected; returns verified skills for staging.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/inject.py` — Render selected skills' name+content into the policy's first-turn system/user prompt block (daVinci 3.4: only name+content injected). Deterministic ordering + token budget cap.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/snapshots/skills_step0.jsonl` — Seed library: 8-15 hand-written Triton skills (coalesced loads, autotune BLOCK sizes, fused epilogue, masking for non-power-of-2, fp16 accum-in-fp32, tl.dot tiling, vectorized loads, grid heuristics). Cold-start so the base run is non-trivial Saturday afternoon.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/prompts.py` — Fixed system prompts for Selection Agent and Summary Agent (verbatim daVinci-style instructions: select<=3, return think+names+reason; summarize into one transferable skill, reject pytorch-fallback).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/skilllib/tests/test_skilllib.py` — Unit tests: JSONL round-trip, BM25 retrieval determinism, lazy-filter rejects 'fall back to torch'/'torch.compile' skills, trigger math Eq5, verify_skill gating Eq6, empty-library no-op path.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/protean/env.py` — EXISTING Protean HUD env (forked from coding-template pattern). EDIT: in the @env.template generator, call select_skills before the first yield (inject skills into prompt) and write rollout metadata so the offline summarizer can pick up successes.

**APIs:** Skill(name:str, description:str, scope:str, tags:list[str], content:str, step_added:int=0, verify_speedup:float=0.0); Skill.to_jsonl()->str ; Skill.from_jsonl(line:str)->Skill ; Skill.doc()->str; SkillLibrary(snap_dir:str) ; .load(step:int|None=None)->None ; .stage(skill:Skill)->bool ; .flush(new_step:int)->str(path) ; .skills:dict[str,Skill]; BM25Index(skills:Iterable[Skill]) ; .topk(query:str, k:int=20, min_score:float=0.1)->list[Skill]; select_skills(task_desc:str, lib:SkillLibrary, llm, k_select:int=3, n_bm25:int=20)->list[Skill]; inject(base_prompt:str, skills:list[Skill], max_tokens:int=2500)->str; triggered(turn_speedups:list[float], alpha:float=1.2, beta:float=1.2)->bool; summarize_rollout(rollout:RolloutMeta, lib:SkillLibrary, llm, policy_runner, S_max:int=1)->list[Skill]; verify_skill(task, candidate:Skill, base_skills:list[Skill], r1:float, policy_runner, alpha=1.2, beta=1.2)->bool; LAZY_PATTERNS: re.Pattern  # lazy-skill filter; RolloutMeta(task, ref_src, injected_skills, turn_speedups, first_turn_code, best_turn_code)  # bridge from grader info -> summarizer

**daVinci integration:** Directly ports daVinci sections 3.2-3.7, lite-scoped for 24h solo. 3.2 Skill Library: exact five-field schema (name/description/scope/tags/content) persisted as versioned JSONL snapshots keyed by step (skills_step{N}.jsonl) for exact checkpoint restart. 3.3 Selection Agent: two-stage BM25 top-20 (rank_bm25.BM25Okapi over concatenated fields, lazy-built/rebuilt-after-flush) then LLM rerank to k_select<=3 via the select_skills tool with think+name+reason; empty/low-score -> empty selection == daVinci null scheme c=∅. 3.4 Policy Agent: inject ONLY name+content into the first turn; c=∅ passes prompt through unchanged. 3.5 Summary Agent: Eq5 trigger (R*>alpha*r1 AND R*>beta, alpha=beta=1.2), update_skill_library tool with S_max=1, lazy-skill regex filter (rejects torch.compile / fallback-to-pytorch), and Eq6 execution verification (re-run policy first-turn with candidate injected, accept iff r_verify>=max(beta, alpha*r1)). 3.7 library update: one-skill-per-task growth, atomic flush of L_cache to a new versioned snapshot, broadcast to rollout workers (Modal volume put / rsync) so every subsequent step benefits. DELIBERATE CUTS vs full daVinci: (1) selection/summary heads are NOT jointly RL-trained -> skip the per-agent LOO advantages Eq7/Eq8/Eq9 and the combined loss Eq10; they run as fixed-prompt calls on the same backbone, so the library co-evolves OFFLINE between GRPO steps. (2) No SFT cold-start of the three agents (3.6) -> use a hand-written 8-15 skill seed snapshot instead. (3) MRS/PRS sampling (Eq3/Eq4) live in the trainer, not here. This keeps the moat-relevant pieces (verified, transferable skills + retrieval) while dropping the multi-agent-RL machinery that cannot land solo in 24h. Anti-hack retained: lazy-skill filter + execution-verification are the two cheapest, highest-value daVinci anti-hacks and both live in this component.

```python

# ===== schema.py =====
SCOPES = {"elementwise","reduction","matmul","conv","norm","attention","fusion","generic"}
@dataclass
class Skill:
    name: str          # snake_case unique key
    description: str   # one sentence (shown to selector)
    scope: str         # in SCOPES
    tags: list[str]
    content: str       # markdown w/ Triton code (injected into policy)
    step_added: int = 0
    verify_speedup: float = 0.0
    def to_jsonl(self)->str: return json.dumps(asdict(self))
    @staticmethod
    def from_jsonl(line)->"Skill": return Skill(**json.loads(line))
    def doc(self)->str: return f"{self.name} {self.description} {' '.join(self.tags)} {self.content}"  # BM25 doc

# ===== library.py =====
class SkillLibrary:
    def __init__(self, snap_dir): self.skills={}; self.cache={}; self.step=0; self.snap_dir=snap_dir
    def load(self, step=None):           # load latest or specific snapshot
        path = latest_or(self.snap_dir, step); self.skills = {s.name:s for s in map(Skill.from_jsonl, open(path))}
        self._index = BM25Index(self.skills.values())   # lazy build (3.3)
    def stage(self, skill):              # L_cache, one-per-task enforced by caller
        if skill.name in self.skills: return False
        self.cache[skill.name]=skill; return True
    def flush(self, new_step):           # 3.7 atomic flush + broadcast
        self.skills.update(self.cache); self.cache={}; self.step=new_step
        tmp=snap_path(new_step)+".tmp"; write_all_jsonl(tmp, self.skills.values()); os.replace(tmp, snap_path(new_step))
        self._index = BM25Index(self.skills.values())   # rebuild after flush
        broadcast(snap_path(new_step))   # rsync/Modal volume put -> rollout workers

# ===== retrieve.py =====
class BM25Index:
    def __init__(self, skills): self.skills=list(skills); self.bm25=BM25Okapi([tok(s.doc()) for s in self.skills])
    def topk(self, query, k=20, min_score=0.1):
        if not self.skills: return []
        scores=self.bm25.get_scores(tok(query)); ranked=sorted(zip(scores,self.skills),reverse=True)
        return [s for sc,s in ranked[:k] if sc>=min_score]

# ===== select.py =====  (daVinci 3.3 two-stage, k_select<=3)
SELECT_TOOL = {"name":"select_skills","parameters":{"think":"str","skills":[{"name":"str","reason":"str"}]}}
def select_skills(task_desc, lib, llm, k_select=3, n_bm25=20)->list[Skill]:
    cands = lib._index.topk(task_desc, k=n_bm25)
    if not cands: return []                              # empty lib -> null scheme (3.4 c=∅)
    menu = "\n".join(f"- {s.name}: {s.description}" for s in cands)
    msg = SELECT_SYS_PROMPT + f"\nTASK:\n{task_desc}\nSKILLS:\n{menu}"
    resp = llm.chat(msg, tools=[SELECT_TOOL], temperature=0.0)
    try: chosen=[c["name"] for c in parse_tool_call(resp)["skills"]][:k_select]
    except: chosen=[s.name for s in cands[:k_select]]    # robust fallback
    return [lib.skills[n] for n in chosen if n in lib.skills]

# ===== inject.py =====  (daVinci 3.4: inject name+content only)
def inject(base_prompt, skills, max_tokens=2500)->str:
    if not skills: return base_prompt
    blk="\n\n".join(f"### Skill: {s.name}\n{s.content}" for s in skills)
    blk=truncate_tokens(blk, max_tokens)
    return base_prompt + "\n\nYou MAY use these verified Triton optimization techniques:\n"+blk

# ===== summarize.py =====  (daVinci 3.5 Eq5 trigger, lazy filter, Eq6 verify)
LAZY = re.compile(r"torch\.compile|fall ?back to (py)?torch|torch\.\w+\(|use torch|aten::", re.I)
def triggered(turn_speedups, alpha=1.2, beta=1.2)->bool:
    R_star=max(turn_speedups); r1=turn_speedups[0]; return R_star>alpha*r1 and R_star>beta
def summarize_rollout(rollout, lib, llm, policy_runner, S_max=1)->list[Skill]:
    if not triggered(rollout.turn_speedups): return []
    ctx = SUMMARY_SYS_PROMPT + render(rollout.task, rollout.injected_skills,
                                      rollout.first_turn_code, rollout.best_turn_code)
    resp = llm.chat(ctx, tools=[UPDATE_TOOL], temperature=0.7)
    props = parse_tool_call(resp)["skills"][:S_max]
    out=[]
    for p in props:
        sk=Skill(**p, scope=infer_scope(rollout.task), step_added=lib.step+1)
        if LAZY.search(sk.name+sk.description+" ".join(sk.tags)+sk.content): continue  # lazy-skill filter
        r1=rollout.turn_speedups[0]
        r_verify = policy_runner.first_turn_speedup(rollout.task, inject_skills=rollout.injected_skills+[sk])
        if r_verify >= max(beta, alpha*r1):              # Eq6 execution verification
            sk.verify_speedup=r_verify; out.append(sk)
    return sorted(out, key=lambda s:-s.verify_speedup)[:1]   # one-skill-per-task (3.7)

# ===== env.py edit (HUD two-yield, real API) =====
@env.template(id="protean")
async def kernel_forge(task_id, ref_src, eval_shapes, train_shapes, validate_mode=None):
    setup_task(task_id, validate_mode)
    lib = SkillLibrary(SNAP_DIR); lib.load()                       # latest broadcast snapshot
    skills = select_skills(make_task_desc(ref_src), lib, SEL_LLM)  # BEFORE first yield
    prompt = inject(make_prompt(ref_src, train_shapes), skills)
    answer = yield prompt                                          # policy writes Triton (multi-turn in harness)
    ev = grade_kernel(task_id, ref_src, eval_shapes)               # allclose x measured speedup on HELD-OUT shapes
    ev.info = {**(ev.info or {}), "injected_skills":[s.name for s in skills],
               "turn_speedups": ev.info.get("turn_speedups",[]), "task_id":task_id, "ref_src":ref_src}
    yield ev    # EvaluationResult(reward=C*speedup, subscores=[correctness, speedup, PR])
# Offline (between GRPO steps, NOT in env): scan rollouts -> summarize_rollout -> lib.stage -> lib.flush(step) -> broadcast

```

**Risks:** LLM rerank/tool-call parsing is brittle on a 7B model -> MITIGATED: temperature=0.0, strict JSON tool schema, and BM25-top-k fallback on any parse failure so selection never crashes a rollout.; verify_skill re-runs the policy (extra GPU time) for EVERY candidate -> at S_max=1 and summary triggered only on strong rollouts this is bounded, but if many tasks trigger it can steal training GPU. Cap: run summarization on at most ~20 best rollouts per step, time-boxed; skip if behind schedule (library still works, just stops growing).; Empty/weak seed library means the base-vs-trained delta could be mostly the library, not the policy -> report base WITH and WITHOUT library so the delta attribution is honest (phase14 'directional deltas').; Skill injection eats context window of a 7B model (4-8k practical) -> max_tokens=2500 cap on the injected block + <=3 skills; truncate content, keep code blocks.; Snapshot broadcast race: a worker reads a half-written file -> MITIGATED by write-to-.tmp + os.replace (atomic) and step-keyed filenames; workers load latest fully-written step.; Lazy-skill regex is heuristic and can miss obfuscated pytorch fallbacks -> execution-verification (Eq6) is the real backstop; the regex is just a cheap first pass.; SCHEDULE: this is additive to the core env+grader+training path. If Saturday slips, ship with the seed snapshot + selection+inject only (drop summarize/verify) -- still a real daVinci retrieval moat, ~4h instead of ~9h.; rank_bm25 tokenization is naive whitespace -> fine for code-ish docs; if recall is poor, add simple identifier-splitting in tok(), but do not over-invest.

## Round 2 — daVinci Integration (KEEP/CUT/LITE)


### Protean — daVinci Agent Architecture: KEEP/CUT/LITE Decisions + Exact Data Flow  (~2h)
**Goal:** Decide which of daVinci's three agents (Selection, Policy, Summary) to integrate into the 24h solo Protean build, with what scope, in what implementation form, and at what hour cost — grounded in the paper's own ablation numbers, the verified trl environment_factory API, the verilog-template scaffold, and the revised 13.5h serial spine from Round-1.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/skilllib/schema.py` — Skill dataclass: name/description/scope/tags/content/step_added/verify_speedup — verbatim daVinci Section 3.2 five-field schema, persisted as JSONL
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/skilllib/snapshots/skills_step0.jsonl` — 5 hand-authored seed skills (tiled_matmul, vectorized_load, shared_mem_reduction, persistent_kernel, fused_elementwise); the ONLY library content on night-1
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/skilllib/retrieve.py` — BM25 index over name+description+tags+content; rank_bm25 (pure-Python, no conflict risk); top-3 by score. Rebuilt lazily on each library load.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/skilllib/inject.py` — Prepend <=3 skill content blocks (capped at 2500 tokens total) into system prompt before first policy turn. Degrades to null-scheme if library empty.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/rewards.py` — THE single canonical reward formula imported by BOTH grade.py and kernel_env.py. R = 0 if gates fail else clip(0.3 + speedup/P_TARGET + PR_BONUS, 0.0, 2.0). Never duplicated.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/env.py` — HUD two-yield @env.template; no `from __future__ import annotations`; _AgentWorkspace uid-wall copied from verilog-template verbatim; single-turn primary path
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tasks/elementwise/donotaccess/grade.py` — Hidden grader for fused-elementwise op; imports rewards.py; 3-seed correctness + binary PR gate + launch counter; verilog run() fail-closed subprocess pattern
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/kernel_env.py` — Single-turn GRPO environment class; imports rewards.py; calls grade.py via grader.evaluate_kernel(); exposes submit_kernel tool for environment_factory path
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_loop.py` — trl GRPOTrainer with reward_funcs (primary/stable path); environment_factory wired as stretch-only branch activated by --multi-turn flag
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — Pre-GRPO gate: base model 10 rollouts/task, assert >=5% compilable + >=2% allclose + nonzero group variance + 20-50% in-band. Hard GO/NO-GO before kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scripts/redteam.py` — Saturday-night red-team checklist: assert reward~0 for passthrough/never-launched/try-except/bf16-downcast/import-laundering/delegating-wrapper kernels
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/splits.py` — Frozen disjoint train/test shape split; train M in {256,512,1024,2048}; test M in {400,800,1600,383,769,1537} (off-grid); _assert_split_disjoint() at import time
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/modal_app.py` — Modal H100 image build + @modal.enter() clock-lock + TRITON_CACHE warmup + smoke_test function; lifted from ml-template/modal_runner.py @app.function(gpu=H100) pattern

**APIs:** rank_bm25.BM25Okapi([tok(s) for s in corpus]) — pure-Python BM25, no LLM call, no parse-failure risk on 7B; triton.runtime.jit.JITFunction.__call__ monkeypatch — launch counter; patched at Python boundary not CudaDriver.launch internal (version-agnostic); verified by smoke_test asserting known-good kernel > 0 launches; trl GRPOTrainer(reward_funcs=grade_kernel) — PRIMARY stable path; single-turn; ~30 lines; trl GRPOTrainer(environment_factory=KernelEnv) — STRETCH multi-turn; local class pattern (no OpenEnv WebSocket server); activated only by --multi-turn flag after green curve; hud.graders.EvaluationResult + SubScore — hand-built (NOT combine()); hard_cap_penalty SubScore reconciliation verbatim from verilog grader.py L72-85; modal.Image.from_dockerfile + @app.function(gpu='H100', timeout=86400) — lifted from ml-template/modal_runner.py L88-93; subprocess.Popen + os.killpg(SIGKILL) fail-closed runner — verbatim from verilog repair grade.py L73-109; hashlib.sha256 for deterministic RNG seeding across machines (NOT builtin hash(), which varies by PYTHONHASHSEED)

**daVinci integration:** KEEP Policy Agent (mandatory, the RL subject): single-turn primary, multi-turn stretch via environment_factory local class. No changes to its generation path.

LITE Selection Agent (BM25-only, no LLM rerank): daVinci Section 3.3 Stage-1 only. Grounding: paper Ablation 4 (BM25 + no LLM rerank) achieves 51.2% L2 Fast1 for 8B, beating full daVinci at Fast1 but losing at Fast1.2. For 60-120 overnight steps, the Fast1.2 divergence has not opened, so BM25-only captures the dominant early-training value. LLM rerank (Stage 2) is deferred: it adds tool-call JSON parsing on a 7B model (fragile, temperature-sensitive) and a second model forward pass per rollout (throughput cost). BM25 fallback on any failure = zero added risk to rollouts. Implementation: rank_bm25.BM25Okapi over concatenated name+description+tags+content fields, lazily rebuilt on library load. Returns top-3 Skill objects. Cost: 1.5h total for retrieve + inject + seed JSONL.

CUT Summary Agent: daVinci Section 3.5 + Eq 5/6/9. Grounding: paper Ablation 3 (policy RL only, no joint summary/selection training) is competitive with full daVinci through roughly step 100 on both 8B and 14B curves before diverging at higher thresholds. Our overnight run targets 60-120 steps. The divergence window never opens in 24h. More critically: verify_skill re-runs the policy on the original task with the candidate skill injected (Eq 6), costing one full rollout per candidate. At s=2 summary calls per triggered task, this steals GPU time from the GRPO step budget. Static seed library avoids this cost entirely. Honest demo claim: 'static seed library; online summarization is the roadmap, and the paper proves it matters at step 200+ for Fast1.2 precision.'

PR Reward (daVinci Eq 1/4): KEEP as binary gate. Continuous torch.profiler PR cut (50-200ms overhead, segfault risk, fail-closed collapses whole groups to zero). Binary PR_BONUS=0.2 when (launches_timed > 0 AND not delegates_to_matrix_unit(src, op)) is deterministic, adds zero per-rollout overhead, closes the delegating-wrapper exploit, and is the single most defensible daVinci integration for a judge audience (citable to the June 15 paper, closes a named exploit).

Skill schema (daVinci Section 3.2): KEEP verbatim five fields (name, description, scope, tags, content) as JSONL snapshots keyed by step. This is the exact persistence format from the paper; step_added=0 for all seed skills.

Skill injection (daVinci Section 3.4): KEEP verbatim. content + name fields injected into system prompt before first policy turn. When c=empty (library empty or BM25 returns nothing), prompt passes unmodified — exact null-scheme from paper Section 3.4 last paragraph. 2500-token hard cap protects 7B context window.

SFT cold start diversity filter (daVinci Section 3.6 Qwen3-Embedding-8B + HDBSCAN): CUT entirely. Loading a second 8B embedding model on the 80GB H100 competes with the policy + vLLM. The seed library is hand-authored (5 skills), providing manual diversity with zero compute cost.

```python

# ============================================================
# DECISION TABLE (grounded in paper ablations + 24h constraint)
# ============================================================
#
# AGENT          | DECISION   | JUSTIFICATION (paper-grounded)
# -------------- | ---------- | ------------------------------
# Policy Agent   | KEEP-FULL  | Mandatory. Is the RL subject.
#                |            | Single-turn primary; multi-turn
#                |            | stretch via environment_factory.
# Selection Agent| LITE       | Ablation 4 (BM25-only, no LLM
#                |            | rerank) still beats Dr.Kernel
#                |            | at Fast1 but collapses at
#                |            | Fast1.2+. For 24h, BM25-only
#                |            | is the CORRECT lite scope:
#                |            | captures most of the Fast1
#                |            | signal, honest about the Fast1.2
#                |            | gap, and costs ~1h to build.
#                |            | LLM rerank = stretch (adds tool-
#                |            | call parsing fragility on 7B).
# Summary Agent  | CUT        | Ablation 3 shows policy-RL-only
#                |            | is competitive through ~step 100
#                |            | before diverging; for 60-120
#                |            | GRPO steps overnight, the
#                |            | divergence window never opens.
#                |            | verify_skill re-runs cost GPU
#                |            | time that reduces step count.
#                |            | Cut entirely; seed library
#                |            | remains static. Honest claim:
#                |            | "daVinci-style static skill
#                |            | injection; online summarization
#                |            | is the roadmap."
#
# ============================================================
# WHY ABLATION NUMBERS DICTATE THE CUT PATTERN
# ============================================================
# Paper Table 2 (8B series, Fast1 / Fast1.2):
#   daVinci-8B full:         44.8 / 22.1
#   Ablation1 (no inj eval): 20.6 / 10.5   <- -24.2 / -11.6 LARGEST DROP
#   Ablation3 (policy only): 45.1 / 16.9   <- Fast1 OK, Fast1.2 -5.2
#   Ablation4 (BM25 only):   51.2 / 8.2    <- Fast1 BEST, Fast1.2 -13.9
#   Ablation5 (no skills):   51.6 / 2.1    <- Fast1 OK, Fast1.2 COLLAPSE
#
# Conclusion for 24h:
#   - Skill INJECTION at inference is non-negotiable (Ablation 1).
#   - BM25-only retrieval (Ablation 4 pattern) is the correct
#     lite scope: Fast1 is fine, Fast1.2 degrades but we are
#     running ~60 steps not 300, so divergence has not opened.
#   - Joint summary training (Abl 3 -> full gap) matters at
#     step 200+, irrelevant for our overnight window.
#
# ============================================================
# EXACT DATA FLOW (single file of truth)
# ============================================================
#
# [OFFLINE, Friday night / early Sat]
# 1. Hand-author skills_step0.jsonl (5 skills):
#    {"name":"tiled_matmul","description":"...","scope":"matmul",
#     "tags":["tiling","shared_mem"],"content":"```triton\n...\n```",
#     "step_added":0,"verify_speedup":0.0}
#    [repeat x5 for vectorized_load, shared_mem_reduction,
#     persistent_kernel, fused_elementwise]
#
# [RUNTIME: per-rollout data flow]
#
# retrieve.py::rank_bm25(task_description, library_path) -> top3_skills
#   - loads skills_step0.jsonl (or latest snapshot by mtime)
#   - builds BM25Index over (name + description + tags + content)
#     via rank_bm25.BM25Okapi([tok(s) for s in corpus])
#   - returns top min(3, len(corpus)) Skill objects
#   - FALLBACK: if library empty or rank_bm25 import fails,
#     returns [] (null-scheme, env degrades gracefully)
#
# inject.py::build_system_prompt(task_desc, skills) -> str
#   - if skills == []: return base_system_prompt (null-scheme)
#   - inject_block = "\n\n# Retrieved Optimization Skills\n"
#   - for skill in skills[:3]:
#       inject_block += f"## {skill.name}\n{skill.content}\n\n"
#   - HARD CAP: if len(inject_block.split()) > 2500:
#       truncate to 2500 tokens (protect 7B context window)
#   - return base_system_prompt + inject_block
#   # This is daVinci Section 3.4 verbatim: content+name injected
#   # into first user turn before policy generation begins.
#
# kernel_env.py (GRPOTrainer reward_funcs path):
#   def grade_kernel(kernel_src, op, M, N, dtype, task_desc):
#       skills = rank_bm25(task_desc, LIBRARY_PATH)
#       system_prompt = build_system_prompt(task_desc, skills)
#       # system_prompt is used for the NEXT rollout's generation,
#       # not for grading the current submission.
#       # Grade is purely: rewards.compute_reward(kernel_src, op, M, N)
#       result = subprocess_runner.run_grade(kernel_src, op, M, N, dtype)
#       return rewards.compute_reward(**result)
#
# rewards.py::compute_reward(correct_42, correct_44, dtype_ok,
#                             shape_ok, launches_timed,
#                             speedup, pr_ok) -> float
#   # Canonical formula — imported by BOTH grade.py and kernel_env.py
#   # NEVER duplicated, NEVER routed through hud.graders.combine()
#   if not (correct_42 and correct_44 and dtype_ok
#           and shape_ok and launches_timed > 0):
#       return 0.0                          # Layer 1-4 hard cap
#   if speedup < SPEEDUP_FLOOR:            # 1.1x dead-band
#       return CORRECT_FLOOR               # 0.3 (Kevin floor)
#   PR_BONUS = 0.2 if pr_ok else 0.0      # binary daVinci PR gate
#   raw = CORRECT_FLOOR + speedup / P_TARGET + PR_BONUS
#   return float(clip(raw, 0.0, 2.0))     # bounded [0,2] for GRPO
#   # P_TARGET=1.5, CORRECT_FLOOR=0.3, SPEEDUP_FLOOR=1.1
#
# ============================================================
# ENVIRONMENT_FACTORY DECISION (trl API grounding)
# ============================================================
# VERIFIED from trl v1.6.0 docs (fetched live):
#   environment_factory IS the recommended stable path for
#   multi-turn tool-calling in GRPOTrainer (not experimental).
#   Requires openenv package + WebSocket server OR local class.
#   vllm_mode="colocate" confirmed working on single GPU.
#   rollout_func = manual alternative (full control).
#
# DECISION for Protean:
#   PRIMARY PATH = reward_funcs (single-turn, ~30 lines,
#     rock-solid, kicks at midnight, no openenv dependency).
#   STRETCH PATH = environment_factory with a local KernelEnv
#     class exposing submit_kernel(src:str)->str tool method,
#     NO WebSocket server needed (local class pattern).
#     Activated ONLY by --multi-turn flag, attempted Sun morning
#     ONLY if reward curve is already green.
#
# KernelEnv class (stretch, kernel_env.py):
#   class KernelEnv:
#       def __init__(self):
#           self.reward = 0.0
#           self.turn = 0
#           self.last_feedback = ""
#       def reset(self, op, M, N, dtype, task_desc, **kw) -> str:
#           self.reward = 0.0; self.turn = 0
#           skills = rank_bm25(task_desc, LIBRARY_PATH)
#           self._system = build_system_prompt(task_desc, skills)
#           return self._system + "\n\n" + build_task_prompt(op,M,N,dtype)
#       def submit_kernel(self, kernel_src: str) -> str:
#           """Submit a Triton kernel for compilation, correctness
#           check, and speedup measurement.
#           Args:
#               kernel_src: Complete Python/Triton kernel source.
#           Returns:
#               Feedback string with compile status, allclose result,
#               and speedup if correct.
#           """
#           result = subprocess_runner.run_grade(
#               kernel_src, self._op, self._M, self._N, self._dtype)
#           self.reward = rewards.compute_reward(**result)
#           self.turn += 1
#           return format_feedback(result)  # compile_err | allclose_fail | "speedup=X.Xx"
#   def reward_func(environments, **kw):
#       return [env.reward for env in environments]
#
# ============================================================
# HOUR BUDGET (against 13.5h serial spine)
# ============================================================
# skilllib seed-JSONL + BM25 retrieve + inject:  1.5h
#   (written Sat afternoon while Modal warmup runs — overlaps
#    dead time, costs ~0 spine hours)
# rewards.py canonical formula + unit test:       0.5h
# environment_factory KernelEnv class (stretch):  1.0h
#   (Sunday morning only, gated on green curve)
# TOTAL daVinci-related build time on spine:      2.0h
# TOTAL including stretch:                        3.0h
#
# ============================================================
# WHAT TO CLAIM AT DEMO (honest, grounded)
# ============================================================
# CLAIM: "We integrated daVinci-kernel's profiling-ratio reward
#   term (Eq1, published June 15 2026, days before this hackathon)
#   as our primary anti-hack gate — the PR_BONUS rewards agents
#   whose Triton kernels own the measured GPU time, closing the
#   delegating-wrapper exploit daVinci identified."
# CLAIM: "We inject BM25-retrieved optimization skills into the
#   policy's context before each task, following daVinci's
#   Section 3.4 skill-injection pattern (their Ablation 1 shows
#   this is the single highest-value daVinci technique: -24
#   percentage points when removed)."
# DO NOT CLAIM: "We train the Selection Agent jointly."
# DO NOT CLAIM: "The skill library grows from rollouts."
# DO NOT CLAIM: "We implement the full 3-agent daVinci loop."
# HONEST FRAMING: "daVinci-lite: skill injection (full) +
#   PR reward (full) + BM25 selection (Ablation-4 scope) +
#   static seed library. Online summarization is the roadmap."
```

**Risks:** CRITICAL (verified from trl docs): environment_factory requires openenv package and a concurrent-session-capable WebSocket server OR a local class. The local class pattern IS supported (no server needed if class is self-contained). Risk: if submit_kernel tool method docstring is malformed, trainer silently skips tool discovery and degrades to single-turn. Mitigation: keep reward_funcs as primary path; never attempt environment_factory on the critical overnight run.; VERIFIED CLAIM FLAG: daVinci paper reports 37.2/70.6/32.2% Fast1 on KernelBench L1/L2/L3 for 14B. These are VENDOR-STATED numbers from arXiv 2606.16497 (June 15 2026), not independently reproduced. The ablation pattern (Ablation 1 largest drop, Ablation 5 Fast1.2 collapse) is the load-bearing grounding for KEEP/CUT decisions, not the absolute percentages.; CLAIMED (not verified): rank_bm25 top-3 skill retrieval will improve reward density over null-scheme for a 7B model that has never seen the seed skills. The paper proves this for a 3-agent jointly-trained system; it does not directly prove it for a frozen static BM25+7B combination. Risk: skill injection may add noise rather than signal for an undertrained 7B. Mitigation: ablate by running null-scheme vs skill-injected calibration rollouts Saturday afternoon; if skill injection does not improve mean reward in 20 rollouts, disable it (set retrieve to return [] until after GRPO warms up).; LaunchCounter monkeypatch: patching JITFunction.__call__ (Python boundary) is more version-stable than patching CudaDriver.launch (C++ internal that KernelGYM targets). However, if triton is not pinned exactly in the Modal image, torch pulls a different version and the __call__ signature may differ. Mitigation: Dockerfile installs triton LAST with ==EXACT_VERSION; build asserts import triton; assert triton.__version__=='X.Y.Z'; smoke_test (G0 gate) confirms known-good kernel registers >0 launches IN the Modal image before Saturday 12:30.; trl environment_factory server-concurrency requirement: 'max_concurrent_envs must be >= generation_batch_size'. For local class pattern this is not a bottleneck, but if the class holds stateful GPU resources (e.g., a live subprocess), N concurrent instances may OOM. Mitigation: KernelEnv holds NO GPU state; subprocess is spawned-and-killed per submit_kernel call; env is stateless between tool calls.; Reward formula [0,2] range vs GRPO effective LR: clip at 2.0 limits gradient scale. With CORRECT_FLOOR=0.3, a correct kernel at 1.5x speedup with PR earns 0.3+1.0+0.2=1.5; near-optimal at 3x earns 0.3+2.0+0.2=2.5 clipped to 2.0. The LOO advantage normalizes within the group so absolute scale matters less than within-group variance. Verified: single rewards.py file prevents silent divergence between grade.py and kernel_env.py.; KernelGYM license: verified Apache 2.0. The distributed Redis-queue architecture is NOT needed; Protean lifts only the instrumentation concept (JITFunction patch). No KernelGYM import dependency in the critical path — avoids any setup.sh / pydantic-settings conflict in the Modal image.

### Protean — Reward Port: daVinci R=C*(1+speedup+PR) with held-out-shape generalization + 4-layer anti-hack  (~11h)
**Goal:** Produce one canonical, non-gameable, GRPO-stable reward scalar in [0, 2.0] for an agent-submitted Triton kernel, grounded in daVinci Eq1 (verified from paper pages 3-6), hardened by 4 programmatic gates, covering held-out continuous shapes disjoint from training shapes. The formula must live in ONE file (rewards.py) imported by BOTH the hidden grade.py and kernel_env.py — divergence between these two is the single most likely silent overnight failure. Decision on each daVinci term is KEEP/CUT/LITE with justification and hour cost.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/rewards.py` — THE single canonical reward formula. Imported by grade.py and kernel_env.py. Contains compute_reward(), REWARD_CONSTANTS, and the 4-layer gate logic. Never duplicated. Unit-tested before GRPO kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tasks/<op>/donotaccess/grade.py` — Hidden per-op grader (root:700). Mirrors verilog-template grade.py signature: grade(workdir, override, hidden_root) -> dict. Calls rewards.compute_reward() — never re-implements the formula. Also calls launch_probe and bench.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/_harness/launch_probe.py` — JITFunction.__call__ monkeypatch counter. Wraps triton.runtime.jit.JITFunction.__call__ (stable Python boundary, not CudaDriver.launch internal). Returns (launches_train, launches_timed). Smoke-tested in-image before kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/_harness/bench.py` — Timing harness: 3-seed protocol (seed_correct=42, seed_timed=43, seed_post=44). L2 flush, locked clocks, median-of-100 on seed_timed only. allclose checked on seeds 42 AND 44. Returns speedup float.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/_harness/anti_hack.py` — AST ban + restricted exec namespace. Bans torch.matmul/F.softmax/torch.compile/importlib/__import__/eval/exec in submitted source via ast.walk. Exec'd kernel sees only {'triton': triton, 'tl': tl, '__builtins__': SAFE_BUILTINS}.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/splits.py` — Frozen disjoint train/test shape split. TRAIN_M = (256, 512, 1024, 2048). TEST_M = (400, 800, 1600, 383, 769). _assert_split_disjoint() runs at import time. freeze() re-asserts before writing manifest.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scripts/redteam.py` — Saturday-night red-team checklist. Asserts reward~0 for: (a) PyTorch passthrough, (b) never-launched @triton.jit stub, (c) try/except fallback, (d) bf16-downcast, (e) tl.dot-delegating wrapper on matmul tasks. Hard gate before GRPO kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scripts/check_calibration.py` — Pre-GRPO calibration gate. Runs base Qwen2.5-Coder-7B on ~30 tasks (group=4). Requires: median group reward in (0.0, 1.5), per-group std > 0.05, at least 2% allclose rate. BLOCKS kick if failing. Mirrors verilog check_calibration.py pattern.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tests/test_rewards.py` — Unit tests asserting exact reward scalars for known inputs: incorrect->0.0, correct@1.0x speedup->0.3, correct@1.5x+pr_ok->1.5 (capped at 2.0), never-launched->0.0, bf16-downcast->0.0. Run in Sat-night red-team checklist.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/env.py` — HUD two-yield env. Cloned from verilog-template/env.py. NO from __future__ import annotations. _AgentWorkspace with setpriv uid wall. @env.template kernel_task(op_id: str, M: int, N: int, dtype: str). Calls grade.py via grader.py mapper.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/grader.py` — Public-side grader mapper. Mirrors verilog-template/grader.py exactly: _load_grade_module + _subscores_from_result + evaluate_task. Builds EvaluationResult BY HAND — never through hud.graders.combine() (confirmed erases hard cap at verilog grader.py L72-85).

**APIs:** triton.runtime.jit.JITFunction.__call__ — monkeypatched in launch_probe.py as the stable Python boundary for launch counting (NOT CudaDriver.launch which is a C-extension internal that changes across versions); hud.graders.EvaluationResult + SubScore — built BY HAND in grader.py (never via hud.graders.combine() which renormalizes weights and erases the hard cap; verified at verilog grader.py L72-85); hud.environment.Workspace.shell_argv — subclassed in _AgentWorkspace for setpriv uid-drop; NO from __future__ import annotations in env.py (PydanticUserError -32000 if present; documented in verilog env.py line 9-13); @env.template(id='kernel_task') — HUD two-yield: yield prompt string; yield EvaluationResult; modal.Image.from_dockerfile + @app.function(gpu='H100') — lifted from ml-template/modal_runner.py lines 34-93; torch.cuda.synchronize() + time.perf_counter() — timing pair in bench.py (not CUDA events to avoid driver-level timing overhead on H100)

**daVinci integration:** KEEP/CUT/LITE decisions grounded in paper (arXiv 2606.16497, verified pages 3-9):

KEEP (directly implemented):
- daVinci Eq1 structure R = C(y) * (1 + speedup + PR): KEPT as the template. Our formula is CORRECT_FLOOR + speedup/P_TARGET + PR_BONUS, which is the affine transformation of Eq1 with a floor offset and a cap. The correctness gate C(y) is binary and mapped to the hard_caps system.
- daVinci Skill Library schema (5 fields: name/description/scope/tags/content, JSONL snapshots, step-keyed): KEPT verbatim in skills_step0.jsonl seed file. Each skill is exactly the paper's format (Section 3.2, p.4).
- daVinci lazy-skill filter (reject pytorch-fallback skills): KEPT in ast_check() — the same AST patterns that block agent kernels (torch.matmul, tl.dot delegation) also screen out any skill whose content field contains them.
- daVinci skill verification conceptual gate: KEPT as the red-team checklist (redteam.py) which verifies that the reward oracle correctly rejects exploit patterns before the skill library can encode them.

LITE (simplified for 24h solo):
- daVinci PR (profiling ratio = T_generated/T_total via torch.profiler): LITE — replaced with binary pr_ok = launches_timed > 0 AND NOT delegates_to_matrix_unit. Reason: torch.profiler adds 50-200ms/bench, can segfault on unusual kernels, and its fail-closed fallback (pr=0.0 tripping pr_min gate) drives all rollouts in a noisy group to zero causing GRPO gradient collapse. Binary gate closes the same exploit class (lazy passthrough kernels that register zero Triton launches) at zero overhead.
- daVinci PRS sampling (Eq4: trajectories sampled proportional to PR): LITE — approximated as easy-op curriculum start (elementwise -> reduction -> matmul ordering). This achieves the same intent (prioritize tasks where agent kernels dominate runtime) without the continuous profiler dependency.
- daVinci BM25 pre-filtering (N_BM25=20 candidates -> LLM rerank k_select=3): LITE — BM25 top-3 direct inject without LLM rerank. Reason: LLM rerank at inference time adds latency and requires the 7B model to parse a tool-call schema reliably (fragile at temperature=0 on Qwen2.5-Coder-7B). BM25-only ablation (daVinci Table 2, Ablation 4) still achieves competitive Fast1 scores at L1/L2, degrading mainly at L3. Acceptable for 24h.
- daVinci TRLOO multi-scheme LOO (Eq7: (k+1)*n samples across k skill schemes + null): LITE — standard single-group LOO with group=4. The multi-scheme grouping requires the Selection Agent to propose k=3 different skill subsets per task (Section 3.7, p.6), which is the full 3-agent joint training loop. Out of scope for 24h solo.

CUT (not implemented, explicitly):
- daVinci Joint RL of Selection Agent (Eq8 selection advantage) and Summary Agent (Eq9 summary advantage): CUT. These require two additional LLM forward passes per step plus the full (k+1)*n rollout expansion. The paper's ablation (Table 2, Ablation 3: policy RL only) shows competitive Fast1 scores at L1/L2 without joint selection/summary RL, degrading mainly at L3. Our target is L1-L2 in 24h.
- daVinci SFT cold start (3-phase: Summary SFT on Dr.Kernel trajectories -> Selection SFT -> Policy SFT with skill injection): CUT. Would require Dr.Kernel trajectory dataset (not locally available) and 2+ hours of SFT before RL. Replaced by easy-op curriculum start and bootstrap credit in compute_bootstrap_reward().
- daVinci Qwen3-Embedding-8B + HDBSCAN diversity filter for SFT data: CUT. Loading a second 8B model into the 80GB H100 alongside the policy + vLLM would OOM. Not applicable since we skip the SFT phase.
- daVinci MRS importance-ratio filter (Eq3): CUT. trl's GRPO implementation handles importance ratio clipping internally via clip_range. Not re-implemented at the reward level.
- daVinci continuous PR reward bonus: CUT from reward (binary gate retained). Cannot safely use torch.profiler in the 24h overnight path.

SLIDE CLAIM: 'We integrated daVinci-kernel's profiling-ratio reward term (published June 15 2026, during this hackathon) into our verifier to close the partial-runtime exploit.' This is ACCURATE for the binary gate version. Do not claim 'we implemented daVinci's full PR measurement' — that would require torch.profiler."


```python

# ================================================================
# FILE: kernelforge/rewards.py  (THE single source of truth)
# Imported by grade.py AND kernel_env.py — never duplicated.
# ================================================================

# --- VERIFIED daVinci Paper Sources ---
# Eq1 (paper p.3): R_{i,t} = C(y) * (1 + speedup_{i,t} + PR_{i,t})
#   where PR_{i,t} = T_generated / T_total (fraction of end-to-end runtime from agent kernels)
# Eq4 (paper p.4): PRS sampling weight p_{i,t} = clip((PR_{i,t} - tau) / s, 0, 1)
# Ablation Table 2: daVinci-8B w/o skills = 44.8% Fast1 on L2 vs base RL = 51.6% Fast1
#   -> skill injection is additive but base RL already works; PR reward is the anti-exploit moat.
# CLAIM (vendor): "+46% over Dr.Kernel-14B on L3" — paper Table 1 confirmed: daVinci-14B
#   L3 Fast1 = 32.2% vs Dr.Kernel-14B = 22.1%. Ratio = 1.456. The +46% claim is VERIFIED
#   from paper data, not just vendor marketing. Cite as: "daVinci Table 1".

# --- KEEP/CUT/LITE decisions (grounded) ---
# daVinci Eq1 C(y)*(1+speedup+PR):
#   LITE — KEEP the structure (correctness gate * speedup term) but:
#   (a) Replace continuous PR (T_generated/T_total via torch.profiler) with BINARY pr_ok gate.
#       Reason: torch.profiler adds 50-200ms/bench (verified risk), can segfault, and its
#       fail-closed default (pr=0.0 -> pr_min gate -> all-zero group) causes GRPO gradient
#       collapse. The paper uses PR as a sampling weight (Eq4) and reward multiplier (Eq1)
#       for a 14B model with a full profiling stack — not appropriate for a 24h solo 7B run.
#       Binary gate: pr_ok = launches_timed > 0 AND NOT delegates_to_matrix_unit(src, op).
#       This closes the same exploit (lazy passthrough kernel) at zero profiling overhead.
#   (b) Add CORRECT_FLOOR = 0.3 (from Kevin/Dr.Kernel pattern, not in daVinci) to ensure
#       reward density for GRPO when speedup is near 1.0x. daVinci uses pure speedup credit;
#       we add the floor because our model is 7B base, not 8B cold-start post-SFT like daVinci.
#   (c) Bound at [0, 2.0] not [0, ~3+]. daVinci's R can exceed 2 on fast kernels; 3x LR
#       effective multiplier on a 7B model risks gradient explosion. Clip at 2.0.
#
# daVinci TRLOO (Eq7 multi-agent LOO): CUT from reward formula; KEEP for GRPO advantages.
#   Single-turn GRPO uses standard LOO within each group. The (k+1)*n multi-scheme grouping
#   from daVinci (k=3 skill schemes + 1 null) is overkill for a 24h build without the full
#   skill library RL loop. Use group=4, standard trl LOO.
#
# daVinci Skill verification Eq5/6 (R* > alpha*r1 AND R* > beta): CUT from reward formula.
#   Skill library is seed-JSONL + BM25 inject only; no online skill verification loop.
#
# daVinci MRS importance-ratio filter (Eq3): CUT. trl handles this via clip_range in PPO
#   or the equivalent in GRPO. Not re-implementing at the reward level.
#
# daVinci PRS sampling (Eq4): KEEP as conceptual curriculum.
#   Use easy-op curriculum start (elementwise before matmul) which approximates PRS without
#   the continuous PR measurement. Formally: start with fused_add_relu (high PR trivially),
#   then softsign+silu fusion, then matmul variants. ~0h, just task ordering.

# ================================================================
# CANONICAL REWARD CONSTANTS (single source of truth)
CORRECT_FLOOR = 0.3      # Kevin/Dr.Kernel: non-zero floor keeps GRPO reward dense at 7B
P_TARGET = 1.5           # daVinci-style normalizer; speedup/P_TARGET saturates near 1.0 at 1.5x
SPEEDUP_FLOOR = 1.1      # Layer 4: dead-band below 1.1x -> zero speedup credit (anti bf16/noise)
PR_BONUS = 0.2           # daVinci PR term, binary: +0.2 iff pr_ok (launch confirmed + no delegation)
REWARD_CAP = 2.0         # Bound for GRPO stability; prevents 3x effective-LR gradient blowup
BOOTSTRAP_COMPILE = 0.1  # Annealed: +0.1 if kernel compiles (active until calibration gate passes)
BOOTSTRAP_IMPORTS = 0.2  # Annealed: +0.2 if source contains 'import triton' (step 0 only)

# ================================================================
# 4-LAYER ANTI-HACK GATES (programmatic, no LLM judge)
# Layer 1: AST ban (anti_hack.py) - runs BEFORE exec
# Layer 2: Launch counter (launch_probe.py) - runs during correctness + timing pass
# Layer 3: dtype/shape/seed match (bench.py) - checked post-timing
# Layer 4: Speedup floor (rewards.py) - no credit below SPEEDUP_FLOOR

# ================================================================
def compute_reward(
    correct_seed42: bool,     # allclose(output, eager_ref, atol=1e-3) on seed_correct=42
    correct_seed44: bool,     # allclose on seed_post=44 (materialized AFTER timing ends)
    dtype_ok: bool,           # output.dtype == reference.dtype
    shape_ok: bool,           # output.shape == reference.shape
    launches_train: int,      # JITFunction.__call__ count during correctness pass
    launches_timed: int,      # JITFunction.__call__ count during timing pass (seed_timed=43)
    speedup: float,           # eager_median_ms / agent_median_ms (both median-of-100)
    pr_ok: bool,              # binary: launches_timed > 0 AND not delegates_to_matrix_unit
    bootstrap_step: int = -1, # training step; bootstrap active if < BOOTSTRAP_CUTOFF
    ast_banned: bool = False, # True if anti_hack.py found banned nodes
) -> dict:
    """
    Single canonical reward function for Protean.
    Called by BOTH grade.py and kernel_env.py via: from rewards import compute_reward
    Returns dict with 'reward' (float in [0, 2.0]) and 'hard_caps' (list[str]).
    """
    hard_caps = []

    # LAYER 0: AST ban (pre-exec; if anti_hack found banned nodes, never even ran)
    if ast_banned:
        hard_caps.append("ast_ban")
        return {"reward": 0.0, "hard_caps": hard_caps}

    # LAYER 1: Correctness (allclose on TWO independent seeds)
    # seed_correct=42 checked pre-timing; seed_post=44 materialized AFTER timing ends
    # -> defeats async-output-overwrite: kernel cannot precompute seed_post tensors
    if not (correct_seed42 and correct_seed44):
        hard_caps.append("allclose_failed")
        return {"reward": 0.0, "hard_caps": hard_caps}

    # LAYER 2: dtype + shape integrity
    if not dtype_ok:
        hard_caps.append("dtype_mismatch")
        return {"reward": 0.0, "hard_caps": hard_caps}
    if not shape_ok:
        hard_caps.append("shape_mismatch")
        return {"reward": 0.0, "hard_caps": hard_caps}

    # LAYER 3: Triton launch proof (both passes must show >0 JIT calls)
    # launches_train: correctness pass; launches_timed: timing pass (disjoint seed)
    # -> defeats never-launched stub: @triton.jit decorated but never called
    if launches_train == 0 or launches_timed == 0:
        hard_caps.append("no_triton_launch")
        return {"reward": 0.0, "hard_caps": hard_caps}

    # LAYER 4: Speedup floor dead-band (no credit below 1.1x)
    # -> defeats bf16-downcast: lower precision looks faster but often <1.1x on well-tuned eager
    # Also catches: torch-wrap passthrough (speedup ~1.0), try/except silent fallback (~1.0)
    if speedup < SPEEDUP_FLOOR:
        hard_caps.append("speedup_below_floor")
        # Note: return CORRECT_FLOOR only if bootstrap_step active (see below)
        # Default: reward=0.0 when below speedup floor
        return {"reward": 0.0, "hard_caps": hard_caps}

    # All 4 gates passed -> compute reward
    # daVinci-inspired structure: correctness * (floor + speedup_credit + pr_bonus)
    # Differences from daVinci Eq1: (a) floor added, (b) PR is binary not ratio, (c) capped
    speedup_credit = speedup / P_TARGET  # saturates ~1.0 at 1.5x speedup; can exceed 1.0 beyond that
    reward = CORRECT_FLOOR + speedup_credit + (PR_BONUS if pr_ok else 0.0)
    reward = min(reward, REWARD_CAP)  # cap at 2.0 for GRPO stability

    return {"reward": round(reward, 6), "hard_caps": hard_caps, "speedup": speedup, "pr_ok": pr_ok}


def compute_bootstrap_reward(
    compiled: bool,
    imports_triton: bool,
    step: int,
    bootstrap_cutoff: int = 30,
) -> float:
    """
    Additive bootstrap credit for early training steps when base 7B produces near-zero
    allclose rate. Active only when step < bootstrap_cutoff. Anneals to 0.0 after cutoff.
    This is NOT in daVinci; it addresses the 7B cold-start problem identified in phase14 audit.
    Called in kernel_env.py ONLY IF the calibration gate reports reward variance near 0.
    """
    if step >= bootstrap_cutoff:
        return 0.0
    credit = 0.0
    if imports_triton:
        credit += BOOTSTRAP_IMPORTS
    elif compiled:
        credit += BOOTSTRAP_COMPILE
    return credit


# ================================================================
# FILE: kernelforge/_harness/launch_probe.py
# ================================================================
# Wraps triton.runtime.jit.JITFunction.__call__ (stable Python API boundary,
# NOT CudaDriver.launch internal which changes across triton versions).
# Reason for this boundary: CudaDriver.launch is a C-extension internal;
# JITFunction.__call__ is the Python-level entry point for any @triton.jit call
# and has been stable since triton 2.x. Pinning triton version in Dockerfile
# is a necessary companion (pip install triton==3.0.0 LAST, after torch).

import triton
from triton.runtime.jit import JITFunction
from contextlib import contextmanager

_launch_count = 0
_orig_call = None

def _counting_call(self, *args, **kwargs):
    global _launch_count
    _launch_count += 1
    return _orig_call(self, *args, **kwargs)

@contextmanager
def count_launches():
    """Context manager: patches JITFunction.__call__, yields, returns count, restores."""
    global _launch_count, _orig_call
    _launch_count = 0
    _orig_call = JITFunction.__call__
    JITFunction.__call__ = _counting_call
    try:
        yield lambda: _launch_count
    finally:
        JITFunction.__call__ = _orig_call
        _orig_call = None

# Smoke test (run IN Modal image via modal run modal_app.py::smoke_test):
# def smoke_test():
#     import torch, triton, triton.language as tl
#     @triton.jit
#     def _add(x_ptr, y_ptr, z_ptr, N, BLOCK: tl.constexpr):
#         pid = tl.program_id(0)
#         offs = pid * BLOCK + tl.arange(0, BLOCK)
#         x = tl.load(x_ptr + offs, mask=offs < N)
#         y = tl.load(y_ptr + offs, mask=offs < N)
#         tl.store(z_ptr + offs, x + y, mask=offs < N)
#     x = torch.ones(1024, device='cuda')
#     y = torch.ones(1024, device='cuda')
#     z = torch.zeros(1024, device='cuda')
#     with count_launches() as get_count:
#         _add[(1,)](x, y, z, 1024, BLOCK=1024)
#     assert get_count() == 1, f"smoke FAIL: expected 1 got {get_count()}"
#     # Never-launched stub test:
#     @triton.jit
#     def _never(x_ptr): pass
#     with count_launches() as get_count:
#         _ = torch.ones(4, device='cuda') + 1  # torch op, not triton
#     assert get_count() == 0, "smoke FAIL: got launches from non-triton op"
#     print("launch_probe smoke test PASSED")


# ================================================================
# FILE: kernelforge/_harness/bench.py
# ================================================================
# 3-seed protocol (closes async-output-overwrite attack confirmed in Round-1 revisions):
#   seed_correct=42: pre-timing correctness check (inputs exist before kernel runs)
#   seed_timed=43:   timing loop ONLY; never correctness-checked (kernel can't know inputs)
#   seed_post=44:    materialized AFTER timing returns; never passed during warmup or timing
#                    -> kernel cannot precompute against seed_post by construction
#
# Also: L2 cache flush between warmup and timed runs (allocate+fill a buffer > L2 size)
# GPU clock locking: done in @modal.enter() of the Modal function, not here
# Median-of-100 for stability (not mean, which is skewed by thermal throttle spikes)

import torch, time
from contextlib import contextmanager

L2_SIZE_BYTES = 40 * 1024 * 1024  # 40MB > H100 L2 (35MB); adjust if A100 (40MB)

def _flush_l2(device='cuda'):
    buf = torch.empty(L2_SIZE_BYTES // 4, dtype=torch.float32, device=device)
    buf.zero_()
    torch.cuda.synchronize()

def bench_kernel(
    kernel_fn,          # callable: (inputs_dict) -> output tensor
    eager_fn,           # callable: (inputs_dict) -> reference output tensor
    make_inputs,        # callable: (seed: int) -> dict of tensors on cuda
    n_warmup: int = 10,
    n_timed: int = 100,
    device: str = 'cuda',
) -> dict:
    """
    Returns: {speedup, correct_s42, correct_s44, launches_train, launches_timed,
              dtype_ok, shape_ok, agent_ms, eager_ms}
    """
    from _harness.launch_probe import count_launches

    # Correctness pass on seed_correct=42
    inputs_42 = make_inputs(42)
    ref_42 = eager_fn(inputs_42)
    with count_launches() as get_train_count:
        out_42 = kernel_fn(inputs_42)
    launches_train = get_train_count()
    correct_s42 = bool(torch.allclose(out_42.float(), ref_42.float(), atol=1e-3, rtol=1e-3))
    dtype_ok = (out_42.dtype == ref_42.dtype)
    shape_ok = (out_42.shape == ref_42.shape)

    # Timing pass on seed_timed=43 (never correctness checked)
    inputs_43 = make_inputs(43)
    # Warmup (not timed)
    for _ in range(n_warmup):
        kernel_fn(inputs_43)
    torch.cuda.synchronize()

    # Timed agent kernel (median-of-100 with L2 flush)
    agent_times = []
    with count_launches() as get_timed_count:
        for _ in range(n_timed):
            _flush_l2(device)
            t0 = time.perf_counter()
            kernel_fn(inputs_43)
            torch.cuda.synchronize()
            agent_times.append(time.perf_counter() - t0)
    launches_timed = get_timed_count()
    agent_ms = float(sorted(agent_times)[n_timed // 2]) * 1000.0

    # Eager baseline on same seed_timed=43 (cached per op+shape to avoid re-timing)
    eager_times = []
    for _ in range(n_timed):
        _flush_l2(device)
        t0 = time.perf_counter()
        eager_fn(inputs_43)
        torch.cuda.synchronize()
        eager_times.append(time.perf_counter() - t0)
    eager_ms = float(sorted(eager_times)[n_timed // 2]) * 1000.0

    # Post-timing allclose on seed_post=44 (tensors materialized HERE, after timing ends)
    inputs_44 = make_inputs(44)   # <- these tensors do not exist during warmup or timed loop
    ref_44 = eager_fn(inputs_44)
    out_44 = kernel_fn(inputs_44) # one more execution; launch count not tracked here
    correct_s44 = bool(torch.allclose(out_44.float(), ref_44.float(), atol=1e-3, rtol=1e-3))

    speedup = eager_ms / agent_ms if agent_ms > 0 else 0.0

    return {
        "speedup": speedup,
        "correct_s42": correct_s42,
        "correct_s44": correct_s44,
        "launches_train": launches_train,
        "launches_timed": launches_timed,
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "agent_ms": agent_ms,
        "eager_ms": eager_ms,
    }


# ================================================================
# FILE: kernelforge/_harness/anti_hack.py
# ================================================================
import ast

BANNED_NAMES = {
    "torch", "F",             # torch.matmul, F.softmax passthrough
    "__import__", "eval", "exec", "compile", "open", "importlib",
    "getattr",                # getattr(torch, 'matmul') laundering
}
BANNED_ATTRS = {
    ("torch", "matmul"), ("torch", "mm"), ("torch", "bmm"),
    ("torch", "compile"),
    ("F", "softmax"), ("F", "layer_norm"), ("F", "linear"),
    ("tl", "dot"),            # tl.dot delegates to tensor core (daVinci delegating-wrapper attack)
}
BANNED_IMPORT_MODULES = {"importlib", "ctypes", "cffi"}

SAFE_BUILTINS = {
    k: __builtins__[k]
    for k in ("len", "range", "enumerate", "zip", "int", "float", "bool",
              "str", "list", "tuple", "dict", "set", "print", "abs", "min", "max")
    if k in (d := __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__))
}

def ast_check(source: str, op: str = "generic") -> tuple[bool, list[str]]:
    """
    Returns (is_banned: bool, reasons: list[str]).
    Layer 1 gate: runs before exec() in subprocess_runner.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return True, [f"syntax_error: {e}"]

    reasons = []
    for node in ast.walk(tree):
        # Ban dynamic imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, 'names', []):
                mod = alias.name.split('.')[0]
                if mod in BANNED_IMPORT_MODULES:
                    reasons.append(f"banned_import: {alias.name}")

        # Ban specific attribute access (torch.matmul, tl.dot etc.)
        if isinstance(node, ast.Attribute):
            for (obj_name, attr_name) in BANNED_ATTRS:
                if (isinstance(node.value, ast.Name) and
                    node.value.id == obj_name and
                    node.attr == attr_name):
                    reasons.append(f"banned_attr: {obj_name}.{attr_name}")

        # Ban dynamic name calls
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES - {"torch", "F"}:
            reasons.append(f"banned_name: {node.id}")

    # For matmul-class ops: also ban tl.dot (delegates to cuBLAS/tensor-cores)
    # This is the daVinci delegating-wrapper attack fix
    if op in ("matmul", "batched_matmul"):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and
                isinstance(node.value, ast.Name) and
                node.value.id == "tl" and
                node.attr == "dot"):
                reasons.append("banned_tl_dot_on_matmul_op")

    return len(reasons) > 0, reasons


def make_restricted_exec_globals(triton_mod, tl_mod):
    """Restricted globals for exec(compiled_src, ...) in subprocess_runner."""
    return {
        "triton": triton_mod,
        "tl": tl_mod,
        "__builtins__": SAFE_BUILTINS,
        # torch is NOT in exec globals: kernel must use Triton pointer arithmetic only
        # (grader's correctness harness runs eager reference separately with full torch access)
    }


# ================================================================
# FILE: kernelforge/tasks/<op>/donotaccess/grade.py
# ================================================================
# Signature mirrors verilog grade.py: grade(workdir, override, hidden_root) -> dict
# Root:700 file; agent cannot read it.

# from rewards import compute_reward, SPEEDUP_FLOOR, P_TARGET  <- canonical import
# from _harness.anti_hack import ast_check
# from _harness.bench import bench_kernel
# (reference eager_fn loaded from hidden_root/reference.py, agent-inaccessible)

def grade(workdir, override=None, hidden_root=None):
    import importlib.util, sys
    from pathlib import Path
    from rewards import compute_reward

    # Load agent's submitted kernel source
    kernel_path = Path(workdir) / "kernel.py"
    if not kernel_path.exists():
        return {"reward": 0.0, "hard_caps": ["no_submission"], "subscores": {}}

    source = kernel_path.read_text(encoding="utf-8")

    # LAYER 0: AST ban (before exec)
    from _harness.anti_hack import ast_check, make_restricted_exec_globals
    ast_banned, ast_reasons = ast_check(source, op=OP_NAME)
    if ast_banned:
        return {"reward": 0.0, "hard_caps": ["ast_ban"] + ast_reasons, "subscores": {}}

    # Load reference module from hidden_root (agent cannot access this path)
    ref_mod = _load_hidden_module(hidden_root / "reference.py")
    eager_fn = ref_mod.eager_fn
    make_inputs = ref_mod.make_inputs  # make_inputs(seed: int) -> dict

    # Exec agent kernel in restricted namespace
    import triton, triton.language as tl
    g = make_restricted_exec_globals(triton, tl)
    try:
        exec(compile(source, "kernel.py", "exec"), g)
        kernel_fn = g["triton_kernel"]  # agent must define this name
    except Exception as e:
        bootstrap_r = _bootstrap_credit(source, step=_get_current_step())
        return {"reward": bootstrap_r, "hard_caps": ["exec_error"], "subscores": {"error": str(e)}}

    # bench_kernel handles all 3 seeds + L2 flush + launch counting
    from _harness.bench import bench_kernel
    try:
        br = bench_kernel(kernel_fn, eager_fn, make_inputs)
    except Exception as e:
        return {"reward": 0.0, "hard_caps": ["bench_error"], "subscores": {"error": str(e)}}

    # Binary PR gate (daVinci-inspired, simplified for 24h solo)
    from _harness.anti_hack import ast_check as _  # already done above
    pr_ok = br["launches_timed"] > 0  # tl.dot ban for matmul already handled in ast_check

    # THE canonical reward call
    r = compute_reward(
        correct_seed42=br["correct_s42"],
        correct_seed44=br["correct_s44"],
        dtype_ok=br["dtype_ok"],
        shape_ok=br["shape_ok"],
        launches_train=br["launches_train"],
        launches_timed=br["launches_timed"],
        speedup=br["speedup"],
        pr_ok=pr_ok,
        bootstrap_step=_get_current_step(),
        ast_banned=False,  # already checked above
    )

    # Build subscores for EvaluationResult (grader.py maps these via _subscores_from_result)
    subscores = {
        "correctness": {"weight": 0.3, "raw_score": 1.0 if (br["correct_s42"] and br["correct_s44"]) else 0.0,
                        "weighted_score": 0.3, "result": {"s42": br["correct_s42"], "s44": br["correct_s44"]}},
        "speedup": {"weight": 0.5, "raw_score": min(br["speedup"] / P_TARGET, 1.0),
                    "weighted_score": 0.5 * min(br["speedup"] / P_TARGET, 1.0),
                    "result": {"speedup": br["speedup"], "agent_ms": br["agent_ms"], "eager_ms": br["eager_ms"]}},
        "pr_gate": {"weight": 0.2, "raw_score": 1.0 if pr_ok else 0.0,
                    "weighted_score": 0.2 * (1.0 if pr_ok else 0.0),
                    "result": {"launches_train": br["launches_train"], "launches_timed": br["launches_timed"]}},
    }
    # hard_cap_penalty reconciliation (mirrors verilog grader.py L72-85)
    # grader.py's _subscores_from_result will add hard_cap_penalty SubScore if reward < weighted_sum
    return {"reward": r["reward"], "hard_caps": r["hard_caps"], "subscores": subscores}


# ================================================================
# FILE: kernelforge/splits.py  (the held-out shape moat)
# ================================================================
# Disjointness is the primary scientific claim: test shapes prove the trained kernel
# generalizes to shapes it was never trained on (not just recompiled to a new tile size).
# Key insight from phase14 audit: test shapes must be OFF the power-of-two grid to be
# defensible in CTO Q&A ("powers-of-two are everywhere in pretraining corpora").

TRAIN_M = (256, 512, 1024, 2048)   # powers of two; common in pretraining, explicitly labeled training
TEST_M  = (400, 800, 1600,          # multiples of 100, rare in GPU code; off-grid
           383, 769, 1537)          # off-by-one from powers of two; provably unusual

# N dimension: same split per op, or fixed at 512 for elementwise
TRAIN_N = (256, 512, 1024, 2048)
TEST_N  = (400, 800, 384, 1536)

def _assert_split_disjoint():
    train_set = set(TRAIN_M) | set(TRAIN_N)
    test_set  = set(TEST_M)  | set(TEST_N)
    overlap = train_set & test_set
    assert not overlap, f"DISJOINTNESS VIOLATED: {overlap}"

_assert_split_disjoint()  # runs at import time; fail-fast if a careless edit creates overlap

def is_train_shape(M: int, N: int) -> bool:
    return M in TRAIN_M and N in TRAIN_N

def is_test_shape(M: int, N: int) -> bool:
    return M in TEST_M  # N can vary; M is the primary generalization axis


# ================================================================
# FILE: tests/test_rewards.py  (the red-team gate, run before kick)
# ================================================================
# Exact scalar assertions on known inputs:

def test_incorrect_kernel():
    r = compute_reward(correct_seed42=False, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=1, launches_timed=1,
                       speedup=2.0, pr_ok=True)
    assert r["reward"] == 0.0
    assert "allclose_failed" in r["hard_caps"]

def test_never_launched():
    r = compute_reward(correct_seed42=True, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=0, launches_timed=1,
                       speedup=2.0, pr_ok=True)
    assert r["reward"] == 0.0
    assert "no_triton_launch" in r["hard_caps"]

def test_below_speedup_floor():
    r = compute_reward(correct_seed42=True, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=1, launches_timed=1,
                       speedup=1.05, pr_ok=True)  # 1.05 < SPEEDUP_FLOOR=1.1
    assert r["reward"] == 0.0
    assert "speedup_below_floor" in r["hard_caps"]

def test_correct_at_1x():
    # Speedup exactly at floor (1.1x), no PR bonus
    r = compute_reward(correct_seed42=True, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=1, launches_timed=1,
                       speedup=1.1, pr_ok=False)
    # reward = 0.3 + 1.1/1.5 + 0.0 = 0.3 + 0.7333 = 1.033
    assert abs(r["reward"] - (0.3 + 1.1/1.5)) < 1e-4

def test_correct_at_1p5x_with_pr():
    # Ideal case: 1.5x speedup + PR bonus
    r = compute_reward(correct_seed42=True, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=2, launches_timed=2,
                       speedup=1.5, pr_ok=True)
    # reward = 0.3 + 1.5/1.5 + 0.2 = 0.3 + 1.0 + 0.2 = 1.5
    assert abs(r["reward"] - 1.5) < 1e-4

def test_cap_at_2():
    # Very fast kernel: would be 0.3 + 3.0 + 0.2 = 3.5 without cap
    r = compute_reward(correct_seed42=True, correct_seed44=True, dtype_ok=True,
                       shape_ok=True, launches_train=1, launches_timed=1,
                       speedup=4.5, pr_ok=True)
    assert r["reward"] == 2.0  # capped

def test_ast_ban():
    from _harness.anti_hack import ast_check
    source = "import torch\ndef triton_kernel(x): return torch.matmul(x, x)"
    banned, reasons = ast_check(source)
    assert banned
    assert any("banned_attr" in r or "banned_import" in r for r in reasons)

def test_tl_dot_ban_on_matmul():
    from _harness.anti_hack import ast_check
    source = "@triton.jit\ndef triton_kernel(a, b, c): tl.dot(a, b, c)"
    banned, reasons = ast_check(source, op="matmul")
    assert banned
    assert any("tl_dot" in r for r in reasons)

```

**Risks:** CRITICAL: JITFunction.__call__ hook silently broken by triton version drift — if torch pulls triton transitively at a newer version than pinned, hook either counts 0 (caps all rewards) or no-ops (security hole). Mitigation: Dockerfile installs torch first, then pip install triton==3.0.0 LAST; build asserts import triton; assert triton.__version__ == '3.0.0'; smoke_test() runs IN the Modal image before kick (not just locally). This is the single highest-severity silent failure.; GRPO reward scale: R in [0.3, 2.0] for a correct kernel means effective LR is 2x-6x compared to a [0,1] reward — gradient noise on early steps. Mitigation: cap at 2.0 (already in compute_reward), use learning_rate=1e-6 not 1e-5 in grpo_loop.py, group=4 not 8 (smaller group = smaller gradient step variance). Monitor gradient norm in callbacks.py; abort if > 10.0.; Async-output-overwrite attack: seed_post=44 tensors materialized AFTER timing — this fully closes the attack BUT requires that make_inputs(44) is called in grade.py AFTER bench_kernel() returns, not inside bench_kernel itself. If a future refactor moves seed_post materialization earlier (into warmup scope), the attack reopens. Mitigation: comment in bench.py explicitly marking the invariant; test_rewards.py cannot test this statically — manual red-team with a precomputing kernel required in redteam.py.; Held-out shapes: TEST_M = (400, 800, 1600, 383, 769, 1537) are off the power-of-two grid but 800 and 1600 are round numbers that may appear in model hidden dims. A judge CTO may argue these are in-distribution. Mitigation: add 3 additional primes (TEST_M_EXTRA = (401, 797, 1601)) as optional eval-only shapes for the slide; frame the claim as 'disjoint from training set' (literally true) not 'never seen in pretraining' (unverifiable).; SPEEDUP_FLOOR=1.1 dead-band may cut legitimate early kernels from 7B base model (base often produces 0.9-1.05x kernels). Mitigation: bootstrap_credit in compute_bootstrap_reward() adds 0.1 for compile-success during steps < 30, giving a non-zero gradient signal even when speedup < floor. Calibration gate must confirm reward variance > 0.05 before kick; if not, lower SPEEDUP_FLOOR to 1.05 and re-calibrate.; rewards.py import divergence: if grade.py OR kernel_env.py ever re-implements the formula (copy-paste during a debugging session), the reward signal seen by the trainer will differ from the reward seen by the HUD grader — invisible until the eval delta looks wrong. Mitigation: both files contain ONLY 'from rewards import compute_reward'; the formula constants (CORRECT_FLOOR, P_TARGET, etc.) live only in rewards.py; test_rewards.py imports from rewards directly and would fail if someone changes the constants in one file.; PR_BONUS=0.2 is small relative to CORRECT_FLOOR=0.3 and speedup_credit — a correct 1.5x kernel gets 1.5 with or without PR. This means the PR gate is more of a hard-cap mechanism than a meaningful reward differentiator. Intentional: binary PR is the anti-exploit gate, not the primary reward signal. Do NOT increase PR_BONUS to make it a primary signal without re-running the calibration gate (it would skew the reward distribution for matmul ops where tl.dot is banned).; daVinci numbers cited on slide as 37.2/70.6/32.2 Fast1 L1/L2/L3 are VERIFIED from paper Table 1. The +46% over Dr.Kernel-14B on L3 is verified: 32.2/22.1 = 1.456. These are SAFE to cite. The '70-90% token savings' from the task description is NOT in the paper and should NOT appear on the slide without a source.

### RL algorithm: daVinci per-agent LOO advantages — KEEP / CUT / LITE decision for Protean solo 24h build  (~3.5h)
**Goal:** Decide exactly which daVinci RL advantage machinery to adopt, simplify, or cut, grounded in (a) the paper's exact equations, (b) what trl GRPOTrainer actually implements, (c) the confirmed 24h solo constraint, and (d) the revised schedule (midnight kick, ~60-120 steps, single-turn primary path). Emit concrete implementation pseudocode, exact hour cost, and residual risks for each decision.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/rewards.py` — SINGLE canonical reward formula — imported by both hidden grade.py and kernel_env.py; the one source of truth for R, advantage inputs, and bootstrap floor
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_loop.py` — Outer training loop: single-turn reward_funcs path (stable trl API); LOO advantage computed by hand over group=4; calibration gate + step-50 abort
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/kernel_env.py` — Thin wrapper: calls rewards.py grade_kernel() and returns the scalar; used by grpo_loop.py reward_funcs callback
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — Pre-GRPO gate: base model 10x per op, assert reward variance > 0 and 2-50% in-band; hard GO/NO-GO
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/callbacks.py` — Step-50 abort callback + held-out eval interleaved every 25 steps; pre-records curve at step 0 (base) for fallback
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_modal.py` — Modal H100 decorator wrapping grpo_loop.py; image build + TRITON_CACHE_DIR env; timeout=86400
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/configs.py` — GRPOConfig: num_generations=4, loss_type=dapo, scale_rewards=False, beta=0.0, max_completion_length=2048, use_vllm=True vllm_mode=colocate
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tests/test_rewards.py` — Unit test: asserts exact scalars from rewards.py — incorrect->0.0, correct@1.0x->0.3, correct@1.5x+pr_ok->1.5; run in red-team checklist before kick

**APIs:** GRPOTrainer(model, reward_funcs, args=GRPOConfig(num_generations=4, loss_type='dapo', scale_rewards=False, beta=0.0, use_vllm=True), train_dataset=...) — STABLE trl API, not environment_factory; GRPOConfig(loss_type='dapo') — token-level normalization, confirmed in trl docs; reduces length bias in variable-length kernel completions; GRPOConfig(scale_rewards=False) — disables std-normalization; per trl docs 'may cause question-level difficulty bias' when enabled; disable for sparse reward distribution; TrainerCallback.on_step_end(control) -> control.should_training_stop=True — step-50 abort, standard HuggingFace Trainer API; EvaluationResult + SubScore built by hand (NOT hud.graders.combine()) — confirmed from verilog-template/grader.py lines 72-101: combine() renormalizes and erases hard cap; triton.runtime.jit.JITFunction.__call__ monkeypatch — LaunchCounter; wraps stable Python boundary not backend driver internal; smoke-tested IN Modal image (not locally) per G0 gate; modal.Image H100 decorator (@app.function gpu='H100', timeout=86400) — from ml-template/modal_runner.py lines 88-93

**daVinci integration:** 
KEEP (LITE): Policy advantage — use trl's built-in LOO over num_generations=4 instead of daVinci Eq7's (k+1)*n=16 group. This IS the daVinci policy advantage at k=0 (null scheme only). daVinci Table 2 Ablation 3 (policy RL only, no selection/summary RL) shows this is competitive for the first ~150 steps, which equals our entire overnight run. Zero additional implementation.

KEEP (LITE): Turn reward Eq1 — implemented in rewards.py as R = clip(0.3 + speedup/1.5 + binary_PR_bonus, 0, 2.0), gated by 4 hard caps. Preserves C(y) correctness gate, speedup/P_TARGET normalization, and PR term; drops continuous torch.profiler in favour of binary launch-counter + AST delegate check. This closes the partial-runtime exploit daVinci Eq1/Eq4 was designed to close, without the profiler fragility that would drive all-zero rewards.

CUT: Selection advantage (Eq8) — requires k>=2 independent skill schemes per task per step and a separate Selection Agent loss term; neither is available in trl GRPOTrainer reward_funcs nor in the seed-JSONL-only skilllib. Slide claim: "equivalent to daVinci Ablation 4 (BM25-only selection), which still beats Dr.Kernel-14B on Level 1 Fast_1 per Table 2."

CUT: Summary advantage (Eq9) — requires policy re-runs for verification per candidate skill and a multi-objective loss trl does not expose. Deferred; in daVinci's own ablations the summary advantage benefit does not appear until after ~step 150.

CUT: MRS filter (Eq3) — is a no-op at single-turn with colocate vLLM and num_iterations=1 (rollout policy = training policy, IS ratio = 1 by construction).

CUT: PRS sampling (Eq4) — replaced by curriculum ordering (elementwise first), which achieves the same high-PR-first prioritization without torch.profiler.

CUT: environment_factory multi-turn path — confirmed broken in trl issue #4543 (open, filed Nov 2025, unresolved): server mode breaks IS for multi-step agents; colocate mode is EXPERIMENTAL with no stability guarantee. Primary path is reward_funcs single-turn. Manual 2-turn loop (50 lines, offline batch, feeds reward_funcs) is the ONLY safe stretch for multi-turn credit within 24h.


```python

# ===================================================================
# DECISION SUMMARY (read this first)
# ===================================================================
# daVinci Eq7  (Policy advantage  A^pol over (k+1)*n=16 samples)  → LITE (simplified)
# daVinci Eq8  (Selection advantage A^sel over k=3 skill schemes)  → CUT (no skill RL)
# daVinci Eq9  (Summary advantage  A^sum gated by verification)    → CUT (no summary RL)
# daVinci Eq1  (Turn reward R=C(y)*(1+speedup+PR))                → LITE (bounded, binary PR)
# daVinci Eq3  (MRS importance-ratio filter)                       → CUT (overkill for 1 agent)
# daVinci Eq4  (PRS profiling-ratio sampling)                      → CUT (binary PR gate replaces)
# daVinci Eq5/6 (Skill verification threshold)                     → PARTIAL (seed JSONL only)
# daVinci multi-turn TRLOO (K=5 turns, gamma discounted returns)   → SINGLE-TURN GRPO primary
#                                                                      manual K=2 turn as stretch
# ===================================================================

# ===================================================================
# WHAT trl GRPOTrainer ACTUALLY GIVES YOU (grounding facts)
# ===================================================================
# VERIFIED from docs + issue #4543 (open, filed Nov 2025, unresolved):
#
# reward_funcs path (STABLE):
#   GRPOTrainer(model=..., reward_funcs=[fn], train_dataset=..., args=GRPOConfig(...))
#   fn signature: fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]
#   Advantage = (r_i - mean(r)) / std(r) computed internally per group of num_generations=G
#   This IS LOO in the G=group sense: mean(r) over G samples is the LOO baseline (minus self
#   is approximate when G is large; at G=4 it is exact LOO = (sum - r_i) / (G-1)).
#   loss_type="dapo" gives token-level normalization (recommended for code gen).
#   scale_rewards=False disables std-scaling (removes difficulty bias, per trl docs).
#   STABLE API — ship this.
#
# environment_factory path (EXPERIMENTAL, BROKEN for multi-turn):
#   Issue #4543: server mode breaks importance sampling for multi-step agents because
#   each rollout gets a different prompt prefix (env state) but the trainer forces one
#   shared prompt per dataset example duplicated num_generations times.
#   Status: OPEN, no fix, no timeline.
#   DECISION: DO NOT USE on the overnight path. Not even colocate mode is safe for
#   multi-step because the IS correction assumes shared prefix.
#   The manual 50-line Python multi-turn loop feeding reward_funcs IS safe (see below).

# ===================================================================
# DECISION 1: POLICY ADVANTAGE (daVinci Eq7) → LITE
# ===================================================================
# daVinci Eq7: A^pol_{i,j,t} = G_{i,j,t} - 1/((k+1)*n - 1) * sum_{(i',j')!=(i,j)} G_{i',j',t}
# where k=3 skill schemes + 1 null, n=4 rollouts = (k+1)*n = 16 samples per group per turn.
# This is TRLOO over BOTH the rollout dimension AND the skill-scheme dimension simultaneously.
#
# WHY CUT THE SKILL-SCHEME DIMENSION:
#   - Requires k=3 independent Selection Agent calls per task per step (k*n=12 rollouts
#     + n=4 null rollouts = 16 forward passes per gradient step)
#   - Solo with seed-JSONL-only skilllib: only 1 skill context, not k=3 independent ones
#   - 16 rollouts/step on one H100 with 7B policy + vLLM colocate → severe memory/throughput
#     constraint; at ~3-9 min/step that is 20-60 GRPO steps overnight max with group=16
#
# LITE VERSION (what we actually implement):
#   Use trl's built-in advantage over group=4 rollouts (the "n" dimension only).
#   This IS valid LOO: A_i = R_i - mean_{j!=i}(R_j) with G=4 groups.
#   It is daVinci Eq7 with k=0 (null scheme only), which equals plain GRPO-LOO.
#   daVinci paper ablation (Table 2, Ablation3 "policy RL only") shows this still achieves
#   competitive Fast_1 scores — it is the established baseline, not a degraded hack.
#
# HOUR COST: 0h (trl computes this internally; we just set num_generations=4)
#
# IMPLEMENTATION in grpo_loop.py:
#
from trl import GRPOConfig, GRPOTrainer
from kernel_env import grade_kernel_reward   # calls rewards.py

def reward_fn(prompts, completions, **kwargs):
    # Called once per group batch; returns list[float] of length num_generations
    # trl then computes A_i = (r_i - mean(r)) / std(r) internally (or /1 if scale_rewards=False)
    return [grade_kernel_reward(p, c) for p, c in zip(prompts, completions)]

training_args = GRPOConfig(
    num_generations=4,          # G=4: LOO baseline = mean of other 3
    loss_type="dapo",           # token-level norm (good for variable-length code)
    scale_rewards=False,        # disable std-scaling to avoid difficulty bias
    beta=0.0,                   # no KL penalty (standard for code RL, per trl docs)
    max_completion_length=2048, # cap kernel length
    use_vllm=True,
    # vllm_mode="colocate" is default; avoids server-mode issue #4543
    output_dir="checkpoints/kernelforge",
    save_steps=10,
    logging_steps=1,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-7B",
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=task_dataset,  # shape-sampled tasks, JSONL
)

# ===================================================================
# DECISION 2: SELECTION ADVANTAGE (daVinci Eq8) → CUT
# ===================================================================
# daVinci Eq8: A^sel_i = R_bar_i - 1/k * sum_{i'!=i} R_bar_{i'}
# where R_bar_i = 1/n * sum_j max_t R_{i,j,t} = mean-max return per skill scheme.
# This trains the Selection Agent to prefer skill combinations that lift policy returns
# over alternative selections on the SAME task.
#
# WHY CUT:
#   - Requires k>=2 independent skill schemes per task per step (k=3 in daVinci)
#   - With seed-JSONL + BM25-inject, we have exactly 1 deterministic context (or null)
#   - No separate Selection Agent head to train (same Qwen backbone, separate lora would
#     require a second optimizer or multi-head forward, not in trl GRPOTrainer)
#   - 0 skill RL = 0 selection advantage = clean cut; the ablation (Table 2, Ablation4
#     "policy RL + BM25 selection") shows BM25-only selection still reaches competitive
#     Fast_1 — that is exactly what we are doing
#
# HOUR COST: 0h (not built)
# SLIDE CLAIM: "daVinci-style BM25 skill injection without joint selection RL;
#   equivalent to their Ablation 4, which exceeds Dr.Kernel-14B on Level 1 Fast_1"

# ===================================================================
# DECISION 3: SUMMARY ADVANTAGE (daVinci Eq9) → CUT
# ===================================================================
# daVinci Eq9: A^sum_m = R^sum_m - 1/(s-1) * sum_{m'!=m} R^sum_{m'}
# where R^sum_m = r_verify^(m) * 1[r_verify^(m) >= max(beta, alpha*r1^(m))]
# s=2 independent summary candidates per triggered task.
#
# WHY CUT:
#   - Requires running the policy AGAIN with the candidate skill to compute r_verify
#     (Eq6): one additional forward+grade per candidate skill per triggered task
#   - 60-120 GRPO steps overnight; even 10% trigger rate = 6-12 summary episodes =
#     12-24 extra grade() calls, non-trivial on a shared H100
#   - The Summary Agent needs its own loss term (Loss_summary with w_sum weight) on
#     top of the policy loss, requiring multi-objective gradient that trl GRPOTrainer
#     does not expose in a single reward_funcs call
#   - Ablation 3 (policy RL only, no summary RL) is still competitive early-training
#     (daVinci Figure 3 shows it tracks full system up to ~step 150 before diverging)
#     and we have at most 120 steps overnight — we are in the competitive window
#
# HOUR COST: 0h (not built)
# SLIDE CLAIM: "Skill library is hand-curated seed at step 0; Summary Agent is
#   deferred — in daVinci's own ablations, removal of summary RL is competitive
#   within the first 150 steps, which is our entire overnight window"

# ===================================================================
# DECISION 4: TURN REWARD Eq1 → LITE (bounded, binary PR)
# ===================================================================
# daVinci Eq1: R_{i,t} = C(y) * (1 + speedup_{i,t} + PR_{i,t})
# PR_{i,t} = T_generated / T_total  (continuous ratio via torch.profiler)
# Return G_{i,t} = sum_{t'>=t} gamma^{t'-t} R
#
# VERIFIED ISSUES with the full Eq1:
#   - torch.profiler attribution on H100 adds 50-200ms overhead per bench + can segfault
#     on unusual Triton kernels (confirmed in harness component red-team analysis)
#   - Continuous PR ≠ blocks the delegating-wrapper exploit (tl.dot launders matmul)
#   - In single-turn mode gamma=1, G_{i,t}=R_{i,t} so the return collapses to the reward
#
# LITE VERSION (canonical rewards.py, the single source of truth):
#
# rewards.py
P_TARGET    = 1.5   # daVinci-style normalizer; credit saturates near 1.5x
SPEEDUP_FLOOR = 1.1 # dead-band: no speedup credit below 1.1x (anti dtype/noise hack)
PR_BONUS    = 0.2   # binary: +0.2 iff pr_ok (Triton launches > 0 in timed pass
                    #         AND kernel passes delegates_to_matrix_unit AST check)
CORRECT_FLOOR = 0.3 # Kevin/audit: dense reward even at 1.0x speedup, keeps GRPO variance
BOOTSTRAP_COMPILE = 0.1   # annealed: +0.1 if code compiles (no allclose); zero after
BOOTSTRAP_IMPORT  = 0.2   # annealed: +0.2 if triton imported but no launch; zero after
BOOTSTRAP_ANNEAL_STEP = 30  # disable bootstrap credits after step 30

def compute_reward(
    allclose_correct: bool,  # seed_correct=42
    allclose_post: bool,     # seed_post=44 (materialized AFTER timing ends, anti async-overwrite)
    dtype_ok: bool,
    shape_ok: bool,
    launches_timed: int,
    speedup: float,          # wall-clock ratio vs PyTorch eager
    pr_ok: bool,             # binary PR gate (launch counter + AST delegate check)
    compiles: bool,
    imports_triton: bool,
    training_step: int = 0,
) -> dict:
    hard_caps = []

    # 4-layer hard cap (all must pass before any credit)
    if not (allclose_correct and allclose_post):
        hard_caps.append("incorrect")
        reward = 0.0
    elif not dtype_ok:
        hard_caps.append("dtype_mismatch")
        reward = 0.0
    elif not shape_ok:
        hard_caps.append("shape_mismatch")
        reward = 0.0
    elif launches_timed == 0:
        hard_caps.append("no_triton_launch")
        reward = 0.0
    elif speedup < SPEEDUP_FLOOR:
        hard_caps.append("below_speedup_floor")
        reward = 0.0
    else:
        # All gates pass: compute credit
        speedup_credit = min(speedup / P_TARGET, 1.0)  # clips at P_TARGET=1.5x
        pr_credit = PR_BONUS if pr_ok else 0.0
        reward = min(CORRECT_FLOOR + speedup_credit + pr_credit, 2.0)
        # 2.0 bound: at P_TARGET=1.5x + PR_BONUS=0.2 + floor=0.3 = 2.0 max
        # bounded [0, 2.0] for GRPO stability (no 3x effective-LR blowup)

    # Bootstrap credits: annealed, only if all hard caps pass except correctness
    # Purpose: manufacture group variance when base model is near-0 (calibration gate)
    if not hard_caps and training_step < BOOTSTRAP_ANNEAL_STEP:
        pass  # already has reward; bootstrap not needed
    elif training_step < BOOTSTRAP_ANNEAL_STEP and hard_caps == ["incorrect"]:
        # partial credit to avoid all-zero groups
        if imports_triton and launches_timed == 0:
            reward = BOOTSTRAP_IMPORT  # 0.2: imported triton but no launch
        elif compiles:
            reward = BOOTSTRAP_COMPILE  # 0.1: at least compiled

    return {"reward": reward, "hard_caps": hard_caps, "speedup": speedup, "pr_ok": pr_ok}

# WHY THIS WORKS AS A daVinci SUBSTITUTE:
# - C(y) gating is preserved (reward=0 on any hard cap failure)
# - speedup/P_TARGET term is daVinci's speedup_{i,t} normalized to [0,1]
# - PR_BONUS is the daVinci PR term reduced to binary (no profiler fragility)
# - CORRECT_FLOOR=0.3 ensures nonzero gradient for correct-but-slow kernels
# - Bounded [0,2]: at G=4 group, GRPO advantage = (r_i - mean)/1 which is
#   at most ~1.5 in magnitude; no gradient explosion risk
# UNIT TEST (test_rewards.py — must pass before kick):
#   assert compute_reward(False,...)[reward] == 0.0        # incorrect
#   assert compute_reward(True,True,True,True,1,1.0,True,True,True)[reward] == 0.3+1/1.5+0.2 clipped to 2.0
#   assert compute_reward(True,True,True,True,0,...)[reward] == 0.0   # no launch
#   assert compute_reward(True,True,True,True,1,1.05,...)[reward] == 0.0  # below floor

# ===================================================================
# DECISION 5: MULTI-TURN TRLOO vs SINGLE-TURN → SINGLE-TURN PRIMARY
# ===================================================================
# daVinci multi-turn: K=5 turns, gamma-discounted returns G_{i,t}=sum_{t'>=t} gamma^{t'-t} R
# TRLOO advantage: A_{i,t} = G_{i,t} - 1/(N_t-1) * sum_{j!=i} G_{j,t}  (Eq2)
# where N_t = total rollouts with >=t turns (trajectories that survived to turn t)
#
# WHY environment_factory IS BLOCKED:
#   trl issue #4543 (open, filed Nov 2025): multi-step environment_factory in vLLM server
#   mode breaks importance sampling because the trainer forces one shared prompt per
#   example; multi-turn trajectories have different prompt prefixes at each decision point.
#   In colocate mode the IS issue may be less severe, but the API is still EXPERIMENTAL
#   with no stability guarantee (trl docs: "may change or be removed without notice").
#
# SINGLE-TURN IS NOT A DEGRADED FALLBACK — it is the correct primary for this build:
#   - daVinci's ablation (Table 2 / Table 3): policy RL SINGLE-TURN (last_turn) still
#     reaches 26.1% Fast_1 on 8B (vs 37.2% for full daVinci) on Level 1; and more
#     importantly the BEST-TURN result from daVinci-8B is 42.1% (Table 3 best-turn)
#     which IS achievable by running the policy multiple times at inference and taking
#     best — this requires ZERO multi-turn training
#   - For the demo: "best of 4" at inference is free given group=4 rollouts in GRPO;
#     the demo can show the best-scoring kernel from 4 samples without any multi-turn RL
#   - The generalization MOAT is the held-out shape split, not the turn count
#
# PRIMARY PATH (implement this, time-boxed to 30 min):
#   Single-turn: prompt = task_description + skill_context (BM25 inject)
#                completion = full Triton kernel source
#                reward = compute_reward(grade(completion, op, shape))
#   trl handles the rest. K=1 turn, gamma=1.0, G_{i,t}=R_{i,t}=R_i.
#
# MANUAL MULTI-TURN STRETCH (implement ONLY if single-turn is green by 22:00 Sat):
#   50-line Python loop feeding reward_funcs — NOT environment_factory.
#   Works because each "turn" is just an extended prompt fed to reward_funcs
#   with the previous completion appended as context:
#
def manual_multiturn_rollout(policy, task, k_turns=2, grade_fn=grade_kernel_reward):
    # Returns a list of (full_prompt, completion, reward) for each turn
    # Used to build the training batch; reward_funcs receives the FINAL turn's reward
    # LOO advantage over the group dimension only (trl standard)
    history = task["prompt"] + skill_context
    best_reward = 0.0
    for turn in range(k_turns):
        completion = policy.generate(history, max_new_tokens=2048)
        reward_dict = grade_fn(task["op"], task["shape"], completion)
        reward = reward_dict["reward"]
        feedback = _build_feedback(reward_dict)   # "Speedup: 1.2x. No Triton launch detected."
        history += completion + "\n<feedback>" + feedback + "</feedback>\n"
        if reward > best_reward:
            best_reward = reward
            best_completion = completion
    # Return only the best turn for reward_funcs (LOO advantage over group dimension)
    return history, best_completion, best_reward
#
# CRITICAL NOTE: this loop is NOT environment_factory. It runs OUTSIDE trl's rollout
# manager, builds a dataset of (prompt, best_completion, best_reward) tuples, and then
# calls trainer.train() on that dataset with reward_funcs returning the pre-computed
# scalar. This is the "offline multi-turn with online reward" pattern — less sample-
# efficient than true online TRLOO but safe on the overnight path.
# The LOO advantage is still computed over group=4 (the n rollout dimension in daVinci
# Eq7 with k=0); the turn dimension contributes only via best-of-K selection, which
# daVinci Table 3 shows is a strong inference-time signal.

# ===================================================================
# DECISION 6: MRS FILTER (Eq3) → CUT
# ===================================================================
# MRS: w_t = exp(1/|T_i| * sum_k log pi_train(a_k|s_k) / pi_rollout(a_k|s_k))
# Purpose: filter samples with large importance ratio deviation (off-policy correction)
# WHY CUT: at single-turn with vLLM colocate mode and num_iterations=1 (default trl),
# the policy used for rollout IS the current policy; importance ratio = 1 by construction.
# MRS is only needed when rollout policy lags training policy (server mode, mu>1).
# With colocate mode + mu=1: MRS = no-op. Do not implement.

# ===================================================================
# DECISION 7: PRS SAMPLING (Eq4) → CUT
# ===================================================================
# PRS: p_{i,t} = clip((PR_{i,t} - tau)/s, 0, 1), sample proportional to p
# Purpose: prioritize tasks where Triton kernels dominate measured GPU time
# WHY CUT: binary PR gate already handles the same signal (reward=0 if no launch);
# continuous PRS requires torch.profiler per task which is fragile (see Decision 4).
# Replace with curriculum: start with elementwise ops (high PR by construction),
# then add matmul. This is a ~5-line config change, not a sampler rewrite.

# ===================================================================
# CALIBRATION GATE (mandatory pre-kick, runs Sat ~23:00)
# ===================================================================
# calibrate.py: runs before grpo_loop.py is allowed to start
#
def run_calibration_gate(grade_fn, task_dataset, n_samples=10):
    rewards = []
    for task in task_dataset[:5]:  # 5 ops x 2 shapes = 10 tasks
        for _ in range(n_samples):
            # sample a random completion from base model
            completion = base_model.generate(task["prompt"])
            r = grade_fn(task["op"], task["shape"], completion)["reward"]
            rewards.append(r)
    in_band = [r for r in rewards if 0.05 < r < 0.9]
    variance = float(np.var(rewards))
    pct_nonzero = sum(1 for r in rewards if r > 0) / len(rewards)
    # GATES (all must pass):
    assert variance > 0.001, f"ABORT: reward variance {variance:.4f} — all-zero groups, GRPO will not move"
    assert pct_nonzero >= 0.02, f"ABORT: only {pct_nonzero:.1%} nonzero — too sparse"
    assert len(in_band) / len(rewards) >= 0.10, "ABORT: <10% rewards in training band"
    print(f"CALIBRATION PASS: var={variance:.3f}, pct_nonzero={pct_nonzero:.1%}, in_band={len(in_band)/len(rewards):.1%}")
    return True

# ===================================================================
# STEP-50 ABORT (replaces step-150 — see revised schedule)
# ===================================================================
# In callbacks.py:
class AbortCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            self._reward_at_10 = state.log_history[-1].get("reward", 0.0)
        if state.global_step == 50:
            reward_now = state.log_history[-1].get("reward", 0.0)
            delta = reward_now - self._reward_at_10
            if delta < 0.05:
                print(f"ABORT at step 50: reward delta {delta:.3f} < 0.05 threshold")
                print("Switching to pre-recorded curve. Saving current checkpoint.")
                control.should_training_stop = True
        # Held-out eval every 25 steps
        if state.global_step % 25 == 0:
            self._run_heldout_eval(state.global_step)
        return control

# ===================================================================
# HOUR COST SUMMARY
# ===================================================================
# Decision 1 (LITE LOO via trl num_generations=4):     0.0h  (trl built-in)
# Decision 2 (CUT selection advantage):                 0.0h
# Decision 3 (CUT summary advantage):                   0.0h
# Decision 4 (LITE reward in rewards.py):               1.0h  (write + unit test)
# Decision 5 (single-turn primary):                     0.5h  (30-line reward_funcs callback)
# Decision 5 stretch (manual 2-turn loop):              1.0h  (only if single-turn green by 22:00)
# calibrate.py:                                         0.5h
# AbortCallback + held-out eval interleave:             0.5h
# grpo_loop.py / configs.py / grpo_modal.py:            1.0h
# Total on-clock training-loop work:                   ~3.5h  (fits Sat 21:00-00:30 with slack)

```

**Risks:** CRITICAL — trl issue #4543 (open): environment_factory multi-step IS broken in server mode. MITIGATED by: (1) unconditionally shipping reward_funcs single-turn; (2) manual 2-turn loop as stretch that bypasses environment_factory entirely; (3) colocate mode not server mode. Risk is fully mitigated IF we never call environment_factory.; HIGH — advantage collapse at G=4: with group=4 rollouts, if all 4 score 0 the group contributes zero gradient. MITIGATED by: calibration gate (blocks kick if variance=0); CORRECT_FLOOR=0.3 bootstrap; elementwise-first curriculum; BOOTSTRAP_COMPILE/IMPORT credits active until step 30. If collapse still occurs at runtime, it is silent (zero gradient) not a crash — monitor reward/std in trl logs.; HIGH — reward formula divergence between rewards.py and grade.py if one is edited independently. MITIGATED by: rewards.py is the SINGLE import in BOTH files (one source of truth); test_rewards.py asserts exact scalars; run in red-team checklist before kick. Solo build = self-inflicted risk, zero external coordination required to close it.; MEDIUM — binary PR gate (launch counter) tied to triton version. MITIGATED: (a) JITFunction.__call__ wrap is version-stable vs CudaDriver.launch internal; (b) smoke_test runs IN Modal image asserting launches>0 for known-good kernel AND 0 for never-launched stub; (c) triton pinned in Dockerfile.hud with build-time assert triton.__version__==X.Y.Z.; MEDIUM — 60-120 GRPO steps may not produce a statistically clean reward curve (noise band may overlap base). MITIGATED by: midnight kick (not 8AM) buys ~7-8h; at 3-9 min/step = 52-160 steps; step-50 abort switches to pre-recorded curve (06:00 Sunday harness run with base vs 3 manually improved kernels gives a clean before/after bar independent of GRPO); directional framing per audit.; LOW — scale_rewards=False removes variance normalization: per trl docs, update magnitudes then depend directly on raw reward scale. With reward bounded [0,2] this is controlled; monitor grad_norm in trl logs and add gradient_clip_val=1.0 in GRPOConfig if norms spike above 5.0.; LOW — daVinci Eq7 TRLOO vs trl's standard (r_i - mean(r))/std(r): trl's formula includes std-scaling by default; setting scale_rewards=False gives (r_i - mean(r)) which is LOO with group-mean baseline but no std normalization. This matches daVinci Eq2's A_{i,t} = G_{i,t} - 1/(N_t-1)*sum_{j!=i} G_{j,t} exactly at G=4, gamma=1 (single-turn). The formulas are identical; no algorithmic gap.

### Protean — Skill-Library-Lite: KEEP-LITE decision  (~1.5h)
**Goal:** Capture the single largest daVinci-kernel ablation gain (inference-time skill injection, Ablation 1: Level 2 Fast1 drops from 44.8% to 20.6% on 8B when removed) using a STATIC, read-only, BM25-indexed 5-skill JSONL library injected as a system-prompt prefix, at a cost of 1.5h during Modal warmup dead time. Explicitly excludes: Summary Agent, LLM rerank, Qwen3-Embedding-8B, online library growth. Degrades to null scheme on any failure. Honest demo claim: two daVinci techniques integrated (PR reward term + static skill injection), full co-evolution explicitly deferred.

**Files:**
- `kernelforge/skilllib/skills_v0.jsonl` — 5 hand-authored Triton optimization skills in daVinci five-field schema (name/description/scope/tags/content). Frozen at v0; never modified by the training run. Skills: fused_load_store (elementwise), shared_mem_reduction (reduction), tiled_mac_loop (matmul), l2_cache_hint (generic), persistent_kernel (generic). Each content field: 15-25 lines Triton code + 2-3 bullet annotations. Total ~200 lines JSONL.
- `kernelforge/skilllib/inject.py` — 50-line module: load_skills() reads skills_v0.jsonl; retrieve_skills(task_description, k=3) runs BM25Okapi on (name+description+tags) corpus, returns top-k with score>0.0; build_skill_prefix(task_description) assembles the system-prompt block, hard-capped at 10k chars (~2500 tokens). Returns empty string on any failure (null-scheme fallback). No GPU dependency, no LLM call, runs on control plane sub-millisecond.
- `kernelforge/env.py` — Modified at integration point only: two added lines before first yield — `skill_prefix = build_skill_prefix(op_description)` and `prompt = skill_prefix + newline + base_prompt if skill_prefix else base_prompt`. Import: `from skilllib.inject import build_skill_prefix`. Zero other changes to env.py.

**APIs:** rank_bm25.BM25Okapi(corpus: list[list[str]]) -> BM25Okapi — VERIFIED on PyPI v0.2.2; BM25Okapi.get_scores(query: list[str]) -> np.ndarray — returns per-document BM25 scores; build_skill_prefix(task_description: str) -> str — returns system-prompt prefix or empty string (null scheme); called once per task before first yield in env.py

**daVinci integration:** DECISION: KEEP-LITE. Grounds: daVinci Table 2 (verified from PDF pages 7-8) Ablation 1 (no skill injection at inference) produces the LARGEST single-component performance drop: Level 2 Fast1 44.8%->20.6% on 8B, Level 3 Fast1 10.1%->2.0%. This effect is largest in early training (Figure 3) — exactly the regime of 60-120 GRPO steps in a 24h window. A 5-skill static BM25 injector captures this gain. WHAT IS KEPT: inference-time skill injection (daVinci Section 3.3 Stage 1 BM25 pre-filtering only) + daVinci five-field schema (name/description/scope/tags/content) + JSONL versioned snapshot format (Section 3.2). WHAT IS CUT AND WHY: (1) LLM rerank (Section 3.3 Stage 2) — solving a non-problem; BM25 on 5 skills is exact selection; Ablation 4 shows BM25-only still captures meaningful signal; (2) Summary Agent (Section 3.5, Eq5/6) — requires joint RL training of summary head (Eq9) to be non-trivial; at step 60 the policy produces insufficient improvement signal to trigger R*>1.2*r1; would cost 3-4h and produce no verified skills; (3) Qwen3-Embedding-8B + HDBSCAN — incompatible with single-H100 VRAM budget (7B policy + LoRA + vLLM KV cache already ~50-60GB); (4) Online library growth — logically downstream of a stable GRPO run, not a prerequisite; (5) Skill verification execution re-runs (Eq6) — requires re-running policy on original task with candidate skill, adds 30-90s per skill candidate, produces nothing without the Summary Agent. SLIDE CLAIM (exact, honest): 'We integrated two daVinci-kernel techniques (arXiv 2606.16497, published June 15 2026, one week before this hackathon): the profiling-ratio reward term as a programmatic anti-hack gate, and inference-time skill injection from a 5-skill BM25-indexed library. Ablation 1 from the paper confirms skill injection is the largest single-component gain; we capture it with a static library that degrades gracefully to the null scheme. Full co-evolution (Summary Agent, joint multi-agent RL) is explicitly future work.'

```python
# ===== kernelforge/skilllib/inject.py (complete, 50 lines) =====
import json, re
from pathlib import Path
from rank_bm25 import BM25Okapi  # pip: rank-bm25>=0.2.2, pure Python, Apache2.0

_SKILLS_PATH = Path(__file__).parent / "skills_v0.jsonl"
_MAX_SKILL_CHARS = 10_000  # hard cap: never exceed ~2500 tokens of skill context

def _tok(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())

def load_skills(path: Path = _SKILLS_PATH) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def retrieve_skills(task_description: str, k: int = 3) -> list[dict]:
    """BM25 top-k from static seed library. With |L|=5 and k=3, this is
    effectively exact selection — BM25 prunes genuinely irrelevant skills
    (e.g. tiled_mac_loop for a pure elementwise task). No LLM call needed
    at library size 5; LLM rerank is solving a non-problem here."""
    try:
        skills = load_skills()
        if not skills:
            return []
        corpus = [_tok(s["name"] + " " + s["description"] + " " + " ".join(s["tags"]))
                  for s in skills]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tok(task_description))
        ranked = sorted(zip(scores, skills), key=lambda x: -x[0])
        return [s for score, s in ranked[:k] if score > 0.0]
    except Exception:
        return []  # fail-open: return empty -> null scheme, never crash a rollout

def build_skill_prefix(task_description: str) -> str:
    """Returns string to prepend to system prompt. Empty string = null scheme.
    Integration: prompt = (skill_prefix + '\n\n' + base_prompt) if skill_prefix else base_prompt"""
    selected = retrieve_skills(task_description)
    if not selected:
        return ""
    lines = ["## Optimization Skills (apply these where relevant)\n"]
    for skill in selected:
        lines.append(f"### {skill['name']}\n{skill['content']}\n")
    block = "\n".join(lines)
    return block[:_MAX_SKILL_CHARS] + "\n...[truncated]\n" if len(block) > _MAX_SKILL_CHARS else block

# ===== skills_v0.jsonl (5 hand-authored entries, one per line) =====
# {"name":"fused_load_store","description":"Vectorized tl.load+store with eviction hints for fused elementwise ops","scope":"elementwise","tags":["vectorize","memory","fused"],"content":"```triton\n@triton.jit\ndef fused_ew_kernel(x_ptr,y_ptr,out_ptr,N,BLOCK:tl.constexpr):\n    pid=tl.program_id(0)\n    offs=pid*BLOCK+tl.arange(0,BLOCK)\n    mask=offs<N\n    x=tl.load(x_ptr+offs,mask=mask,other=0.,eviction_policy='evict_last')\n    y=tl.load(y_ptr+offs,mask=mask,other=0.,eviction_policy='evict_last')\n    tl.store(out_ptr+offs,x+y,mask=mask)\n```\n- Use BLOCK=1024 for L2-fitting; evict_last avoids cache pollution for streaming fused ops.\n- Launch: grid=(cdiv(N,BLOCK),); no shared memory needed for pure elementwise."}
# {"name":"shared_mem_reduction","description":"Warp-level tl.sum reduction via shared memory tile, avoids atomic fallback","scope":"reduction","tags":["reduction","shared_memory","sum"],"content":"..."}
# {"name":"tiled_mac_loop","description":"Outer-product tile loop for matmul with configurable BLOCK_M/N/K and autotuner stub","scope":"matmul","tags":["matmul","tiling","autotune"],"content":"..."}
# {"name":"l2_cache_hint","description":"tl.load cache_modifier='.ca' for data reused across CTA; '.cs' for streaming","scope":"generic","tags":["cache","memory","l2"],"content":"..."}
# {"name":"persistent_kernel","description":"Grid-stride loop over flattened output; one CTA handles multiple tiles to amortize launch overhead","scope":"generic","tags":["persistent","grid_stride","throughput"],"content":"..."}

# ===== env.py integration (2 lines added, zero other changes) =====
# from skilllib.inject import build_skill_prefix   # add to imports
# ...inside @env.template, before `kernel = yield build_prompt(...)`:
# skill_prefix = build_skill_prefix(op_description)
# prompt = (skill_prefix + "\n\n" + base_prompt) if skill_prefix else base_prompt
# kernel = yield prompt

# ===== G0 preflight smoke test addition (Friday night) =====
# assert build_skill_prefix("fused elementwise gelu activation") != ""
# assert build_skill_prefix("nonexistent zyx quux") == ""  # prunes zero-score skills
```

**Risks:** rank-bm25 import failure in Modal image: MITIGATED — pure Python, no C extension, install after triton in Dockerfile; `import rank_bm25` asserted in G0 smoke_test before Sat 12:30.; skills_v0.jsonl malformed (bad JSON line crashes load_skills): MITIGATED — load_skills() wrapped in try/except returning []; build_skill_prefix returns '' (null scheme); rollout never crashes.; Skill context exceeds 7B model's practical context window (4-8k tokens): MITIGATED — 10k char hard cap in build_skill_prefix (~2500 tokens); 5 skills x ~500 tokens each = well within limit.; BM25 retrieves wrong skills for an op (e.g. tiled_mac_loop for elementwise task): LOW RISK — only skills with score>0.0 are returned; BM25 on (name+description+tags) correctly separates scopes. Worst case: slightly irrelevant skill injected = no worse than no skill (null scheme) since the model ignores irrelevant context.; Time overrun: 1.5h estimate is generous for 50 lines of code + 5 hand-authored JSONL entries. If behind schedule Saturday, cut to 3 skills (further reducing JSONL authoring time to 30min) — minimum viable is 1 skill (fused_load_store for elementwise, the most common op in L1 curriculum).; Claim dishonesty: claiming 'daVinci skill library' when it is a static 5-entry JSONL is a judge-legibility risk. MITIGATED — slide says exactly 'static hand-curated 5-skill library' and explicitly calls out what is NOT implemented. Precision here is a strength, not a weakness.

### SFT Cold-Start Decision for Protean  (~2h)
**Goal:** Decide whether to run a supervised fine-tuning cold start on Dr.Kernel/KernelGYM data before GRPO, apply a daVinci-style diversity filter, or skip SFT entirely and bootstrap with partial-credit reward shaping. Produce a concrete KEEP/CUT/LITE verdict with exact implementation and hour cost grounded in the 24h solo constraint and verified data availability.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/sft_coldstart.py` — LITE path only: 60-line script that downloads hkust-nlp/drkernel-coldstart-8k, filters by final_speedup>=1.2 AND entry_point in OPS_SUBSET, truncates messages to max_length=8192, and writes a trl-compatible JSONL for SFTTrainer. No diversity filter. Estimated 1.5h total including download.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — Mandatory pre-GRPO gate (mirrors verilog-template check_calibration.py pattern): samples base or SFT model on 10x per op for 3 ops, checks compilable rate >= 5%, allclose rate >= 2%, nonzero group reward variance. GO/NO-GO output. Run this BEFORE deciding whether SFT was sufficient.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_loop.py` — GRPO trainer with CORRECT_FLOOR=0.3 partial-credit bootstrap annealed over first 20 steps, which replaces SFT if calibrate.py shows sufficient variance without it.

**APIs:** datasets.load_dataset('hkust-nlp/drkernel-coldstart-8k', split='train') -- MIT, verified 8920 rows, messages/final_speedup/entry_point columns; trl.SFTTrainer with SFTConfig(max_seq_length=8192) -- NOT 18432 (VRAM constraint for 7B LoRA on H100); peft.LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj']) -- standard 7B LoRA; calibrate.py gate: compilable>=5% AND allclose>=2% AND reward_std>0 on at least one op; rewards.CORRECT_FLOOR=0.3 bootstrap (annealed to 0 at step 20) as fallback if SFT insufficient

**daVinci integration:** WHAT IS PORTED (lite, verified buildable in 2h): (1) Lazy-skill filter analog: filter drkernel-coldstart-8k by final_speedup >= 1.2, rejecting pytorch-fallback/trivial trajectories exactly as daVinci Section 3.5 lazy-skill filter rejects 'fall back to PyTorch' content. Honest claim: 'daVinci-style execution-quality threshold applied to cold-start data selection.' (2) Dr.Kernel trajectory reuse: daVinci Section 3.6 explicitly states it builds SFT data FROM the Dr.Kernel cold-start dataset filtered by trigger condition (Eq5). We do the same using drkernel-coldstart-8k (MIT license). Honest claim: 'same source dataset as daVinci cold start, without the 3-phase re-generation pipeline that requires GPT-5.4.' WHAT IS CUT AND WHY: (1) Qwen3-Embedding-8B + HDBSCAN diversity filter: requires loading a separate 8B model on the H100, ~30-60 min extra, VRAM conflict with SFT training. Ablation2 shows it mainly helps at Fast1.5+, not our target. CUT. (2) 3-phase Summary SFT (generate update_skill_library tool calls): requires GPT-5.4 as policy to generate quality demonstrations, then re-injecting skills into Phase 3 examples. No daVinci SFT dataset released. Estimated 4-6h to replicate. CUT. (3) Selection SFT (select_skills tool call demonstrations): requires the seed skill library to already exist. In our LITE skilllib (hand-authored seed JSONL), BM25 retrieve + inject replaces this entirely. CUT. HONEST SLIDE CLAIM: 'We initialize from the publicly-released Dr.Kernel cold-start trajectories (hkust-nlp/drkernel-coldstart-8k, MIT), filtered by final_speedup >= 1.2 as a quality gate analogous to daVinci's lazy-skill rejection criterion. The full daVinci 3-phase SFT pipeline (GPT-5.4 generation + Qwen3-Embedding-8B diversity filter) is out of scope for the 24h window; the calibration gate and CORRECT_FLOOR=0.3 bootstrap serve as the fallback if this filtered warm-start is insufficient.'

```python

# ============================================================
# DECISION: LITE — conditional SFT, data-filter only, no diversity filter
# ============================================================
#
# VERIFIED FACTS (grounding this decision):
# - hkust-nlp/drkernel-coldstart-8k: MIT license, 8920 rows, ~163MB Parquet,
#   5-turn Triton trajectories, columns: messages/final_speedup/entry_point/...
#   Source: https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k
# - daVinci SFT pipeline (Section 3.6): uses GPT-5.4 as policy, Qwen3-Embedding-8B
#   + HDBSCAN diversity filter. NO pre-built daVinci SFT dataset released publicly.
#   The daVinci GitHub repo (GAIR-NLP/daVinci-kernel) returns HTTP 404 as of 2026-06-21.
# - KernelGYM license: Apache 2.0 (confirmed). Dr.Kernel dataset: MIT.
# - Base Qwen2.5-Coder-7B compilable rate on Triton kernel tasks: NOT published.
#   Phase14 audit estimates 1-5% Fast1; daVinci Ablation5 (no skills, after RL) is 51.6%
#   Level2 Fast1 on Qwen3-8B -- incomparable to zero-shot 7B base.
# - SFT training cost estimate on one H100 with LoRA:
#   8K examples x ~4K avg token length (truncated from 18432 at 8192) x 4 epochs
#   ~ 8000 * 8192 * 4 / (H100 ~50K tok/s) = ~52 min. Full fine-tune: ~2-3h.
#   This is WITHIN the Friday-evening pre-build window (G0), not on the 24h clock.
# - daVinci diversity filter: Qwen3-Embedding-8B + HDBSCAN. Cost: loading a separate
#   8B embedding model on the same H100 as training = memory conflict + 30-60 min extra.
#   BLOCKED in 24h solo window. Ablation2 (unfiltered) shows only mild degradation
#   on Fast1 level 1/2; the penalty concentrates at Fast1.5/Fast2 -- not our target.
#
# ============================================================
# THE DECISION: LITE (conditional on calibrate.py gate)
# ============================================================
#
# RULE: Run SFT ONLY if calibrate.py on the raw base model fails the gate
#       (compilable < 5% OR group reward variance == 0 on the fused-elementwise op).
#       If the base model passes calibrate.py, SKIP SFT entirely and let
#       CORRECT_FLOOR=0.3 bootstrap GRPO directly.
#
# RATIONALE:
# (1) CUT full daVinci SFT pipeline: daVinci's 3-phase SFT requires GPT-5.4 as the
#     policy generator, Qwen3-Embedding-8B, and HDBSCAN -- none available in-window.
#     The daVinci SFT dataset itself is NOT released. Attempting to re-run the pipeline
#     would require generating fresh trajectories with an API model (~hours + $$$) and
#     running the diversity filter (~1h extra + 16GB VRAM conflict). BLOCKED.
#
# (2) CUT KernelGYM trajectory re-collection: collecting fresh Dr.Kernel-style
#     multi-turn trajectories requires KernelGYM server running + a strong policy
#     (GPT-5.4 or equivalent). Not feasible in 24h solo without API costs and time.
#
# (3) KEEP (conditional) filtered drkernel-coldstart-8k download:
#     - Available: MIT license, 163MB, instant download via datasets.load_dataset()
#     - Filter to our 3 ops by entry_point (or 'elementwise'/'matmul'/'softmax' substring)
#       AND final_speedup >= 1.2 (removes lazy/trivial trajectories, daVinci lazy-skill filter analog)
#     - Truncate to max_length=8192 (not 18432) to fit 7B LoRA training
#     - Run trl SFTTrainer with LoRA (r=16, alpha=32), 2 epochs, lr=2e-4
#     - Estimated wall-clock: ~50-70 min on H100 with LoRA
#     - This lifts compilable rate from estimated 1-5% to ~30-50% (Dr.Kernel paper
#       shows Cold-Start-8B* at 11.4% Fast1 Level1 before RL, vs base Qwen3-8B* at 7.5%)
#     - CLAIMED lift is ~2-4x compilable rate; treat as directional, not guaranteed for 7B
#
# (4) SKIP daVinci diversity filter (Qwen3-Embedding-8B + HDBSCAN):
#     - Ablation2 shows the penalty for skipping is primarily at Fast1.5/Fast2, not Fast1
#     - We target Fast1 (any speedup > 1x) for the calibration gate, not precision speedups
#     - Loading Qwen3-Embedding-8B alongside SFTTrainer on 80GB H100 requires careful
#       sequencing and adds ~30-60 min; not worth the marginal gain for a 24h demo
#     - CLAIM: "we applied the daVinci speedup-threshold filter (final_speedup >= 1.2)
#       as a lightweight analog of daVinci's execution-verification gate" -- honest and
#       defensible without the full HDBSCAN pipeline
#
# (5) ALTERNATIVE to SFT: CORRECT_FLOOR=0.3 partial-credit bootstrap
#     - If calibrate.py shows >=5% compilable AND nonzero variance, skip SFT entirely
#     - The 0.3 floor gives a gradient signal even for incorrect-but-compilable kernels
#     - Anneal CORRECT_FLOOR to 0.0 after step 20 so the signal stays honest
#     - This is the cleaner path if the base model can write syntactically valid Triton
#       (Qwen2.5-Coder-7B-Instruct has seen Triton code in pretraining)
#
# ============================================================
# IMPLEMENTATION (sft_coldstart.py, 60 lines)
# ============================================================

OPS_SUBSET = {'fused_elementwise', 'softmax', 'layer_norm'}  # match 3 of our 3 ops

def filter_dataset(ds):
    # daVinci lazy-skill analog: reject low-speedup trajectories
    ds = ds.filter(lambda x: float(x['final_speedup'] or 0) >= 1.2)
    # op-scope: keep only trajectories for our 3 ops
    ds = ds.filter(lambda x: any(op in (x.get('entry_point') or '') for op in OPS_SUBSET))
    # truncate messages to fit 7B LoRA context
    # messages is a list of {role, content} dicts; keep all 10 but truncate content
    return ds

def run_sft(model_name='Qwen/Qwen2.5-Coder-7B-Instruct', output_dir='sft_ckpt'):
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset('hkust-nlp/drkernel-coldstart-8k', split='train')
    ds = filter_dataset(ds)
    # Expected: ~300-800 rows after filtering (est: ~10% match our 3 ops, ~70% pass speedup gate)
    # If < 100 rows: lower speedup threshold to 1.1 or expand OPS_SUBSET

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj'],
                          lora_dropout=0.05, task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=2,         # not 4: 24h constraint
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            max_seq_length=8192,        # not 18432: VRAM constraint
            logging_steps=10,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

# ============================================================
# CALIBRATION GATE (calibrate.py) -- runs AFTER SFT or on base
# ============================================================
# Mirrors verilog-template check_calibration.py pattern.
# Criterion: sample 10 rollouts per op for 3 ops using the current model.
# REQUIRE: compilable_rate >= 0.05 AND allclose_rate >= 0.02 AND reward_std > 0.0
# on at least ONE op. If gate fails after SFT, lower speedup threshold in rewards.py
# and add +0.1 compile credit. If gate fails on base without SFT, trigger SFT.

def calibration_gate(model, harness, ops, n=10):
    results = {}
    for op in ops:
        rewards = []
        for _ in range(n):
            kernel_src = model.generate(build_prompt(op, sample_train_shape(op)))
            r = harness.grade(kernel_src, op)
            rewards.append(r)
        compilable = sum(1 for r in rewards if r['compilable']) / n
        allclose = sum(1 for r in rewards if r['allclose']) / n
        std = statistics.stdev(r['reward'] for r in rewards) if len(rewards) > 1 else 0.0
        results[op] = {'compilable': compilable, 'allclose': allclose, 'std': std}
    gate_pass = any(
        v['compilable'] >= 0.05 and v['allclose'] >= 0.02 and v['std'] > 0.0
        for v in results.values()
    )
    return gate_pass, results

# ============================================================
# SCHEDULE PLACEMENT
# ============================================================
# G0 (Friday evening, OFF CLOCK):
#   Step 1: Modal image build + LaunchCounter smoke test
#   Step 2: Run calibrate.py on BASE Qwen2.5-Coder-7B-Instruct (10 rollouts x 3 ops)
#   Step 3a: IF gate passes -> SKIP SFT, proceed to Sat 12:30 with base model
#   Step 3b: IF gate fails -> run sft_coldstart.py (~60-70 min on H100) then
#            re-run calibrate.py on SFT checkpoint; if still fails, enable
#            CORRECT_FLOOR=0.3 + compile-credit bootstrap and proceed anyway
#   HARD RULE: Do NOT run SFT during the 24h window. It must complete Friday night
#              or be declared unnecessary by the calibration gate.
# Hour A (Sat 12:30): Start with whichever model passed the gate.

```

**Risks:** VERIFIED RISK: drkernel-coldstart-8k covers KernelBench L1/L2 ops (matmul, softmax, layernorm, conv, etc.) but our op subset may have too few rows after filtering. MITIGATION: if filtered rows < 100, lower speedup threshold to 1.1 or expand OPS_SUBSET to 5 ops; if < 50, skip SFT entirely and rely on CORRECT_FLOOR bootstrap.; VERIFIED RISK: daVinci SFT dataset NOT released publicly (GAIR-NLP/daVinci-kernel GitHub returns 404). No daVinci-specific cold-start data available. We can only use drkernel-coldstart-8k as the source. CLAIM: call this 'Dr.Kernel-initialized policy' not 'daVinci-initialized.'; CLAIMED (not verified): SFT on 7B lifts compilable rate from ~1-5% to ~30-50% based on Dr.Kernel's Cold-Start-8B numbers. The actual lift for Qwen2.5-Coder-7B is unknown because the base is a different model family (Qwen2.5 vs Qwen3). Treat as directional. The calibration gate catches failure.; RISK: max_seq_length=8192 truncates the 5-turn trajectories (recommended 18432 by drkernel). Truncation='right' means the final turns (refinements) get cut -- exactly the turns that show optimization. MITIGATION: truncate='left' to keep the later refinement turns, or reduce to 3 turns per example by slicing messages[:6]. Budget: adds ~10 min of data prep code.; RISK: SFT checkpoint trained on Dr.Kernel's op distribution (KernelBench tasks) may not match our 3 ops exactly. The policy may be overtrained on ops not in our task bank. MITIGATION: the calibration gate catches this -- if allclose_rate on our ops is still near-zero after SFT, the gate fails and we fall back to CORRECT_FLOOR bootstrap.; SCHEDULE RISK: if SFT is triggered during G0 (Friday night) and takes >2h, the builder loses sleep before the 24h window. MITIGATION: cap SFT at 2 epochs (not 4 as in daVinci), LoRA not full fine-tune. At 50-70 min estimated, this is acceptable. Hard rule: if SFT has not completed by 11PM Friday, abort and use CORRECT_FLOOR bootstrap.; RISK: diversity filter (Qwen3-Embedding-8B + HDBSCAN) is CUT. Ablation2 in daVinci Table 2 shows unfiltered SFT underperforms at Fast1.5/Fast2 but is competitive at Fast1. Since our calibration gate targets Fast1 (any speedup), this cut is safe. Do NOT claim diversity-filtered cold start -- be honest on the slide: 'speedup-threshold filtered (final_speedup >= 1.2)'.

## Round 3 — Finalization


### Protean — Hour-by-hour build schedule (Sat 12:30 -> Sun 13:00, 24h solo) with go/no-go gates and the Sun 08:00 GRPO kick  (~24h)
**Goal:** Execute the locked Protean scope (held-out continuous-shape generalization verifier as the moat; exactly TWO daVinci techniques integrated — multiplicative binary PR/launch gate + post-training static skill prefix) on a single Modal H100 within 24h solo, such that: (1) the HUD two-yield env clones cleanly from verilog-template, (2) rewards.py is the single byte-identical source of truth imported by BOTH grade.py and kernel_env.py (hash-pinned, preflighted), (3) the 4-layer anti-hack red-team asserts reward~0 on passthrough/never-launched/try-except/bf16 hacks BEFORE the kick, (4) a hard two-stage calibration gate (compile+allclose+variance AND a nonzero-speedup-reward gate on >=2 of 5-10 tasks) blocks GRPO until signal exists, (5) overnight single-turn GRPO (LoRA + vLLM colocate) kicks by Sun 08:00 with a step-150 / flat-curve abort, and (6) a base-vs-trained reward delta on held-out shapes is produced by Sun 13:00, with a pre-recorded curve fallback by 06:00 regardless. The deliverable is the non-gameable verifier and the train-vs-held-out gap slide, not a daVinci reproduction.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/rewards.py` — SINGLE SOURCE OF TRUTH for reward. compute_reward(correct, speedup, pr_frac, launches, dtype_ok, shape_ok) -> dict. Multiplicative binary PR/launch gate (tau=0.5), CORRECT_FLOOR=0.3, SPEEDUP_FLOOR=1.1, P_TARGET=1.5, speedup hard-capped at 20x, fresh-inputs-per-timed-iter protocol constants. NOT a pip package; flat COPY into image. Hash-pinned via REWARDS_HASH.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tasks/<op>/donotaccess/grade.py` — Hidden root:700 verifier, importlib-loaded by grader.py (spec_from_file_location). Top: sys.path.insert(0,'/donotaccess'); from rewards import compute_reward; assert sha256(rewards.py)==REWARDS_HASH else HALT. Runs subprocess bench, allclose on held-out shapes, returns dict with reward/hard_caps/subscores.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tasks/<op>/donotaccess/reference.py` — Hidden PyTorch eager reference module per op (KernelBench Model/get_inputs/get_init_inputs contract). Never copied into agent tree.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/grader.py` — Public mapper lifted verbatim from verilog grader.py: _load_grade_module + hand-built EvaluationResult (NO combine(), negative-weight hard_cap_penalty reconcile, fail-closed try/except -> reward 0).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/env.py` — HUD two-yield @env.template (clone of verilog env.py). NO from __future__ import annotations. params typed plain str. _AgentWorkspace setpriv uid wall. yield prompt; yield evaluate_task(task_id).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scenario_helpers.py` — Clone of verilog scenario_helpers: _resolve_workspace_root, HIDDEN_ROOT=/donotaccess, hidden_dir(), setup_task() that renders the shape-specific reference into the workspace.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/bench_core.py` — do_bench-style timing: locked clocks (@modal.enter), L2 flush, FRESH random inputs per timed iteration (defeats scratchpad cache), median-of-N, CUDA events. Returns (eager_ms, kernel_ms, pr_frac).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/anti_hack.py` — 4-layer gate: L1 AST ban (torch.matmul/F.softmax/torch.compile/aten), L2 dtype+shape match, L3 Triton JITFunction LaunchCounter monkeypatch (>0 train AND timed), L4 measured Triton GPU-time fraction > tau.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/subprocess_runner.py` — Isolated subprocess kernel exec with 60-90s timeout, fail-closed -> reward 0 (illegal-memory-access kills subprocess not rollout).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/modal_app.py` — Modal H100 service lifted from ml-template/modal_runner.py: @app.function(image=image, gpu='H100', secrets=[hud-keys]). Warm class @modal.enter locks clocks + warms TRITON_CACHE.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/sampler.py` — Shape sampler: hashlib.sha256-seeded RNG (NOT builtin hash) -> reproducible shape sets across control-plane and workers.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/splits.py` — THE MOAT. train grid M in {256,512,1024,2048}; test continuous M in {384,768,1536}+. _assert_split_disjoint() at import AND in tests AND in freeze().
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/manifest.py` — Freeze sampled shapes to versioned JSONL (daVinci snapshot pattern); loop consumes FROZEN manifest, never re-samples.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_modal.py` — Modal launcher for the overnight run: H100, vLLM colocate, TRITON_CACHE_DIR baked, mounts kernelforge + manifest.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_loop.py` — trl GRPOTrainer: LoRA, num_generations=8 (drop to 4 if OOM), gpu_memory_utilization tuned, single-turn primary (reward_funcs) with environment_factory multi-turn as 1h-budgeted stretch.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/reward.py` — trl reward_func wrapper: parses completion -> kernel src -> calls grade.py path (SAME rewards.py). Per-(op,shape) eager baseline cached.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/callbacks.py` — RewardCurveLogger (held-out eval rollouts every 25 steps) + step-150/flat-curve auto-abort callback.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — Two-stage GO/NO-GO gate: (A) hash preflight + known-good>0/known-bad==0 + compile>=5%/allclose>=2%/variance>0; (B) >=1 nonzero-SPEEDUP-reward rollout on >=2 of 5-10 tasks. P_TARGET escalation ladder 1.5->1.2->1.1->L1-elementwise.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scripts/redteam.py` — Asserts reward~0 on the 4 named hacks: passthrough, never-launched, try-except-torch-fallback, bf16-downcast. Build-blocking before 08:00 kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/Dockerfile.hud` — Lift verilog Dockerfile.hud + ml-template image. COPY rewards.py /donotaccess/rewards.py; bake TRITON_CACHE_DIR + warmup-compile reference shapes; pin triton + hud SDK commit.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/skilllib/skills_v0.jsonl` — CONTINGENT, hour 16-18 only: 1-3 hand-distilled skills from real successful rollouts, injected unconditionally only for the slide hero kernel. Empty == null scheme.

**APIs:** hud.Environment(name='kernelforge-v6')  # string literal name; hud deploy statically parses it; @env.template(id=...) async def task(op:str, M:str, N:str, dtype:str)  # NO from __future__ import annotations; params plain str (Literal/Optional crash deploy -> PydanticUserError -32000); two-yield: answer = yield prompt; ... ; yield evaluate_task(...); _AgentWorkspace(Workspace).shell_argv -> setpriv --reuid agent --regid --clear-groups (uid wall; no-op off-root); importlib.util.spec_from_file_location(f'{task_id}_hidden_grade', hidden_dir/'grade.py')  # loads root:700 verifier; hud.graders.EvaluationResult / SubScore  (v6 home; was hud.tools.types); DO NOT use hud.graders.combine() (renormalizes weights, erases correctness hard cap); build EvaluationResult by hand + negative-weight hard_cap_penalty; scenario_helpers._resolve_workspace_root() / HIDDEN_ROOT=/donotaccess / hidden_dir(task_id)  # /workdir on image, per-pid tmp local; modal.App; @app.function(image=image, gpu='H100', secrets=[modal.Secret.from_name('hud-keys', required_keys=['HUD_API_KEY'])]); modal.Image.from_dockerfile(...) with TRITON_CACHE_DIR baked + warmup-compile; @modal.enter() lock GPU clocks (nvidia-smi -lgc) + warm TRITON_CACHE; triton.runtime JITFunction.__call__ monkeypatch (LaunchCounter); pin triton version; torch.cuda.Event timing + L2 flush + torch.Generator('cuda').manual_seed(BASE+i) fresh per iter; trl.GRPOTrainer(reward_funcs=[...], num_generations=8, use_vllm=True, peft LoRA); environment_factory = 1h-budget multi-turn stretch; hashlib.sha256(canonical_str)  # shape-sampler RNG seed AND rewards.py hash pin (NOT builtin hash / PYTHONHASHSEED)

**daVinci integration:** EXACTLY TWO daVinci-kernel techniques are integrated, both confirmed against the locked scope and both folded into work already on the critical path (zero new serial nodes). (1) PROFILING-RATIO / LAUNCH GATE as a MULTIPLICATIVE BINARY GATE inside rewards.py (~0.5h, folded into Block 1's reward build): pr_gate=1.0 iff (Triton JITFunction launch count >0) AND (measured Triton GPU-time fraction > tau=0.5), else 0.0; reward = CORRECT_FLOOR + speedup_score*pr_gate. This replaces daVinci Eq1's ADDITIVE continuous PR (R=C*(1+speedup+PR)) because at single-op scope PR~1.0 is a constant offset that GRPO group-relative advantage cancels to ~2e-16, and the additive form trains launch-count maximization (torch wrapped in trivial @triton.jit). The binary gate closes the torch-passthrough exploit a bare launch counter misses (a kernel can fire without doing the compute) at zero extra GPU cost (read from the same timing/profiler pass). Fail-closed: profiler NaN/timeout => pr_gate=0 => reward 0. (2) STATIC SKILL-INJECTION PREFIX as a POST-TRAINING, demo-only artifact (~0.5h, Block 7, hour 16-18, CONTINGENT): only if the overnight run produced >=3 successful rollouts (allclose AND speedup>=1.5x), hand-distill 1-3 into skills_v0.jsonl and inject unconditionally only when generating the slide hero kernel; empty library == null scheme == graceful degradation. EXPLICITLY CUT: LITE Selection Agent (zero signal at <=150 steps), per-agent LOO adapter (trl GRPO LOO is already the default — daVinci framing is 0 new code, documentation not work), Summary-Agent online library growth + Eq5/6 verification, hour-0 hand-authored skills_v0.jsonl + BM25 inject.py (circular: needs successful rollouts that don't exist yet, and inert without SFT skill-conditioning per daVinci Sec 3.4/3.6), and SFT cold-start on drkernel-8k (the pre-GRPO calibration gate already provides go/no-go; schema-probed at hour 0.5 but default OFF). Honest demo claim locked: "two daVinci techniques integrated — profiling-aware launch gate + static skill prefix at inference; full co-evolution explicitly deferred." NO co-evolution claim. The genuine contribution stays the held-out continuous-shape generalization verifier (KernelBench roadmap #74, unshipped); daVinci (published one week pre-hackathon) is cited as currency-with-literature, converting the "did you just reproduce daVinci at 7B?" question from liability to strength.

```python

# =====================================================================
# HOUR-BY-HOUR SCHEDULE  (Sat 12:30 -> Sun 13:00)  solo, 1x H100
# Serial spine ~13.5h of real work; the rest is overnight GPU time + slack.
# Times are wall-clock. [GATE] = hard go/no-go. Credits: HUD $200, Modal $250.
# =====================================================================

# ---- BLOCK 0: HOUR 0  (Sat 12:30-13:30)  SETUP + DECONFLICT --------
# 12:30 git clone verilog-template -> kernelforge/. Strip verilog tasks.
# 12:35 `hud` SDK commit PIN (note exact sha in pyproject). triton version PIN.
# 12:40 LICENSE check: KernelGYM (reuse y/n in 10 min). If license/API mismatch
#        -> DO NOT reuse; harness becomes critical path, use settrace launch-count
#        fallback. Decision logged.
# 12:50 Modal auth; `modal secret create hud-keys HUD_API_KEY=...`. Smoke a
#        trivial @app.function(gpu="H100") that prints torch.cuda + nvidia-smi.
# 13:10 SCHEMA PROBE (10 lines): download drkernel-coldstart-8k FIRST 100 rows,
#        print columns/dtypes/final_speedup dist. Result only gates the (cut-by-
#        default) SFT contingency. SFT stays OFF.
# 13:20 Pick 5 ops: 1xL1-elementwise(safety floor), 3xL2-fused, 1x reduction.
# [GATE-0  13:30]: Modal H100 reachable AND torch.cuda.is_available()==True AND
#        secret resolves. NO-GO=> burn time on Modal support, do NOT proceed.

# ---- BLOCK 1: HOURS 1-3 (13:30-16:30)  REWARDS.PY + ENV SKELETON ---
# rewards.py FIRST (it is the divergence risk). Then env clones.
def compute_reward(correct, speedup, pr_frac, launches, dtype_ok, shape_ok,
                   P_TARGET=1.5, SPEEDUP_FLOOR=1.1, TAU=0.5,
                   CORRECT_FLOOR=0.3, SPEEDUP_CAP=20.0):
    caps=[]
    if not (correct and dtype_ok and shape_ok): return _r(0.0,["incorrect"])
    if launches<=0:                             return _r(0.0,["no_triton_launch"])
    pr_gate = 1.0 if pr_frac>TAU else 0.0          # MULTIPLICATIVE binary gate
    if pr_gate==0.0:                            return _r(0.0,["pr_gate"])
    if speedup<SPEEDUP_FLOOR: speedup_score=0.0    # dead-band, anti dtype/noise
    else: speedup_score=min(speedup,SPEEDUP_CAP)/P_TARGET
    reward = CORRECT_FLOOR + speedup_score*pr_gate # corr is multiplicative floor
    return _r(reward, caps)
# Timing protocol (bench_core): FRESH inputs EVERY iter -> defeats cache exploit
#   for i in range(N): inp=randn(shape, generator=Generator('cuda').manual_seed(BASE+i))
#                      t_i=cuda_event_time(kernel,inp); assert allclose(kernel(inp),ref(inp))
# Hash: REWARDS_HASH=sha256(rewards.py). grade.py asserts it at startup -> HALT.
# Build env.py/grader.py/scenario_helpers.py = verbatim verilog clones, swap
#   evaluate_task to map (op,M,N,dtype). NO from __future__ import annotations.
# [GATE-1 16:30]: `hud dev` serves ONE op locally (CPU settrace fallback ok);
#        two-yield works; grader returns a dict end-to-end. NO-GO=> fix plumbing.

# ---- BLOCK 2: HOURS 3-6 (16:30-19:30)  HARNESS + 4-LAYER ANTI-HACK -
# anti_hack.py + bench_core.py + subprocess_runner.py on Modal H100.
# LaunchCounter = monkeypatch triton JITFunction.__call__ counter.
# PR_frac from the SAME timing/profiler pass (no extra GPU pass).
# subprocess_runner: 60-90s timeout, fail-closed.
# Dockerfile.hud: COPY rewards.py /donotaccess/rewards.py; bake TRITON_CACHE_DIR;
#        warmup-compile reference shapes at build.
# [GATE-2 18:30 RED-TEAM, build-blocking]: scripts/redteam.py asserts reward~0 on
#        ALL FOUR: passthrough / never-launched / try-except-torch-fallback /
#        bf16-downcast. AND test_anti_hack: known-good kernel launches>0,
#        never-launched==0 (fails build if the triton hook silently broke).
#        NO-GO => the verifier is the deliverable; do not proceed to training
#        until all four read ~0. This is the actual product.

# ---- BLOCK 3: HOURS 6-8 (19:30-21:30)  SHAPE SAMPLER + SPLIT (MOAT)-
# splits.py: TRAIN grid M in {256,512,1024,2048}; TEST {384,768,1536}+random
#        on 128-stride off-grid points. _assert_split_disjoint() import-time.
# sampler.py: hashlib.sha256(canonical_str) seeding (cross-machine determinism).
# manifest.py: freeze -> versioned JSONL; loop consumes frozen, never re-samples.
# tests/test_sampler.py: compile EVERY rendered Model + get_inputs() on CPU.
# [GATE-3 21:30]: split disjoint asserted; all 5 op templates compile+run on CPU.

# ---- BLOCK 4: HOURS 8-11 (21:30-00:30)  GRPO WIRING + CALIBRATION ---
# grpo_loop.py: trl GRPOTrainer, LoRA, vLLM colocate, gpu_mem_util~0.5,
#        num_generations=8 (->4 if OOM), single-turn reward_funcs PRIMARY.
# train/reward.py wraps the SAME rewards.py path (byte-identical, hash-checked).
# 1h-BUDGET stretch: validate environment_factory multi-turn. RED by 23:30 =>
#        ship single-turn, no debate.
# [GATE-4a ~23:30]: trl runs 1 step end-to-end on real env, reward flows, no OOM.
#        OOM => drop num_generations to 4, then Qwen2.5-Coder-3B.

# ---- BLOCK 5: HOURS 11-13.5 (00:30-03:00) CALIBRATION GATE (2-STAGE)-
# calibrate.py on 5-10 tasks, base 7B, group=8:
#  STAGE A: hash preflight==pin; grade.py known-good>0 & known-bad==0;
#           compile>=5%, allclose>=2%, group reward variance>0.
#  STAGE B (anti compile-only-variance): >=1 rollout earns NONZERO SPEEDUP
#           reward on >=2 of the tasks.
# Escalation ladder if B fails: P_TARGET 1.5->1.2->1.1; then curriculum floor
#        to L1 elementwise (base-7B corr ~40%, trivial speedup). +0.1 compile /
#        +0.2 imports-triton bootstrap credit (anneals out) only if A is sparse.
# [GATE-5 03:00 THE KICK GATE]: A AND B pass. If still RED at 03:00 with all
#        ladder rungs spent => kick on L1-elementwise-only (degraded but real
#        signal) rather than miss the run.

# ---- BLOCK 6: HOURS 13.5-? (03:00-08:00) OVERNIGHT GRPO --------------
# Launch grpo_modal.py. Watch first 10 steps live, then sleep-poll.
# Callbacks: step-150 OR flat-curve(window) auto-abort -> checkpoint + stop.
# Realistic 30-120 steps (compile+bench bound even with warm cache).
# 08:00 = NOMINAL KICK DEADLINE in the brief; we kick at 03:00 to bank 5h of
#        GPU. If calibration slipped, 08:00 is the HARD latest kick (still ~4h
#        of training before 1PM). Budget guard: step-150 abort, colocate
#        trainer+grader on ONE H100 to protect the $250.
# [GATE-6 06:00 PRE-RECORD MANDATE]: regardless of run health, snapshot the
#        current reward curve + best base-vs-held-out delta to disk. The demo
#        NEVER times live and NEVER depends on a still-running job.

# ---- BLOCK 7: HOURS ?-? (08:00-11:00) HELD-OUT EVAL + DEMO ARTIFACT --
# Post-hoc eval pass: base ckpt vs trained ckpt, reward on TEST shapes
#        {384,768,1536} (the money number). Separate pass, not inline.
# CONTINGENT daVinci #2: if >=3 successful rollouts (allclose AND speedup>=1.5x)
#        exist, hand-distill 1-3 into skills_v0.jsonl, inject UNCONDITIONALLY
#        only for the slide hero kernel (~0.5h). Else empty == null scheme.

# ---- BLOCK 8: HOURS ?-? (11:00-13:00) SLIDES + BUFFER ---------------
# Money slide: "Train M in {256,512,1024,2048}; TEST M in {384,768,1536} NEVER
#        seen. Base reward on test: X%. Trained: Y%. Gap = generalization."
# One prior-art sentence: "folded in daVinci's profiling-ratio anti-hack gate
#        + a static skill prefix; full co-evolution deferred."
# 12:30-13:00 = pure buffer. DELTA DUE 13:00.

```

**Risks:** KICK SLIP: calibration Gate-5 is at 03:00 to bank 5h before the 08:00 brief deadline; if A+B fail through the full escalation ladder (P_TARGET 1.5->1.2->1.1 then L1-elementwise), kick on L1-only at latest 08:00 rather than miss the run. Pre-record mandate at 06:00 makes the demo independent of run health.; REWARD DIVERGENCE (severity 5): rewards.py edited at 2AM updates kernel_env.py but not the deployed image's grade.py. Mitigation is FIRST integration test not last: COPY rewards.py /donotaccess; grade.py asserts sha256==REWARDS_HASH -> HALT; calibration runs grade.py directly on known-good(>0)/known-bad(==0) to catch ImportError-swallowed-to-0. Any rewards.py edit => rebuild image (10-20min) + hash bump before resuming. NO pip package.; RED-TEAM FAILURE = product failure: if any of passthrough/never-launched/try-except/bf16 does NOT read ~0 at Gate-2 (18:30), the verifier (the actual deliverable) is broken; do not proceed to training until fixed. The triton LaunchCounter hook is version-fragile and can silently count 0 (false hard-cap) or always-pass (hole) — test_anti_hack fails the build if a known-good kernel registers 0 launches.; THROUGHPUT: compile+bench per rollout dominates wall-clock even with TRITON_CACHE warm; group=8 x 5 ops likely yields only 30-120 steps overnight, so step-150 abort may never fire — treat abort as flat-at-whatever-step-reached and pre-record by 06:00 regardless.; SCRATCHPAD/CACHE TIMING EXPLOIT (KernelBenchX: 72% of timed results cache-affected): fixed-input timing lets a kernel cache the correctness-pass result. Mitigation: FRESH random inputs every timed iter (cached value is wrong for new input -> post-timing allclose fails -> C=0) + L2 flush + speedup hard-capped at 20x so any slip is clamped not rewarded 100x.; OOM on one 80GB H100 (7B + LoRA + vLLM colocate + Triton-compiling rollouts): drop gpu_memory_utilization, then num_generations 8->4, then Qwen2.5-Coder-3B. Validated at Gate-4a (~23:30) so the fallback happens before, not during, the overnight run.; GRPO GRADIENT COLLAPSE / compile-only variance: Stage-A variance can be driven entirely by compile partial-credit, training syntactically-valid-but-wrong kernels. Stage-B (>=1 nonzero-SPEEDUP rollout on >=2 tasks) is the explicit defense; cap compile-only credit well below any allclose-pass reward so the dominant gradient is correctness-then-speedup.; DISJOINTNESS REGRESSION (single most load-bearing invariant): a careless TRAIN/TEST edit makes a test shape coincide with train, silently destroying the moat. _assert_split_disjoint() at import + tests + freeze(). Frame held-out result as DIRECTIONAL train-vs-test gap (powers-of-two are in pretraining), using 128-stride off-grid test points (384/768/1536).; MULTI-TURN environment_factory is younger/less stable than single-turn reward_funcs: hard 1h budget at Block 4; RED by 23:30 => ship single-turn (rock-solid), multi-turn becomes a stretch not a blocker.; MODAL $250 / HUD $200 budget: overnight GRPO + grading fan-out can exhaust credit. Colocate trainer+grader on ONE H100, cache per-(op,shape) eager baselines, step-150 abort, kill the run the moment the pre-recorded curve is banked at 06:00 if budget is tight.

### Protean — Risk Register (top 8: trigger + mitigation + fallback)  (~3h)
**Goal:** Enumerate the eight highest-severity risks to shipping a non-gameable, held-out-generalization HUD RL environment for multi-turn Triton kernel optimization (Qwen2.5-Coder-7B) in a 24h solo build. Each risk has a concrete numeric trigger, a load-bearing mitigation grounded in the verified verilog-template scaffold + daVinci paper, and an explicit fallback that preserves the Sun 1PM deliverable. Scaffold claims VERIFIED by reading the real files: grader.py loads grade.py via importlib.util.spec_from_file_location (reward-divergence path is real); fail-closed try/except -> reward 0.0 with hard_caps is real; combine() is deliberately bypassed for a hand-built EvaluationResult with a negative-weight hard_cap_penalty subscore (correctness multiplicative cap is real); functional_score_zero forces reward=0 (correctness-floor pattern is real); setpriv --reuid 1000 uid-wall + _chown_tree_to_agent + start_new_session SIGKILL timeout runner are real; env.py deliberately omits `from __future__ import annotations` and types validate_mode as `str | None` (the @env.template typed-param crash is real and documented in-file).

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/rewards.py` — R1 epicenter: single source of truth for compute_reward; COPY-baked to /donotaccess and hash-pinned. Any edit => rebuild image + bump REWARDS_HASH + re-run preflight.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/tasks/<op>/donotaccess/grade.py` — R1/R7: hidden verifier; top imports /donotaccess/rewards.py + asserts sha256==REWARDS_HASH (HALT on mismatch); mirrors verilog grade.py importlib load + fail-closed.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/anti_hack.py` — R4: LaunchCounter monkeypatch on triton.runtime + AST ban; test_anti_hack.py asserts known-good>0 launches AND never-launched==0 (build fails if hook breaks).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/bench_core.py` — R5/R6: do_bench median-of-100 + L2 flush + locked clocks + FRESH-random-input-per-iter timing protocol (defeats scratchpad cache); speedup hard-capped at 20x.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/calibrate.py` — R2/R8 GO/NO-GO gate: hash preflight + known-good/known-bad grade.py smoke + >=2 calibration tasks with >=1 nonzero-SPEEDUP rollout BEFORE 08:00 kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/splits.py` — R7 moat: frozen disjoint train/test shape split; _assert_split_disjoint() at import + in tests; sha256(canonical-string) RNG seeding (not builtin hash()).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/callbacks.py` — R2 step-150 abort + held-out RewardCurveLogger (eval every 25 steps); pre-records the curve by 06:00 regardless.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/scripts/redteam.py` — R3/R5 Saturday-night assertion suite: passthrough / never-launched / try-except-fallback / bf16-downcast / scratchpad-cache kernels MUST all score reward~0 before kick.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/Dockerfile.hud` — R1/R5: COPY rewards.py /donotaccess/rewards.py + warm TRITON_CACHE_DIR bake; flat COPY not pip (rebuild 10-20min).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/env.py` — R-baseline: two-yield @env.template; NO future-import; validate_mode typed str|None (avoids verified PydanticUserError -32000).

**APIs:** importlib.util.spec_from_file_location (grade.py load path); hud.graders.EvaluationResult / SubScore (hand-built, NOT combine()); hud.Environment + @env.template + @env.initialize/@env.shutdown (two-yield); hud.environment.Workspace.shell_argv (uid wall); torch.cuda.Event / triton.testing.do_bench (timing); torch.profiler (PR launch-gate fraction); subprocess.Popen(start_new_session=True) + os.killpg (fail-closed runner); trl GRPOTrainer(reward_funcs=...) single-turn primary; environment_factory multi-turn stretch

**daVinci integration:** Only TWO daVinci techniques touch the risk surface, both reward-side and on the critical path. (1) PR term ported NOT as daVinci's additive Eq1 R=C*(1+speedup+PR) but as a MULTIPLICATIVE BINARY GATE: pr_gate=1.0 iff (Triton GPU-time fraction > tau=0.5 AND JITFunction launch_count>0) else 0.0; reward = correct_floor * speedup_score * pr_gate. This is risk R3's core mitigation (additive PR cancels in single-op GRPO advantage to ~2e-16 and trains launch-count maximization; the gate closes the torch-passthrough-wrapped-in-@triton.jit exploit a bare launch counter misses). Read from the SAME single profiling pass used for timing (zero extra GPU passes). (2) Static skill prefix is a POST-TRAINING demo-only artifact (hour 16-18, contingent on >=3 successful rollouts), explicitly NOT on the overnight path, so it carries no schedule risk. All other daVinci machinery (LITE Selection Agent, LOO adapter, Summary Agent, SFT cold-start, hour-0 skills_v0.jsonl) is CUT and therefore appears only as the de-scoping mitigation under R8, not as live risk.

```python
RISK REGISTER — Protean (24h solo). Ordered by expected loss (severity x likelihood).
Schedule anchors: hack start Sat 12:30; GRPO kick Sun 08:00; base-vs-trained delta due Sun 13:00.
Load-bearing credits: HUD $200, Modal $250. Each risk: TRIGGER (numeric) -> MITIGATION -> FALLBACK.

R1 REWARD-FORMULA DIVERGENCE [SEV5, worst] — likelihood HIGH (multiple 2AM red-team edits expected)
  TRIGGER: sha256(/donotaccess/rewards.py baked in image) != rewards.py that kernel_env.py imports.
  VERIFIED: grader.py:_load_grade_module -> importlib.util.spec_from_file_location(grade_path); the
            hidden grader is a self-contained file loaded from a baked donotaccess path, so a live
            edit to rewards.py does NOT reach the deployed image -> ghost-signal training, no error.
  MITIGATE: Dockerfile COPY rewards.py -> /donotaccess/rewards.py; grade.py asserts sha256==REWARDS_HASH
            (HALT, never silent-0); hash preflight = first line of calibrate.py; known-good/known-bad
            grade.py smoke catches ImportError-swallowed-to-0.0. Edit => rebuild + bump hash + preflight.
  FALLBACK: after 02:00 FREEZE rewards.py; tune only trainer-side knobs that never touch the formula.

R2 GRPO THROUGHPUT / NEVER REACH STEP 150 [SEV4] — likelihood HIGH
  TRIGGER: amortized compile+bench/rollout > ~8-12s => < 150 steps by 06:00 (realistic 30-120).
  MITIGATE: warm TRITON_CACHE bake + per-op warmup; L1 curriculum start; profiler only on timed pass;
            bench_batch warm-container reuse; LoRA; gpu_memory_utilization tuned to fit 7B+vLLM+Triton/80GB.
  FALLBACK: step-150 abort := 'flat at whatever step reached'; pre-record held-out curve by 06:00;
            if OOM persists drop to Qwen2.5-Coder-3B or num_generations=4.

R3 PR-TERM GAMABILITY + ADVANTAGE CANCELLATION [SEV4] — likelihood MED-HIGH at single-op scope
  TRIGGER: additive PR ~1.0 constant => GRPO advantage delta ~2e-16 (no signal) AND trains launch-count max.
  MITIGATE: pr_gate = 1.0 iff (Triton GPU-time fraction > tau=0.5 AND launch_count>0) else 0.0;
            reward = correct_floor * speedup_score * pr_gate (multiplicative, not additive). One profiling
            pass shared with timing (0 extra GPU). Passthrough-in-jit => ~0 GPU-time fraction => gate 0.
  FALLBACK: profiler attribution flaky on H100 => degrade to launch-count-only gate (robust); profiler
            NaN/timeout => pr_gate=0 (fail-closed, mirrors verilog grader run()).

R4 LAUNCHCOUNTER HOOK FRAGILITY [SEV4] — likelihood MED (triton internals shift across versions)
  TRIGGER: triton version drift => hook counts 0 (false cap, all-zero groups) OR always-pass (silent hole).
  MITIGATE: pin triton in Dockerfile+uv.lock; test_anti_hack.py asserts known-good>0 AND never-launched==0
            -> BUILD FAILS on drift. FALLBACK: substitute torch.profiler GPU-time-fraction as launch evidence.

R5 ALLCLOSE+TIMING SCRATCHPAD-CACHE RACE [SEV3-4] — likelihood MED (KernelBenchX: 72% cache-affected)
  TRIGGER: cached-result kernel reports ~0.01ms => fake ~100x; L2 flush misses in-kernel global scratchpad.
  MITIGATE: FRESH random input per timed iter (new seed per call) + median; post-timing allclose on the
            SAME varying inputs (cache => wrong => C=0); speedup HARD-CAPPED 20x; redteam.py asserts ~0.
  FALLBACK: 3 rotated fixed input sets if per-iter randn too slow; 20x cap is the hard backstop.

R6 TIMING VARIANCE -> NOISY REWARD [SEV3] — likelihood MED on shared H100
  TRIGGER: speedup jitter >~10-15% across identical calls => unstable advantages.
  MITIGATE: lock clocks @modal.enter; median-of-100; fixed warmup; L2 flush; SPEEDUP_FLOOR=1.1 dead-band;
            re-check allclose post-timing. FALLBACK: never time live in demo (pre-record); raise floor to 1.2.

R7 GENERALIZATION-MOAT LEAK / SPLIT REGRESSION [SEV4, the differentiator] — likelihood MED
  TRIGGER: a test shape == a train shape after an edit, OR PYTHONHASHSEED divergence across machines.
  MITIGATE: _assert_split_disjoint() at import + tests + freeze(); sha256(canonical) seeding not hash();
            consume FROZEN manifest in loop; off-grid test shapes 384/768/1152. FALLBACK: present directional
            train-vs-test GAP + continuous-interval disjointness from KernelBench discrete values.

R8 CALIBRATION-GATE FAILURE / GRADIENT COLLAPSE [SEV3-4] — likelihood MED-HIGH at 7B
  TRIGGER: ~0% allclose flat groups, OR variance from compile-credit only (learns valid-but-wrong).
  MITIGATE: gate1 (>=5% compile, >=2% allclose, nonzero variance) + gate2 (>=1 NONZERO-SPEEDUP rollout in
            >=2 of 5-10 tasks); correctness is a multiplicative FLOOR (mirrors functional_score_zero cap);
            compile credit (+0.1) << allclose reward; ladder P_TARGET 1.5->1.2->1.1 then L1 elementwise.
  FALLBACK: cannot pass by 07:30 => short SFT cold-start from 50-100 reserve kernels, OR ship directional
            curve + lead with the verifier as the contribution. CUT (de-scope) Selection/LOO/Summary/SFT/
            hour-0 skills so none can steal the kick window.

NOTE — baseline build risk (not top-8, pre-mitigated): @env.template typed-param crash. env.py MUST omit
  `from __future__ import annotations` and type validate_mode as plain str|None (verified: documented in the
  scaffold env.py; adding the future-import => PydanticUserError surfaced as -32000 at deploy). Pin the SDK commit.
```

**Risks:** R1 REWARD-FORMULA DIVERGENCE [SEV 5, worst]. TRIGGER: any rewards.py constant edit (P_TARGET/tol/tau/SPEEDUP_FLOOR) after the image is built but before re-deploy => sha256(/donotaccess/rewards.py) != the rewards.py kernel_env.py imports; model trains on a ghost signal, zero/inverted held-out delta at 13:00 with NO error (verified: grader.py:_load_grade_module uses importlib.util.spec_from_file_location from a baked donotaccess path, so a live edit does not reach the deployed image). MITIGATION: (a) Dockerfile `COPY kernelforge/rewards.py /donotaccess/rewards.py`; (b) grade.py top: `assert hashlib.sha256(open('/donotaccess/rewards.py','rb').read()).hexdigest()==REWARDS_HASH` -> HALT not silent-zero; (c) hash preflight is the FIRST line of calibrate.py before the 08:00 kick; (d) smoke-run grade.py on one known-good (assert reward>0) + one known-bad (assert reward==0) to catch ImportError-swallowed-to-0.0. Rule: edit => rebuild + bump hash + re-preflight (10-20min cost, budgeted). FALLBACK: if rebuild cadence is killing throughput after 02:00, FREEZE rewards.py (no further constant tweaks) and tune only trainer-side knobs (LR, num_generations, curriculum) that do not touch the graded formula.; R2 GRPO THROUGHPUT / NEVER-REACH-150 [SEV 4]. TRIGGER: realistic overnight budget is ~30-120 GRPO steps (each rollout = JIT compile 30-120s cold + allclose + median-of-100 timed bench + profiler pass); if compile+bench per rollout exceeds ~8-12s amortized, group=8 x 5 ops x multi-turn does not reach step 150 by 06:00. MITIGATION: warm TRITON_CACHE_DIR baked at image build + per-op reference-shape cache warmup in @modal.enter(); easy-op (L1 elementwise) curriculum start; profiler gated to the timed pass only; bench_batch() reuses one warm container where safe; LoRA not full-FT; gpu_memory_utilization tuned down to fit 7B+vLLM+Triton on one 80GB H100. FALLBACK: treat step-150 abort as 'flat-at-whatever-step-we-reach'; RewardCurveLogger pre-records the held-out curve by 06:00 REGARDLESS of step count; if OOM/thrash persists, drop to Qwen2.5-Coder-3B or num_generations=4.; R3 PR-TERM GAMABILITY + ADVANTAGE CANCELLATION [SEV 4]. TRIGGER: daVinci's additive PR=T_gen/T_total collapses to ~1.0 at single-op scope (a near-constant added to every group member leaves GRPO advantages unchanged to ~2e-16 => zero training signal) AND additive PR trains launch-count maximization (torch.matmul wrapped in a trivial @triton.jit fires a kernel so a bare launch counter passes). MITIGATION: CUT additive PR; implement MULTIPLICATIVE BINARY GATE pr_gate=1.0 iff (Triton GPU-time fraction>tau=0.5 AND launch_count>0) else 0.0; reward=correct_floor*speedup_score*pr_gate. A passthrough kernel has near-zero Triton GPU-time fraction => fails the gate => reward 0. Read from the same single profiling pass (no extra GPU cost). FALLBACK: if torch.profiler GPU-ns attribution proves flaky/version-fragile on H100 by Sat evening, degrade pr_gate to the binary launch-count gate alone (robust) and present the GPU-time-fraction gate as a stretch on the slide; profiler NaN/timeout => pr_gate=0 (fail-closed, mirrors verilog grader run()).; R4 LAUNCHCOUNTER HOOK FRAGILITY [SEV 4]. TRIGGER: the monkeypatch targets a specific triton.runtime.driver internal; a triton version mismatch silently makes it count 0 (false hard-cap => ALL kernels score 0 => GRPO sees flat zero groups) OR always-pass (silent anti-hack hole). MITIGATION: pin the exact triton version in Dockerfile + uv.lock; test_anti_hack.py asserts a KNOWN-GOOD Triton kernel registers >0 launches AND a never-launched (pure-torch) kernel registers 0 — the BUILD FAILS if either assertion breaks, so a version drift cannot ship silently. FALLBACK: if the internal hook is unstable, replace with the robust torch.profiler GPU-time-fraction signal as the launch evidence (the same pass R3 already runs); the two anti-hack signals are interchangeable for the gate.; R5 ALLCLOSE+TIMING SCRATCHPAD-CACHE RACE [SEV 3-4]. TRIGGER: a kernel seeds a global-memory scratchpad on the correctness pass and returns the cached result on the timed pass (~0.01ms => fake 100x); L2 flush clears L2 but NOT an in-kernel global scratchpad (KernelBenchX: 72% of timed results were cache-affected). MITIGATION: FRESH random inputs for EVERY timed iteration (`for i in range(M): inp=torch.randn(shape,device='cuda',generator=Generator('cuda').manual_seed(BASE+i)); t_i=cuda_event_time(kernel,inp)`), median of t_i — a cached value is wrong for new inputs so post-timing allclose on the SAME varying inputs fails => C(y)=0; keep L2 flush AND fresh inputs (orthogonal); speedup HARD-CAPPED at 20x so any residual exploit is clamped not rewarded. redteam.py asserts a scratchpad-cache kernel scores ~0 before the kick. FALLBACK: if fresh-input-per-iter timing is too noisy/slow, use 3 fixed disjoint input sets rotated across iterations (still defeats single-cache) + keep the 20x cap as the hard backstop.; R6 TIMING VARIANCE -> NOISY REWARD -> GRPO INSTABILITY [SEV 3]. TRIGGER: do_bench median on a shared/contended H100 makes speedup non-deterministic across calls (>~10-15% jitter), so the same kernel gets different rewards => unstable advantages. MITIGATION: lock GPU clocks in @modal.enter(); median-of-100; fixed warmup iters; L2 flush; SPEEDUP_FLOOR=1.1 dead-band (no speedup credit below 1.1x absorbs noise); re-check allclose post-timing. FALLBACK: for the DEMO never time live — pre-record per-op held-out speedups (audit mandate); if jitter still destabilizes training, raise SPEEDUP_FLOOR to 1.2 and widen the dead-band so only robust speedups earn reward.; R7 GENERALIZATION-MOAT LEAK / SHAPE-SPLIT REGRESSION [SEV 4, the differentiator]. TRIGGER: a careless edit makes a test shape coincide with a train shape (train M in {256,512,1024,2048}, test M in {384,768,1536}) OR PYTHONHASHSEED salting diverges between control plane and Modal workers so train/test sets differ across machines — silently destroying the 'reasoning over memorization' claim that IS the project. MITIGATION: _assert_split_disjoint() runs at import time AND in tests AND inside freeze() before writing the manifest; seed RNG with hashlib.sha256 of a canonical string (NOT builtin hash()); always consume the FROZEN manifest JSONL in the loop rather than re-sampling; test shapes chosen off the 128-stride grid (384/768/1152) which are unusual in pretraining corpora. FALLBACK: if disjointness is ever in doubt, frame the result as a DIRECTIONAL train-vs-test reward GAP (not an absolute generalization proof) and lead the demo with continuous-interval disjointness from KernelBench's discrete values (provable, roadmap #74 unshipped).; R8 CALIBRATION-GATE FAILURE / GRPO GRADIENT COLLAPSE [SEV 3-4]. TRIGGER: base 7B produces ~0% allclose => flat zero groups (zero gradient), OR variance is driven entirely by compile partial-credit so the model learns syntactically-valid-but-wrong kernels (speedup gradient too weak in the first 50 steps to generalize). MITIGATION: TWO gates before the 08:00 kick — (1) primary: >=5% compile, >=2% allclose, nonzero group variance; (2) SECONDARY: >=1 rollout earning NONZERO SPEEDUP reward (not just compile/allclose credit) in >=2 of 5-10 calibration tasks. Correctness is a MULTIPLICATIVE FLOOR (correct=0 => reward=0, mirrors verilog functional_score_zero hard cap) so speedup can never be traded for correctness; cap compile-only credit (+0.1) well below any allclose-pass reward. Escalation ladder on failure: P_TARGET 1.5->1.2->1.1, then drop curriculum to L1 elementwise (base-7B correctness ~40%, trivial parallelization speedup). FALLBACK: if the speedup-signal gate cannot pass by ~07:30 even at L1, do NOT burn 8h on a dead run — collect the 50-100 SFT warmup kernels held in reserve for a short cold-start, OR ship the base-vs-base-with-partial-credit directional curve and present the verifier (the actual deliverable) as the contribution. CUT-as-mitigation: LITE Selection Agent, LOO adapter, Summary Agent, SFT cold-start, and hour-0 skills_v0.jsonl are all de-scoped so none can consume the calibration/kick window.

### Protean Demo Plan: money slide (train-shape vs held-out-shape reward curve), live hero kernel, pre-recorded fallbacks, prior-art slide  (~3h)
**Goal:** A 4-minute judge-facing demo that lands ONE differentiated claim — "a held-out continuous-shape generalization verifier proves the trained 7B reasons over Triton kernels rather than memorizing shapes" — with numeric evidence (base-vs-trained reward gap on shapes never seen in training), one live hero-kernel stopwatch moment, and a prior-art slide that converts the daVinci-kernel citation (published ~1 week pre-hackathon) from a "you just reproduced a paper" liability into a "current-with-literature" strength. Every live element has a pre-recorded fallback so nothing in the demo can fail on stage. The deliverable is a slide deck + a /Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/ asset dir, NOT new training code; it consumes artifacts the overnight GRPO run and the held-out eval pass already emit.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/curve_train_vs_heldout.png` — MONEY SLIDE plot. Two reward curves over GRPO steps: train-shape reward (M in {256,512,1024,2048}) and held-out-shape reward (M in {384,768,1536}). Rendered by plot_curve.py from rewards.jsonl logged every step + held-out eval every 25 steps. Title text baked in: 'Held-out shapes never seen in training.'
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/plot_curve.py` — matplotlib script: reads train_rewards.jsonl + heldout_eval.jsonl (emitted by train/callbacks.py RewardCurveLogger), plots two lines + base-model horizontal dashed baselines, annotates final base-vs-trained delta on held-out shapes. Run by hour 6AM regardless of step count (audit mandate: pre-record curve by 6AM).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/heldout_eval.jsonl` — Numbers behind the money slide: per-(step, M-shape, base|trained) records {step,M,model,mean_reward,pct_correct,mean_speedup,n_rollouts}. The single source of the X% base / Y% trained numbers quoted on stage. Produced by a post-hoc eval pass (train/calibrate.py --eval-heldout) so the gap is honest even if training reached only ~30-120 steps.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/hero_kernel.py` — The single best trained-model Triton kernel (allclose-passing, highest measured speedup on a HELD-OUT shape). Frozen artifact for the live hero moment + screenshot fallback. Selected by demo/pick_hero.py from the overnight rollout dump.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/pick_hero.py` — Scans rollout dump for the trained checkpoint, filters allclose==True AND launch_count>0 AND pr_gate==1 AND M in HELDOUT_SHAPES, sorts by measured speedup, writes hero_kernel.py + hero_meta.json (op, shape, speedup, eager_ms, triton_ms). Guarantees the hero is a generalization win, not a train-shape win.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/hero_run.sh` — Live hero stopwatch: invokes the EXISTING harness bench path (modal run harness/modal_app.py::bench --op <op> --M 768 --kernel hero_kernel.py) on a HELD-OUT shape and prints 'PyTorch eager: A ms | Trained Triton: B ms | speedup Bx | allclose PASS | reward R'. Same code path as the grader so the number is the real reward, not a marketing number.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/hero_run.mov` — PRE-RECORDED screen capture of hero_run.sh succeeding (recorded hour ~7-8AM). Primary fallback if live Modal H100 is unavailable/slow on stage. Plays in <20s.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/redteam_proof.txt` — Captured stdout of scripts/redteam.py asserting reward~0 on all four hacks (passthrough / never-launched / try-except-fallback / bf16-downcast). The 'verifier is the moat' evidence slide. This is the env's actual deliverable per the plan; screenshot it.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/SLIDES.md` — 6-slide deck outline (markdown -> any renderer): (1) one-line hook, (2) MONEY slide w/ curve, (3) the verifier/anti-hack moat w/ redteam_proof, (4) live hero kernel, (5) prior-art slide (daVinci one sentence), (6) HUD-env reusability + ask. Speaker notes + the prepped CTO-question answers inline.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/demo/fallback_curve.png` — Backup of curve_train_vs_heldout.png copied to a second location + committed to git at 6AM, so a laptop/Modal failure on stage cannot lose the money slide. Audit mandate: pre-recorded curve is the fallback for step-150 abort / flat-curve.

**APIs:** plot_curve.py: render_money_slide(train_jsonl, heldout_jsonl, base_baselines, out_png) -> writes curve_train_vs_heldout.png; pick_hero.py: pick_hero(rollout_dump, heldout_shapes={384,768,1536}) -> (hero_kernel.py, hero_meta.json) filtered on allclose & launch>0 & pr_gate==1; hero_run.sh -> modal run harness/modal_app.py::bench --op <op> --M 768 --kernel demo/hero_kernel.py (SAME grader code path, real reward); train/calibrate.py --eval-heldout: post-hoc base-vs-trained eval on HELDOUT_M -> heldout_eval.jsonl (the X/Y numbers); train/callbacks.py RewardCurveLogger: per-step train_rewards.jsonl + every-25-step heldout_eval (the curve data source); scripts/redteam.py: asserts reward~0 on {passthrough,never_launched,try_except_fallback,bf16_downcast}; stdout -> redteam_proof.txt; splits._assert_split_disjoint(): the disjoint-by-construction invariant quoted verbatim on the money slide as the generalization guarantee

**daVinci integration:** Demo references exactly the two daVinci techniques the final scope locked (no more): (1) the profiling-ratio term, presented HONESTLY as a multiplicative binary PR-GATE (tau=0.5) not daVinci's additive Eq1 PR — Slide 3 frames it as the extra gate that kills the torch-passthrough-wrapped-in-@triton.jit exploit a bare launch counter misses; (2) static skill-injection prefix at inference, mentioned on Slide 5 as a deferred-co-evolution artifact (only shown live if >=3 successful overnight rollouts let us hand-distill it for the hero kernel; otherwise dropped silently — null scheme). The prior-art slide cites daVinci-kernel (arXiv 2606.16497) in ONE sentence as currency-with-literature and explicitly states full skill co-evolution / 3-agent RL (Eq7-9) is OUT of scope. NO co-evolution claim is made anywhere. The money-slide differentiator (held-out continuous-shape verifier) is positioned as the thing daVinci does NOT have, turning the citation into a strength.

```python
# ============================================================
# DEMO RUN-OF-SHOW  (4 min, 6 slides) — every live beat has a recorded fallback
# ============================================================
#
# SLIDE 1 — HOOK (15s)
#   "KernelBench roadmap item #74 — a generalization verifier for GPU kernels — is
#    unshipped. We built it. A 7B learns Triton optimization and we PROVE it generalizes
#    to tensor shapes it never trained on."
#   (Sets the single differentiated claim before any architecture talk.)
#
# SLIDE 2 — MONEY SLIDE (60s)  [curve_train_vs_heldout.png]
#   Plot: x=GRPO step (0..N, N~30-120 actual), y=mean reward in [0,2].
#     line A = TRAIN shapes  M in {256,512,1024,2048}
#     line B = HELD-OUT shapes M in {384,768,1536}  (continuous off-grid, stride-128 off points)
#     dashed = base-7B baselines for each set.
#   Spoken numbers (read from heldout_eval.jsonl final row):
#     "Base 7B on held-out shapes: mean reward X (Z% allclose).
#      Trained: Y (W% allclose). The held-out gap Y-X is generalization, not memorization."
#   KEY LINE: "Train and test shape ranges are CONTINUOUS and DISJOINT BY CONSTRUCTION —
#              a test shape can never equal a train shape."  (_assert_split_disjoint invariant)
#   Honesty caveat spoken once: "deltas are directional on a single overnight run."
#
# SLIDE 3 — THE MOAT IS THE VERIFIER (45s)  [redteam_proof.txt screenshot]
#   "Reward = correctness(allclose, fresh random inputs) x speedup x PR-gate. Four programmatic
#    gates. Here are four standard reward hacks — all score ~0:"
#     passthrough .......... reward 0.000  (AST ban)
#     never-launched ....... reward 0.000  (launch-count gate)
#     try/except fallback .. reward 0.000  (launch-count + allclose)
#     bf16 downcast ........ reward 0.000  (dtype/shape + 1.1x speedup floor)
#   "PR-gate (daVinci's profiling ratio) makes a torch-passthrough wrapped in @triton.jit
#    fail too — the Triton kernel must own >50% of measured GPU time."
#
# SLIDE 4 — LIVE HERO KERNEL (45s)  [hero_run.sh  ->  fallback hero_run.mov]
#   Run on a HELD-OUT shape (M=768), same harness path as the grader:
#     $ bash demo/hero_run.sh
#     op=<op> shape M=768 (HELD-OUT)
#     PyTorch eager:  A.AA ms
#     Trained Triton: B.BB ms   speedup  C.Cx   allclose PASS   reward R.RR
#   "This kernel was generated by the trained model for a shape it never saw."
#   FALLBACK RULE: if Modal cold-start >20s or any nonzero exit -> cut to hero_run.mov.
#                  NEVER time live twice; one shot then video. (audit: never time live in demo)
#
# SLIDE 5 — PRIOR ART (30s, converts liability->strength)
#   ONE sentence: "Published one week ago, daVinci-kernel does multi-agent skill RL on a 14B;
#    we folded in exactly two of its ideas — the profiling-ratio anti-hack GATE and a static
#    skill-injection prefix at inference — and explicitly DEFERRED full skill co-evolution.
#    Our contribution is the held-out continuous-shape verifier, which daVinci does not have."
#   Prepped CTO answers (speaker notes, do not put on slide):
#     Q "Did you just reproduce daVinci at 7B?"  ->  "No. daVinci optimizes on KernelBench's
#       discrete shapes. We test on continuous off-grid shapes provably disjoint from any
#       published eval; the verifier is the new artifact."
#     Q "How do you know test shapes are unseen?" -> "Disjoint-by-construction: train is a fixed
#       discrete grid, test is the complementary continuous band; _assert_split_disjoint runs at
#       import + in CI. Off-grid points like 384/768/1152 are unusual in code corpora."
#     Q "Is the gap real or noise?" -> "Directional, single run; locked clocks, median-of-100,
#       fresh inputs per timed iter, post-timing allclose recheck. We show the curve, not a point."
#
# SLIDE 6 — REUSABLE HUD ENV + ASK (20s)
#   "It's a standard HUD two-yield env on the verilog-template scaffold: one parameterized
#    template mints unbounded train/held-out tasks from 5 ops x a shape sampler. Anyone can
#    fork it and train their own kernel model. That's the reusable RL environment."
#
# ============================================================
# ASSET BUILD ORDER (when, gated on overnight run)
# ============================================================
# 02:00-06:00  training runs; callbacks.py logs train_rewards.jsonl + heldout_eval every 25 steps
# 06:00  HARD: run plot_curve.py -> curve_train_vs_heldout.png  (even if only ~30 steps; pre-record now)
# 06:00  cp curve_train_vs_heldout.png fallback_curve.png ; git add+commit both
# 06:30  train/calibrate.py --eval-heldout  -> heldout_eval.jsonl (clean base-vs-trained numbers)
# 07:00  pick_hero.py -> hero_kernel.py + hero_meta.json  (best HELD-OUT allclose+speedup rollout)
# 07:15  screen-record hero_run.sh succeeding -> hero_run.mov   (the live-demo insurance)
# 07:30  scripts/redteam.py | tee redteam_proof.txt   (must already pass; this just captures it)
# 08:00  SLIDES.md filled with real X/Y/Z numbers; rehearse once end-to-end with fallbacks
# CONTINGENCY: 0 successful held-out rollouts by 07:00 -> hero slide becomes "best TRAIN-shape
#   kernel + honest 'held-out still converging'"; money slide still shows the reward-gap trend.
#   Flat curve / step-150 abort -> use fallback_curve.png + frame as "directional, run-limited".
```

**Risks:** MONEY-SLIDE GAP IS WEAK OR INVERTED: only ~30-120 steps overnight may yield a held-out gap within noise. MITIGATION: build curve at 6AM regardless (audit mandate), frame as DIRECTIONAL single-run trend, show the curve not a point estimate; the disjoint-by-construction split is the claim even if the gap is modest.; ZERO SUCCESSFUL HELD-OUT ROLLOUTS by 7AM -> no live hero kernel. MITIGATION: hero slide degrades to best TRAIN-shape kernel + 'held-out converging' honesty; money-slide trend still carries the talk. pick_hero.py must not crash on empty filter (return None -> trigger contingency branch).; LIVE HERO TIMING FAILS ON STAGE (Modal cold-start >20s, contention, nonzero exit). MITIGATION: hero_run.mov pre-recorded at 7:15AM is the primary fallback; rule is one live attempt then cut to video, NEVER time live twice (audit: never time live in demo).; NUMBER DIVERGENCE between spoken X/Y and the plot: if heldout_eval.jsonl (post-hoc) and the in-loop curve use different reward constants, judges catch inconsistency. MITIGATION: both must read the SAME hash-pinned rewards.py (REWARDS_HASH preflight from the reward component); quote numbers only from heldout_eval.jsonl, label the curve as in-loop trend.; JUDGE FRAMES IT AS 'daVinci REPRODUCTION'. MITIGATION: Slide 5 one-sentence prior-art + prepped CTO answers (continuous-disjoint shapes vs KernelBench discrete; verifier is the new artifact) steer to the moat; this is pure framing, 0h, but must be rehearsed.; OVER-CLAIMING ANTI-HACK: redteam_proof.txt shows only 4 hacks; a judge names a 5th. MITIGATION: state the verifier is fail-closed (unknown failure -> reward 0) and the 4 are the known KernelBenchX exploit classes incl. the scratchpad-cache race (fresh inputs per timed iter); invite the judge to propose one as future red-teaming.

### Budget &amp; Infrastructure: Modal H100 economics, HUD credits, what-runs-where, and throughput math for Protean  (~2.5h)
**Goal:** Prove the entire 24h solo build fits inside Modal $250 + HUD $200 with margin, by (a) fixing what physically runs where, (b) doing the GPU-hour arithmetic at the verified Modal H100 rate ($3.95/hr = $0.001097/sec, confirmed from modal.com/pricing 2026-06-21), and (c) computing realistic per-rollout and overnight GRPO throughput given TRITON_CACHE_DIR warm-bake + do_bench timing cost, so the Sun-8AM kick and Sun-1PM held-out delta are reachable without exhausting credit. Output is a numeric budget table, a colocation decision, and a throughput model with the step-150/cost abort tied to a dollar ceiling.

**Files:**
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/grpo_modal.py` — The ONE @app.function(gpu='H100', timeout=86400) that colocates vLLM rollout + trl GRPOTrainer + in-process grader. Lifted verbatim from ml-template/modal_runner.py:88-93 (gpu='H100', timeout=86400, secrets). Single long-lived container = single billed GPU-hour stream; do NOT use .spawn() fan-out for training (that multiplies GPU cost N-fold).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/harness/modal_app.py` — Separate @app.function(gpu='H100') for the grader/bench oracle used by check_calibration + redteam ONLY (short-lived, seconds-to-minutes). During overnight GRPO the grader runs IN-PROCESS inside grpo_modal.py (same container, same GPU) — never as a second billed H100. This file's standalone H100 path exists for dev/calibration smoke tests where the trainer isn't up.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/Dockerfile.hud` — Bakes TRITON_CACHE_DIR=/triton-cache + warmup-compiles every reference op at image BUILD time (build minutes are free; cold compile at runtime costs billed GPU-seconds). Pins triton/torch/trl/vllm versions. Layout mirrors verilog Dockerfile.hud (ENV WORKSPACE_ROOT/HIDDEN_ROOT, root:700 donotaccess, COPY rewards.py /donotaccess/rewards.py for the hash-pinned shared reward).
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/configs.py` — Holds the COST KNOBS as named constants so the dollar ceiling is enforceable: GPU_HOUR_USD=3.95, BUDGET_CEILING_USD=200 (of 250, 50 reserved), OVERNIGHT_MAX_HOURS=10, STEP_ABORT=150, plus do_bench knobs (BENCH_ITERS=100, WARMUP=25, TIMEOUT_S=90). A wall-clock cost callback aborts when elapsed*3.95 > ceiling.
- `/Users/nihalnihalani/Desktop/Github/hud/hudupdated/kernelforge/train/callbacks.py` — CostAbortCallback (new): on every trl on_step_end, compute elapsed_h = (now-start)/3600; if elapsed_h*GPU_HOUR_USD > BUDGET_CEILING_USD - spent_so_far OR step==STEP_ABORT with flat reward, checkpoint + stop. This is the dollar-level enforcement of the audit's step-150 abort.

**APIs:** modal.App.function(gpu='H100', timeout=86400, secrets=[...]) — ml-template/modal_runner.py:88; modal.Image.from_dockerfile('Dockerfile.hud', build_args={...}) — ml-template/modal_runner.py:34; Function.spawn() — fan-out for PARALLEL CALIBRATION ONLY, never for training (cost multiplier); triton.testing.do_bench(fn, warmup=25, rep=100, return_mode='median') with L2 flush; modal.enter() to lock GPU clocks (nvidia-smi -lgc) once per warm container; ENV TRITON_CACHE_DIR=/triton-cache baked at image build; trl GRPOTrainer + vLLM colocate (gpu_memory_utilization tuned ~0.45 to share 80GB with LoRA policy + Triton compile); HUD: hud.eval / taskset sync / trace ingestion (control-plane, CPU-side, NOT GPU-billed)

**daVinci integration:** daVinci is cost-neutral as scoped. The two KEPT daVinci items add ~zero GPU spend: (1) PR-as-binary-launch-gate is computed from the SAME single profiled pass already used for do_bench timing — no extra GPU pass, no extra dollars (the additive continuous PR was cut partly because its torch.profiler attribution added 10-30% per-bench overhead, i.e. real GPU cost; the binary gate folded into the timing pass removes that). (2) The post-training static skill prefix is a ~0.5h CPU-side hand-distillation at hour 16-18, off the GPU clock entirely. The daVinci items that WOULD have cost GPU-hours — SFT cold-start on drkernel-8k (~2h H100 ~$8), online Summary-Agent skill verification (re-runs the policy per candidate = extra rollouts = extra $), LLM-rerank Selection — are all CUT, which is why the budget lands at ~$33 not ~$60. The one daVinci-adjacent cost lever retained: profiling-aware sampling intuition (PR gate) keeps reward dense so fewer wasted all-zero groups, marginally improving GPU-hour efficiency per useful gradient.

```python
# ===================================================================
# 1. THE TWO BUDGETS — what each credit pool actually pays for
# ===================================================================
# Modal $250  = the ONLY GPU money. H100 @ $3.95/hr ($0.001097/sec, verified
#               modal.com/pricing 2026-06-21). 250 / 3.95 = 63.3 H100-hours TOTAL.
#               CPU/mem add-on is rounding error: 0.125 core min + a few GiB
#               ~ $0.05/hr next to $3.95 GPU; ignore in planning.
# HUD   $200  = control plane: taskset sync, trace/rollout ingestion, job grouping
#               (hud.eval, _send_job_enter in ml-template/modal_runner.py:300).
#               NO GPU on HUD's side for this build -> $200 is effectively
#               uncapped headroom for traces/evals. Not load-bearing on cost.
#
# RULE: plan to the GPU ceiling. Reserve $50 of the $250 as panic buffer ->
#       BUDGET_CEILING_USD = 200 -> 200/3.95 = 50.6 spendable H100-hours.

# ===================================================================
# 2. GPU-HOUR ALLOCATION across the 24h window (12:30 Sat -> 13:00 Sun)
# ===================================================================
#   PHASE                        WALL-CLOCK   GPU-ON?   H100-h   $cost
#   --------------------------   ----------   -------   ------   ------
#   A image build + cache bake   13:00-..     build min  0.0     $0  (build is free)
#   B single-op end-to-end smoke ~1h on/off   bursty     1.0     $3.95
#   C red-team 4 hack kernels    ~1h bursty    bursty     0.7     $2.77
#   D calibration gate (5-10     ~1h bursty    bursty     1.0     $3.95
#     tasks x group, grade only)
#   E OVERNIGHT GRPO (the kick)  08:00->12:00  CONTINUOUS 4-5     $15.8-19.8
#   F held-out eval (base vs     12:00-12:45   ~0.75      0.75    $2.96
#     trained, both checkpoints)
#   --------------------------   ----------   -------   ------   ------
#   TOTAL committed                                      ~8.5h    ~$33
#   Spendable ceiling                                    50.6h    $200
#   => ~6x headroom. Cost is NOT the binding constraint; THROUGHPUT is.
#
# Implication: you can afford to BURN GPU to de-risk — e.g. keep one warm
# container alive during dev (idle H100 ~$3.95/hr) for fast iteration, and
# still finish at ~$33-40. Only a runaway loop (forgot to stop a container)
# threatens $250; the CostAbortCallback + Modal timeout cap that.

# ===================================================================
# 3. COLOCATION DECISION (single H100, single billed stream)
# ===================================================================
# trl(GRPOTrainer) + vLLM(rollout gen) + grader(compile+bench) ALL in ONE
# @app.function(gpu='H100') container. 80GB budget:
#   - Qwen2.5-Coder-7B policy in LoRA + grads/optim   ~18-22 GB
#   - vLLM colocate inference (gpu_memory_utilization~0.45) ~30-36 GB
#   - Triton JIT compile + do_bench eager baseline + agent kernel ~6-10 GB
#   - headroom                                          ~12-18 GB
# Grader runs IN-PROCESS (not a 2nd H100) — an in-proc illegal-mem-access in a
# bad kernel can kill the rollout, so the bench step uses a subprocess WITHIN
# the same container (CPU-spawn cost, GPU shared) per the harness fail-closed
# pattern (verilog grade.py:85 run() SIGKILLs the process group on timeout).
# COST OF GETTING THIS WRONG: .spawn()-ing the grader as its own gpu='H100'
# function = 2 billed H100s = 2x cost AND cross-container tensor shipping.
# Don't. Spawn fan-out is ONLY for phase D calibration (short, parallel, cheap).

# ===================================================================
# 4. THROUGHPUT MATH — the actual binding constraint
# ===================================================================
# Per-rollout cost = generation + grade(compile + allclose + do_bench + PR pass)
#
#   t_gen   (vLLM, ~512-1024 completion tok, 7B)        ~3-6 s
#   t_compile  WARM (TRITON_CACHE hit, same src)        ~0.2-2 s
#              COLD (novel agent kernel, first-seen)    ~30-120 s   <-- the killer
#   t_allclose (2 seeds, fresh random inputs)           ~0.2 s
#   t_dobench  (warmup=25 + rep=100, median, L2 flush)  ~1-3 s
#   t_PR pass  (single profiled pass, reused for timing) ~0.2 s (folded in)
#   subprocess spawn (python+CUDA init, fail-closed)    ~1-2 s
#   ---------------------------------------------------------------
#   WARM rollout  ~ 6-13 s     COLD rollout ~ 35-130 s
#
# WHY COLD DOMINATES: TRITON_CACHE only helps RE-compiling identical source.
# Every NOVEL agent kernel is a cold compile. Mitigations baked into the plan:
#   - warm-bake reference shapes at IMAGE BUILD (phase A, free) so EAGER
#     baselines + reference launches never cold-compile at runtime.
#   - timeout_s=90 caps a pathological compile -> scores 0, rollout not lost.
#   - easy-op (L1 elementwise) curriculum start: small kernels compile fast.
#
# OVERNIGHT BUDGET (phase E, 08:00->12:00 = 4h = 14400 s wall):
#   GRPO step = num_generations(G) rollouts graded + 1 optim step.
#   Take G=8, single-turn primary path (multi-turn is a stretch).
#   Assume mix: 70% warm-ish (~12s) + 30% cold-capped (~60s avg w/ timeout)
#     => mean rollout ~ 0.7*12 + 0.3*60 = 8.4 + 18 = 26.4 s
#   step grade wall (rollouts can pipeline gen||grade, but be conservative
#     serial) ~ G*26.4 = 8*26.4 = 211 s/step grading
#     + optim/backward (LoRA, accum)                ~ 15-30 s/step
#     => ~230-240 s/step  -> ROUND to ~4 min/step.
#   14400 s / 240 s = ~60 steps overnight.  (range 40-100 given variance)
#
# => REALISTIC overnight = 40-100 GRPO steps, NOT 150. The step-150 abort
#    likely never fires on step count; it fires as the audit says: "flat-at-
#    whatever-step-we-reach". Pre-record the reward curve by 06:00 regardless.
#
# THROUGHPUT LEVERS (in priority order, all already in sibling components):
#   1. curriculum start on fast-compiling L1 ops (halves cold tail)
#   2. cache per-(op,shape) EAGER baseline ONCE (don't re-time torch each step)
#   3. do_bench rep=100 ONLY on the timed pass; allclose pass uses rep=0
#   4. G=8 -> G=4 if behind (trl 2-GRPO shows small groups still learn) ~2x steps
#   5. bench_batch() reuse one warm subprocess where safe

# ===================================================================
# 5. DOLLAR-CEILING ABORT (callbacks.py) — ties cost to the step-150 rule
# ===================================================================
def cost_abort(state, t_start, ckpt_fn,
               rate=3.95, ceiling=200.0, reserve_spent=15.0, max_h=10.0):
    elapsed_h = (time.time() - t_start) / 3600.0
    spent = reserve_spent + elapsed_h * rate            # phases A-D + live E
    if spent > ceiling:                                  # never breach $200
        ckpt_fn(); return "STOP_BUDGET"
    if elapsed_h > max_h:                                # wall-clock safety
        ckpt_fn(); return "STOP_WALLCLOCK"
    if state.step >= 150 and reward_is_flat(state):      # audit step-150
        ckpt_fn(); return "STOP_FLAT"
    return "CONTINUE"
# At 4 min/step, hitting 150 steps would take 10h = $39.5 — well under ceiling,
# so the BUDGET branch is a backstop against a stuck/forgotten container, not
# the expected exit. Expected exit = phase E ends at 12:00 by the clock, ~$33.

PSEUDO_END
```

**Risks:** H100 AVAILABILITY, not credit, is the binding scarcity: $250 buys 63h but Modal may not have an H100 free at 08:00. Mitigation: deploy + warm the container by 07:30; if no H100, fall back to A100-80GB ($2.50/hr, cheaper but ~1.5-2x slower compile/bench -> fewer steps) — A100 is acceptable for the directional delta. Verify quota at hour 0.; COLD-COMPILE TAIL blows the throughput model: if novel agent kernels routinely cold-compile at 60-120s, 4h yields ~20-30 steps not 60, and the held-out delta may be too noisy. Mitigation: hard timeout_s=90 (score 0, don't lose rollout), L1 curriculum start, G=4 fallback, and pre-recorded curve by 06:00 per audit.; FORGOTTEN WARM CONTAINER during dev silently burns the budget: an idle H100 left up overnight = $3.95/hr = ~$40 wasted by morning. Mitigation: Modal timeout on dev functions set to <=3600s (per ml-template run_gpu_tests timeout=3600), CostAbortCallback as backstop, and a manual `modal app stop` checklist item before sleeping.; ACCIDENTAL FAN-OUT cost multiplier: .spawn()-ing the grader or training as N gpu='H100' functions multiplies cost N-fold (ml-template spawns one container PER task — fine for eval, fatal for a training loop). Mitigation: training is ONE long-lived container; spawn fan-out restricted to short phase-D calibration only.; do_bench DETERMINISM on shared/contended H100: if clock-lock (-lgc) is denied in the Modal container, speedup becomes noisy -> reward noise -> GRPO instability AND inconsistent demo numbers. Mitigation: median-of-100 + L2 flush + fixed warmup as the robust fallback; NEVER time live in the demo (pre-record per audit).; BUDGET ARITHMETIC ASSUMES $3.95/hr stays current: if Modal repriced H100 since the 2026-06-21 fetch, the 50.6h ceiling shifts. Mitigation: re-read modal.com/pricing at hour 0 and update GPU_HOUR_USD in configs.py; the 6x headroom absorbs a moderate price change.; vLLM+LoRA+Triton OOM on 80GB forces a smaller config that changes throughput math: dropping to Qwen2.5-Coder-3B or num_generations=4 changes step time and step count. Mitigation: configs.py knobs (gpu_memory_utilization~0.45, G=8->4, 7B->3B) are the documented escalation ladder; all stay within the same single-H100 cost stream.


---

# Devil's-Advocate Debate


## Round 1 attacks

**Target: Protean — Full implementation plan (HUD env skeleton + Modal H100 harness + reward/verifier + shape sampler + GRPO training loop + daVinci skill-library-lite)** — worst: FEASIBILITY / CRITICAL PATH: The five components total 51.5 person-hours in a 24.5h solo window. The serial prerequisite chain (env skeleton -> harness red-team -> calibration gate -> GRPO kick) consumes the plan's own estimates of 28.5h before a single training step runs. The plan is not a plan — it is a wish list with hour labels attached to parallel tracks that are actually serial. The build dies before GRPO kicks, which means there is no delta curve, which is the only thing that wins the hackathon.
- [5] FEASIBILITY / CRITICAL PATH: The hour estimates are a fabrication. The five components total 9 + 10 + 9.5 + 5 + 9 + 9 = 51.5 person-hours on paper. The task window is 24.5h (Sat 12:30 to Sun 1PM). Solo. That is a 2.1x overcommit before a single package fails to install.: Every component plan gives its own hour estimate as if it is the only component. They are not additive-aware. The HUD env skeleton (9h) + Modal harness (10h) alone = 19h of the available 24.5h before a single line of the GRPO loop, shape sa  → demand: Collapse to three concrete deliverables with a hard schedule: (A) By Sat 6PM — one hardcoded fused-elementwise task loops end-to-end: kernel_src in, allclose+sp
- [5] FEASIBILITY / MULTI-TURN trl environment_factory IS THE SILENT SCHEDULE KILLER: The GRPO training loop component bets on trl's environment_factory for multi-turn rollouts. This API is documented as experimental/unstable even in mid-2026.: The trl GitHub issue tracker (issue #4543) explicitly documents multi-step training failures in GRPOTrainer where server-mode GRPO loses per-step prefixes in multi-turn trajectories. The plan's fallback ('Budget 1h to validate the factory p  → demand: Write the single-turn GRPO path FIRST, unconditionally. It uses trl's stable reward_funcs interface: prompt -> one completion -> grade() -> scalar reward. This 
- [4] FEASIBILITY / daVinci SKILL LIBRARY LITE IS SCOPE CREEP THAT WILL SINK THE BUILD: The skilllib component is described as 'additive to the core env+grader+training path' and given its own 9h estimate. In a 24h solo window it is not additive — it is competitive with the primary deliverable.: The daVinci paper's ablation study (Table 2, Ablation 5: 'no skills throughout') shows that removing the skill library entirely still achieves Level 2 Fast1 of 51.6% on 8B — EXCEEDING daVinci-kernel's 44.8% on the same metric, and only drop  → demand: Hard rule: skilllib is a SLIDE not a build target unless the env+harness+GRPO are confirmed green by Sat 8PM, which the schedule above shows is unlikely. The se
- [4] VERIFIER GAMEABILITY / THE PROFILING-RATIO TERM IS MORE FRAGILE THAN ADVERTISED AND ITS FALLBACK IS REWARD-HOLLOW: The plan relies on torch.profiler for PR = T_generated / T_total. The fallback when profiler fails is PR = 0.0, which trips the pr_min gate and awards reward 0.: torch.profiler attribution of GPU nanoseconds to specific Triton vs torch kernels on H100 adds 50-200ms overhead per bench call (confirmed in the profiling literature, including TritonForge's own profiling-guided framework). More critically  → demand: Invert the PR default: if torch.profiler attribution fails or returns noisy results (stddev > 0.3 across three profiler runs on the same kernel), default PR = 1
- [4] VERIFIER GAMEABILITY / THE LAUNCH COUNTER MONKEYPATCH IS TIED TO A TRITON INTERNAL THAT HAS ALREADY CHANGED TWICE ACROSS MINOR VERSIONS AND CAN SILENTLY FAIL IN BOTH DIRECTIONS: The plan says 'pin triton version; test_anti_hack.py asserts a KNOWN-GOOD kernel registers >0 launches AND a never-launched one registers 0.' This is the right instinct but the plan does not specify WHICH internal to patch. KernelGYM instru  → demand: The launch counter must be tested IN the Modal image, not locally. Add a modal run kernelforge.harness.modal_app::smoke_test function that (a) imports triton, (
- [4] FEASIBILITY / THE MODAL H100 IMAGE BUILD IS A HIDDEN 2-4H BOTTLENECK THAT SITS AT HOUR 0 OF THE WINDOW AND BLOCKS EVERYTHING DOWNSTREAM: The plan treats 'bake TRITON_CACHE_DIR into the Modal image' as a bullet point. In practice, a Modal H100 image build that installs torch+triton+trl+vllm+rank_bm25+hud-python in a single Dockerfile takes 20-40 minutes per build attempt. The  → demand: Do the Modal image build on Friday evening before the hackathon starts, or in the first 30 minutes of Saturday using the known-good layer order from ml-template
- [4] RL SIGNAL / THE 7B BASE MODEL MAY PRODUCE NEAR-ZERO COMPILABLE TRITON KERNELS AND THE CALIBRATION GATE THRESHOLD IS NOT VALIDATED AGAINST ACTUAL QWEN2.5-CODER-7B OUTPUT: The plan requires the calibration gate to pass with >=2% allclose before GRPO kicks. TritonForge reports Level 2 Pass@1 of 8% for Qwen3-8B (a stronger base than Qwen2.5-Coder-7B) and Level 1 Pass@1 of 18%. The plan's op curriculum starts at  → demand: The calibration gate must check REWARD VARIANCE, not just allclose rate. Specifically: run 30 rollouts on the easiest task (single-element fused multiply-add), 
- [3] SCHEDULE / THE STEP-150 ABORT RULE IS FANTASY AT REALISTIC OVERNIGHT THROUGHPUT: THE PLAN WILL NEVER REACH STEP 150: The plan sets a 'step-150 abort rule.' Each GRPO step requires: group=8 rollouts x (JIT compile 5-30s + allclose 2s + timed bench 3s + profiler 1s) = 88-280s per step, PLUS vLLM generation time for a 7B model on up to 2048 tokens = 10-30s p  → demand: Re-anchor the overnight claim to what is actually achievable: 50-80 GRPO steps, not 150+. The abort rule becomes 'step-50 checkpoint: if reward_mean has not inc
- [3] CTO KILL SHOT / THE HELD-OUT SHAPE GENERALIZATION CLAIM IS THE MOAT BUT THE SHAPE SAMPLER HAS A SUBTLE DISJOINTNESS FAILURE MODE THAT WILL BE CAUGHT IN JUDGE Q&A: The shape sampler's disjointness guarantee is: train shapes snap to {256, 512, 1024, 2048}, test shapes from {384, 768, 1536}. A CTO judge will immediately ask: 'A Triton matmul kernel that works at M=512 is identical in code to one that wo  → demand: Reframe the generalization claim honestly and defensibly: 'The held-out shape split proves the reward function measures real execution performance on unseen inp
- [3] CTO KILL SHOT / REWARD FORMULA CONFLICT: THE PLAN USES THREE DIFFERENT REWARD FORMULAS ACROSS COMPONENTS AND THEY ARE MUTUALLY INCONSISTENT: The HUD env skeleton pseudo says: 'yield 0.0 if not ok else min(spd / P_TARGET, 1.0)' — reward in [0,1]. The bench harness pseudo says: 'R = C(y)*(1 + speedup + PR)' — reward in [0, ~3+]. The reward/verifier component says: 'CORRECT_FLOOR=0  → demand: Pick ONE formula NOW and make it canonical across all files. Recommendation: R = 0 if not (allclose AND dtype_ok AND launches > 0) else clip((speedup / P_TARGET
- [3] INVESTOR KILL SHOT / THE DAAVINCI INTEGRATION STORY IS BACKWARDS: YOU ARE USING A JUNE 2026 SOTA PAPER AS A COMPONENT PLAN, BUT THE PAPER USED A 14B MODEL TRAINED FOR DAYS WITH 16 PARALLEL ROLLOUTS PER TASK — NOTHING IN DAVINCI IS VALIDATED AT 7B WITH 50 GRPO STEPS: daVinci-kernel-8B achieves Level 2 Fast1 of 44.8%, Level 3 Fast1 of 10.1% (Table 1). But daVinci-kernel was trained using Qwen3-8B-Base with a structured SFT cold start on Dr.Kernel trajectories (a multi-day process), then RL for hundreds o  → demand: Be precise in the demo about what daVinci techniques actually land in the 24h build: (1) the PR term in the reward formula (Eq1) — this is real, 50 lines, verif
- [3] DEMO RISK / THE 'LIVE 2X SPEEDUP STOPWATCH' IS STILL IN THE PLAN DESPITE THE AUDIT EXPLICITLY BANNING LIVE TIMING: phase13_final_survivors.md says: 'Demo / WOW: live hero kernel 4.2->2.0 ms.' phase14_deep_research_audit.md says explicitly at line 42: 'Lock GPU clocks, do_bench median-of-100, L2 flush; present static before/after bar + generalization cur  → demand: Remove the phrase 'live demo stopwatch' from the harness component plan. Replace with 'pre-recorded bench artifact.' The bench service should expose a replay en

**Target: Protean — 24h solo build plan (reward-hacking soundness + feasibility lens)** — worst: VERIFIER SOUNDNESS: The profiling-ratio (PR) gate — the plan's primary defense against the partial-runtime exploit — is simultaneously (a) built on torch.profiler attribution that is version-fragile and can segfault, causing the plan's own fallback (pr=0.0) to drive all rewards to 0 and collapse GRPO training, and (b) does not block the delegating-wrapper exploit where a Triton kernel legitimately fires but passes arithmetic to cuBLAS. This means the verifier's most novel anti-hack mechanism either destroys training or fails to block the attack it was designed to stop, with both failure modes presenting as a plausible-looking reward signal. Replace torch.profiler with a deterministic binary AST+launch-counter gate before Saturday night or the overnight run cannot be trusted regardless of what the reward curve shows.
- [5] VERIFIER SOUNDNESS — The profiling-ratio (PR) term is the plan's own primary anti-hack mechanism, and it is built on torch.profiler GPU-kernel attribution, which is version-fragile, adds 50-200ms per bench call, can segfault on malformed Triton kernels, and — critically — can mis-attribute GPU time when CUDA streams interleave Triton and cuBLAS kernels on H100. The plan's own mitigation ('wrap PR in try/except → pr=0.0') means a flaky profiler silently drives reward to 0 for every rollout that hits the bug, producing an all-zero group and GRPO gradient collapse. That is not fail-closed on the exploit; it is fail-closed on training. The exploit the PR term is meant to stop — a Triton kernel that fires on a trivial sub-op while PyTorch/cuBLAS handles the heavy work — is not actually blocked by PR alone: the agent can write a Triton wrapper that launches on the FULL tensor but immediately hands off to an @triton.jit-decorated stub that calls tl.load and tl.store around a tl.dot (which maps to a cuTENSOR call), so the launch counter fires, PR is near 1.0, but the actual arithmetic is cuBLAS. The binary launch-counter (Layer 2) catches the never-launched case but not this delegating-wrapper case. The plan acknowledges this only obliquely as 'PR gate is the real backstop' without specifying what exactly it blocks.: The profiling-ratio gate simultaneously (a) risks training collapse when torch.profiler is flaky and (b) does NOT block the delegating-wrapper exploit where a Triton kernel legitimately fires but delegates heavy arithmetic to cuBLAS primiti  → demand: Replace torch.profiler as the PR measurement mechanism with a deterministic binary alternative: use the existing LaunchCounter monkeypatch to count launches in 
- [4] REWARD HACKING — The four-layer anti-hack guard has a sequencing hole: Layer 1 (AST ban) runs on the SUBMITTED source, but the grader subprocess-executes the kernel. A model that knows the ban list can pass a source that IMPORTS a helper module at runtime (importlib.import_module inside the @triton.jit body) where the banned torch.matmul call lives. The AST ban sees a clean source; the executed kernel calls torch.matmul. This is not theoretical — the 7B model will not discover it, but a GRPO-trained model that gets partial reward for 'compiles but slow' has gradient pressure toward exactly this kind of indirection as training progresses past step 50. The plan's subprocess isolation only helps if the subprocess itself is sandboxed from the host Python environment, but the plan bakes everything into ONE Modal image where the agent's kernel subprocess shares the same site-packages as the grader. The import path attack is therefore open.: A GRPO-trained model can learn to launder banned torch calls through a runtime import inside the @triton.jit body, defeating the AST ban without triggering the launch counter (since the Triton JIT still fires). This attack vector opens as t  → demand: In the subprocess runner, set PYTHONPATH to an isolated venv that contains ONLY triton, torch (for correctness check only), and the op reference. Block importli
- [4] FEASIBILITY — The plan allocates 9h to the env skeleton, 10h to the harness, 9.5h to the reward+verifier, 5h to the shape sampler, 9h to GRPO, and 9h to the skill library — a sum of 51.5 component-hours for a 24h solo build. The plan acknowledges this as 'components, not serial hours' but the critical path is strictly serial: the harness must be green before the verifier can be red-teamed, the verifier must pass the red-team before the calibration gate can run, and the calibration gate must pass before GRPO can kick. The actual serial critical path is: Modal image build + TRITON_CACHE_DIR bake (2h, first time on a fresh Modal account always takes longer than expected due to layer caching misses) → ONE op end-to-end loop (2h) → four anti-hack layers closed + red-team (2h) → shape sampler + disjointness test (2h) → calibration gate (1h) → GRPO kick (1h to wire) = 10h before GRPO even starts. That leaves 14h for GRPO to run, which at 30-100 steps overnight is plausible — but ONLY if nothing in the first 10h requires a second iteration. The plan's own risks section lists six known failure modes in that path (KernelGYM license, TRITON_CACHE_DIR path divergence, launch-counter Triton version pin, trl environment_factory API instability, vLLM OOM, PR profiler crash). Any single one of these requires 1-3h of unplanned debugging, which collapses the 8AM kick into a 10AM kick, which collapses the 1PM delta into 'no delta, pre-recorded curve only.': The 24h timeline is feasible only if zero of six known serial failure modes materialize. The probability of zero failures across six is the product of six independent p=0.7-0.8 success probabilities, yielding a 12-26% chance of hitting the   → demand: Enforce a Friday-night pre-build gate as a hard prerequisite, not a recommendation: (1) Modal image with Triton + TRITON_CACHE_DIR bake must be GREEN before Sat
- [4] REWARD HACKING — The post-timing allclose re-check (the plan's defense against async-output-overwrite attacks) is described as a separate allclose call after the timed pass. But the plan's bench subprocess runs in a SINGLE process where the agent's kernel object is still in memory after timing. A kernel that writes its output tensor during the warmup passes and then overwrites it with the correct answer during the post-timing allclose can pass both gates. This is the 'async overwrite' attack the plan mentions. The described mitigation ('post-timing allclose re-check') only catches this if the re-check uses a FRESH random input that was NOT present during timing — but the plan's pseudocode shows the same input tensors (x, y) used for both timing and the post-timing check. The second allclose call on the same tensors catches nothing.: The post-timing allclose re-check is structurally ineffective against async-overwrite attacks because it reuses the same input tensors that were present during timing. The kernel can write the correct answer during warmup, run fast (empty o  → demand: Generate a THIRD random seed for the post-timing correctness re-check, using inputs that are materialized AFTER the timing loop completes and are never passed t
- [3] DAVINVI SCOPE CREEP — The plan integrates daVinci at five separate points across four components: PR reward in the harness, skill-library JSONL schema in skilllib/, BM25+LLM rerank in retrieve.py+select.py, Summary Agent in summarize.py, and TRLOO advantages in the GRPO loop. daVinci's own paper required a structured SFT cold-start (three phases: Summary SFT → seed library → Selection SFT → Policy SFT with skill injection) before RL even begins, using GPT-5.4 as the strong-model policy generator. Without the SFT cold start, the skill library starts empty, BM25 retrieves nothing, LLM rerank selects nothing, and the skill-injection path degrades to the null-scheme baseline — which is just plain Protean with extra latency. The plan's workaround is a 'seed snapshot' (skills_step0.jsonl) but provides no concrete source for the seed content. Writing 5-10 high-quality Triton optimization skill entries by hand under time pressure is 2-3h of unplanned work that is not in any hour estimate. The daVinci ablation table (Table 2, Ablation 5 = no skills throughout) shows that skill-free RL still achieves Fast1 51.6% at 8B — competitive with the full system at early steps. The marginal gain from the skill library accrues after step 150+ of joint RL, which a 24h run will not reach.: The skill-library component (9h estimate) delivers near-zero marginal value over plain GRPO in a 24h run because (a) the seed library requires manual authoring that is not budgeted, (b) the Summary Agent cannot trigger meaningfully until th  → demand: Hard-scope the skill-library to: (a) a hand-authored seed JSONL with exactly 5 skills (tiled_matmul, vectorized_load, shared_mem_reduction, persistent_kernel, f
- [3] VERIFIER SOUNDNESS — The held-out shape split claim ('train M in {256,512,1024,2048}, test M in {384,768,1536}') is presented as the generalization moat, but the split is not actually disjoint in the way the plan claims. The plan's own splits.py pseudocode describes 'DISJOINTNESS BY CONSTRUCTION: train draws from a fixed discrete grid; test draws from the complementary continuous range.' However, 384, 768, and 1536 are exact powers-of-two multiples of 128 and 192 — they appear in virtually every CUDA tutorial and Triton documentation example. The 7B model's pretraining corpus certainly contains Triton kernels optimized for M=384 (3/4 of 512), M=768 (3/4 of 1024), and M=1536 (3/4 of 2048). 'Held-out' here means held-out from the RL training distribution, not from the pretraining distribution. The plan acknowledges this ('held-out shapes may sit in the 7B pretraining distribution') but the mitigation ('frame as directional') is a presentation fix, not a scientific fix. A judge who asks 'how do you know this is generalization and not pretraining recall?' has a valid objection that the current split cannot answer.: The train/test shape split does not actually test generalization over memorization because test shapes {384, 768, 1536} are common in Triton pretraining data. The 'generalization verifier as the moat' claim — which the audit identifies as t  → demand: Choose test shapes that are provably unusual: use prime-adjacent or structurally odd dimensions like M in {383, 769, 1537, 2049} (one off from powers of two) or
- [3] FEASIBILITY — The trl environment_factory multi-turn API is the plan's chosen path for driving K-turn rollouts through GRPO, and the plan's own risk section concedes 'trl's environment_factory multi-turn API is younger/less stable than single-turn reward_funcs.' The plan's mitigation is 'budget 1h to validate the factory path Sat night; if red by then, ship single-turn.' But the entire daVinci PR reward, the multi-turn TRLOO advantage computation (Eq2, Eq7), and the per-turn correctness floor all require multi-turn rollouts to be meaningful. Single-turn GRPO collapses the plan to: emit one kernel, grade it, done — which is identical to KernelBench-as-a-benchmark and does not require a HUD environment at all. The 'fallback to single-turn' is not a safe fallback; it is a fallback to a non-environment that loses the primary research claim.: The multi-turn fallback to single-turn GRPO eliminates the MDP structure that distinguishes Protean from KernelBench-in-a-loop. If the trl environment_factory path fails Saturday night — which the plan assigns meaningful probability —   → demand: Implement a minimal multi-turn loop that does NOT depend on trl's environment_factory: a simple Python for-loop (for turn in range(K): completion = model.genera
- [3] CTO KILL SHOT — The LaunchCounter monkeypatch is described as patching 'triton.runtime.driver internal.' The plan's own risk section flags this: 'LaunchCounter monkeypatch is tied to a specific triton.runtime.driver internal that can change across triton versions → silently counts 0 (false hard-cap) or always-pass (hole).' The mitigation is 'pin triton version; test_anti_hack.py asserts a KNOWN-GOOD kernel registers >0 launches.' But the plan does not specify WHICH triton internal is being patched or HOW the patch is structured. KernelGYM's actual implementation (the cited open-source harness the plan recommends forking) patches the CudaDriver.launch method in triton.runtime.driver.driver. If the triton version in the Modal image is different from the version KernelGYM tested against, the attribute path may not exist and the monkeypatch silently no-ops — producing 0 launches for ALL kernels including good ones, capping all rewards at 0, and destroying the training signal. This is not a theoretical concern: the plan's Dockerfile.hud must pin triton to the exact version KernelGYM tested, but no version is specified anywhere in the plan.: The LaunchCounter anti-hack mechanism is the load-bearing Layer 2 gate and its correctness depends on a specific undocumented triton internal that the plan never pins. An unversioned Modal image that pulls latest triton will silently break   → demand: In Dockerfile.hud, pin: RUN pip install triton==3.x.y (use the exact version from KernelGYM's requirements.txt if forking, otherwise pin to the latest stable th

## Round 1 defenses

**Protean - 24h solo build: scope and schedule revisions (lead-engineer decisions)** (hrs now: 13.5)
- CRITICAL PATH (worst): 51.5 component-hours stuffed into a 24.5h solo window; serial chain env->harness->red-team->calib → LANDS. This is decisive and I am collapsing the whole plan to a single serial spine with three hard gates and ruthless cuts. The six 'components' are abolished as parallel tracks; there is ONE builder doing ONE thing at a time. New canonica (residual 3)
- Modal H100 image build is a hidden 2-4h bottleneck at hour 0 (trl/vllm torch-version conflict guaranteed; TRITON_CACHE w → LANDS HARD - this is the single thing most likely to eat Saturday morning, so it moves OFF the 24.5h clock entirely. Mandatory Friday-evening pre-build gate (G0) before Sat 12:30: (1) build image with fixed layer order FROM pytorch/pytorch: (residual 2)
- Multi-turn trl environment_factory is the silent schedule killer (experimental API, issue #4543 multi-step failures); th → LANDS. Single-turn is now the UNCONDITIONAL primary path, written first, on the spine. Use trl's stable reward_funcs interface: prompt -> one completion -> grade() -> canonical scalar. This is ~30 lines and known-good; it is what kicks at m (residual 2)
- daVinci skill-library-lite is scope creep competitive with the primary deliverable (9h est; file list still includes sum → LANDS (both targets raise it). Hard de-scope to a SLIDE-grade artifact: skills_step0.jsonl with exactly 5 hand-authored Triton skills (tiled_matmul, vectorized_load, shared_mem_reduction, persistent_kernel, fused_elementwise; each 10-20 lin (residual 2)
- Step-150 abort is fantasy: realistic 3-9 min/step gives only 53-80 steps in an 8h window; the run ends when the window c → LANDS. Re-anchored to reality. Kicking at MIDNIGHT (per critical-path revision) instead of 8AM converts the 8h training window into ~7-8h, yielding ~60-120 steps. New abort rule replaces step-150: 'step-50 checkpoint - if reward_mean has no (residual 2)
- Reward formula conflict: three+ mutually inconsistent formulas across components ([0,1] vs [0,~3] vs [0.3,~3]); whoever  → LANDS - and in a solo build this divergence is purely self-inflicted, so it is cheap to kill. ONE canonical formula in a single rewards.py imported by BOTH the hidden grade.py and the training loop, never duplicated: R = 0 if not (allclose_ (residual 1)
- Friday-night pre-build is in the audit but NOT in the implementation plan as a step; the 24h timeline is feasible only i → LANDS. Friday-night pre-build is now a HARD PREREQUISITE GATE (G0), explicitly off the 24.5h clock, with three pass/fail checks that must all be green before the Sat 12:30 start: (1) Modal image green via trivial @triton.jit kernel on a rea (residual 2)

**Protean — verifier/reward subsystem (rewards.py + hidden grade.py + bench/anti-hack harness), hardened against the verifier/reward-hacking attacks** (hrs now: 11)
- PR (profiling-ratio) gate built on torch.profiler is version-fragile, adds 50-200ms, can segfault, and its fail-closed f → LANDS (sev 5). CUT torch.profiler from the night-1 reward path entirely. Replace continuous PR with a DETERMINISTIC BINARY gate in harness/anti_hack.py: pr_ok = (launches_timed > 0) AND (not delegates_to_matrix_unit(src, op)). delegates_to_ (residual 2)
- Post-timing allclose re-check (defense vs async-output-overwrite) reuses the SAME input tensors live during the timed pa → LANDS (sev 4). 5-line fix in harness/bench_core.py: three fixed seeds with disjoint roles. seed_correct=42 (pre-timing correctness), seed_timed=43 (timing loop ONLY, never correctness-checked), seed_post=44 (fresh inputs MATERIALIZED AFTER  (residual 1)
- AST-ban runs on submitted SOURCE but grader subprocess EXEC's the kernel in shared site-packages; a trained model can la → LANDS (sev 4), scoped. Two cheap layers in subprocess_runner.py: (1) AST pass also bans dynamic-import/dynamic-attr nodes (Import/ImportFrom of importlib, Name eval/exec/__import__/compile, getattr(torch,...) Call) -> reward 0. (2) exec the (residual 2)
- LaunchCounter monkeypatch tied to an unpinned triton internal (KernelGYM patches CudaDriver.launch); torch pulls triton  → LANDS (sev 4). (1) Dockerfile.hud installs torch first then `pip install triton==<EXACT KernelGYM pin>` LAST; build asserts `import triton; assert triton.__version__==X.Y.Z` -> fail build on mismatch. (2) Patch the stable Python boundary: w (residual 2)
- Reward formula is mutually inconsistent across 4 components ([0,1] vs C(y)*(1+speedup+PR) in [0,~3] vs 0.3+speedup*PR).  → LANDS (sev 3). ONE canonical rewards.py imported by BOTH grade.py and kernel_env.py. Canonical: reward = 0.0 if not (allclose_correct AND allclose_post AND dtype_ok AND shape_ok AND launches_timed>0) else clip(0.3 + speedup/P_TARGET + PR_BO (residual 1)
- Held-out shape split (train {256,512,1024,2048}/test {384,768,1536}) does not prove generalization-over-memorization: te → PARTIALLY LANDS (sev 3). (1) Move test shapes OFF the power-of-two grid to provably-unusual values: test M in {400,800,1600} (multiples of 100, rare in GPU code) plus off-by-one {383,769,1537}; train stays on {256,512,1024,2048}. _assert_sp (residual 2)

## Round 2 attacks

**Target: Protean daVinci integration plan (daVinci scope creep vs delta-by-1PM)** — worst: SCOPE CREEP / HOUR BUDGET: The daVinci integration adds at least 6.5h of new critical-path work to a schedule that already has zero slack. The five daVinci work items (rewards.py redesign, PR term, BM25 inject, SFT cold-start filter, LOO adapter) collectively push the pre-GRPO critical path past the 8AM Sunday kick deadline in expectation, meaning the training run starts late, completes fewer steps, and the base-vs-trained delta on held-out shapes — the one thing that wins or loses the hackathon — does not exist by the 1PM submission. The daVinci paper's own ablation numbers only justify its gains in a fully co-evolved multi-agent system trained for hundreds of steps on thousands of tasks. None of those conditions hold in a 24h solo build at 60-150 GRPO steps. The correct decision is to extract exactly two daVinci contributions (the PR binary launch gate and a post-training static skill prefix for the demo hero kernel), implement both in under 90 minutes total, and spend every other hour on the generalization verifier that is the project's actual moat.
- [5] SCOPE CREEP / HOUR BUDGET: The daVinci integration adds at least 6.5h of new critical-path work that the 24h schedule has no room for, and almost none of it is load-bearing for the delta.: The plan proposes FIVE distinct daVinci-sourced work items: (1) rewards.py as a shared source-of-truth (Eq1 structure + PR term), (2) BM25 skill injection library (inject.py + skills_v0.jsonl), (3) LITE Selection Agent pass, (4) LOO advanta  → demand: Hard-cut the daVinci integration to exactly TWO items that are (a) already decided to be buildable in under 2h total and (b) directly harden the reward that was
- [3] PR TERM GAMABILITY: The profiling-ratio reward PR = T_generated / T_total is gameable in a static-library setting and produces noisy signal at low step counts.: daVinci's PR term was designed for a multi-agent co-evolving setting where the policy agent writes kernels across an entire model's forward pass, making PR a meaningful bottleneck signal (Section 3.1: 'global runtime contribution'). In KERN  → demand: Implement PR as a binary gate rather than a continuous reward term: reward=0 if measured Triton kernel launch count is zero (i.e., no Triton kernel actually fir
- [3] SKILLS_V0.JSONL COLD-START CIRCULARITY: The static skill library must be populated with skills extracted from successful rollouts, but successful rollouts do not exist at hour 0.: The KEEP-LITE decision for skill injection is justified by daVinci Ablation 1: removing skill injection at inference drops Level 2 Fast1 from 44.8% to 20.6% on the 8B model. This is the largest single-component drop in the paper. The plan c  → demand: Drop skills_v0.jsonl and inject.py from the hour 0-8 critical path entirely. The daVinci ablation gain is not reproducible without SFT cold-start conditioning o
- [4] REWARD FORMULA DIVERGENCE: rewards.py as a single shared source of truth between kernel_env.py and grade.py is architecturally correct but the plan does not specify how the hidden grade.py imports it, creating the exact silent overnight failure it claims to prevent.: The plan states: 'rewards.py imported by BOTH the hidden grade.py and kernel_env.py — never duplicated. Divergence between these two is the single most likely silent overnight failure.' This diagnosis is correct and the prescription (single  → demand: Specify the exact import mechanism before writing a single line of rewards.py. Recommended: copy rewards.py into the same directory as grade.py at Modal image b
- [4] CTO KILL SHOT / DIFFERENTIATION: The integrated plan is now three papers glued together (daVinci reward + KernelGYM anti-hack + Dr.Kernel SFT data) with the one genuine contribution — the held-out continuous shape generalization verifier — buried under implementation noise.: A CTO judge who has read daVinci (published June 15, one week before the hackathon), Dr.Kernel, and KernelGYM will look at the integrated plan and see: PR reward term from daVinci Eq1, BM25 skill injection from daVinci Section 3.3, launch i  → demand: Reorganize the demo narrative around the one genuine contribution. The money slide must be: 'Train shapes: M in {256, 512, 1024, 2048}. Test shapes: M in {384, 

**Target: Protean — daVinci Integration Plan (Reward Correctness + PR-term Gaming Axis)** — worst: REWARD DIVERGENCE: rewards.py imported by kernel_env.py but grade.py baked into a deployed Modal image at a different time — a silent constant mismatch (P_TARGET, allclose tolerance, PR gate threshold) between the training reward and the grading reward produces an inverted or zero delta by 1PM Sunday with no error message and no obvious root cause. This is the single most likely cause of project death because it requires no exploit, no model failure, and no infrastructure outage — just one late-night constant tweak to rewards.py after the image is already deployed.
- [5] REWARD DIVERGENCE: rewards.py is not a single source of truth — it is aspirational: The plan declares rewards.py as 'THE single source of truth imported by BOTH grade.py AND kernel_env.py — never duplicated,' but this is an architectural wish, not a build fact. The verilog-template grader (grade.py at /starters/verilog-tem  → demand: At image build time, copy kernelforge/rewards.py into /donotaccess/rewards.py with a file-hash assertion at the top of grade.py: assert hashlib.sha256(open(__fi
- [4] PR-TERM GAMING: the profiling-ratio reward term is gameable in the exact direction it claims to prevent: daVinci Eq1 defines PR = T_generated / T_total, rewarding kernels that dominate runtime. The Protean plan ports this as a bonus term: R = C(y) * (1 + speedup + PR). The exploit is immediate and was not closed by the four anti-hack laye  → demand: Do not add PR as an additive bonus. Replace it with a multiplicative gate: R = C(y) * speedup_score * PR_gate, where PR_gate = 1 if (measured Triton GPU time / 
- [4] DAVINCI INTEGRATION HOURS vs GENERALIZATION VERIFIER HOURS: the daVinci components consume exactly the hours the moat requires: The phase14 audit is explicit: the held-out continuous-shape generalization verifier is the single defensible white space (KernelBench roadmap item #74, unshipped; Dr.Kernel/DRTriton/Kevin all train and test on the same distribution). The p  → demand: Cut Skill-Library-Lite entirely from the 24h critical path. Replace it with a 30-minute stub: a single hardcoded system-prompt prefix containing the top-1 skill
- [3] SFT COLD-START FILTER IS A 2H GAMBLE ON AN UNVERIFIED SCHEMA: The SFT decision plan filters hkust-nlp/drkernel-coldstart-8k by final_speedup >= 1.2, which assumes the Parquet schema has a column named exactly 'final_speedup'. The plan cites this as a verified fact: '8920 rows, ~163MB Parquet, columns:  → demand: Before any SFT pipeline code: run a 10-line schema probe locally against the actual downloaded Parquet file and print all column names, dtypes, and the distribu
- [3] ALLCLOSE + TIMING SPLIT: the two-pass design has a race condition that manufactures false positives: The reward plan runs correctness (allclose on N random inputs) and then timing as two separate passes. The daVinci plan and the KernelGYM pattern both implicitly assume the kernel is stateless across calls. A Triton kernel with a hidden @tr  → demand: Use FRESH random inputs for EVERY timed call — not a fixed tensor reused across warmup and measurement. Generate a new random seed per timing iteration: for i i
- [3] GRPO GRADIENT COLLAPSE AT 7B IS THE SILENT 8AM KILL THAT THE CALIBRATION GATE DOES NOT FULLY PREVENT: The plan's calibration gate requires >=5% compilable, >=2% allclose-passing, and nonzero group variance. This gate correctly prevents kicking GRPO on all-zero reward, but it does not prevent kicking on near-degenerate reward. A group of 8 r  → demand: Add a secondary calibration gate: require that at least 1 rollout in each group achieves nonzero SPEEDUP reward (not just allclose or compile partial credit) ac

## Round 2 defenses

**Protean — Final daVinci KEEP/CUT scope + canonical reward spec (lead-engineer revision after 2 red-team passes)** (hrs now: 3)
- SCOPE CREEP / HOUR BUDGET: 5 daVinci items (~6.5-7h) crowd out the generalization verifier and push past the 8AM GRPO ki → ACCEPTED. Final daVinci KEEP list is exactly TWO items, total budget 1.5h, both folded INTO work already on the critical path (no new serial nodes): (1) PR-as-binary-launch-gate inside rewards.py (~0.5h, folded into the reward build that al (residual 2)
- PR-TERM GAMABILITY + single-op no-op: PR=T_gen/T_total collapses to ~1.0 at single-op scope (constant offset GRPO cancel → ACCEPTED — replace daVinci's additive continuous PR (Eq1: R=C*(1+speedup+PR)) with a MULTIPLICATIVE BINARY GATE. Final: pr_gate = 1.0 if (measured Triton GPU time / total GPU time) > tau ELSE 0.0, with tau=0.5; reward = correct_floor * spee (residual 2)
- REWARD DIVERGENCE (worst, severity 5): rewards.py as 'single source of truth' is aspirational. The confirmed verilog sca → ACCEPTED — this is now the FIRST integration test, not the last. Verified against the real scaffold: grade.py is self-contained and importlib-loaded from a donotaccess path (grader.py:_load_grade_module uses spec_from_file_location), exactl (residual 2)
- SKILLS_V0.JSONL COLD-START CIRCULARITY + inert-without-SFT: daVinci Ablation 1 (44.8->20.6%) measures removal from a TRA → ACCEPTED — skills_v0.jsonl + BM25 inject.py CUT from the hour-0-8 critical path entirely. The ablation gain is unreproducible without SFT skill-conditioning, which is itself cut. Replaced by: at hour 16-18, IF the overnight run produced >=3 (residual 2)
- ALLCLOSE+TIMING SCRATCHPAD RACE: kernel seeds a global-memory scratchpad on the correctness pass, returns cached result  → ACCEPTED — mandatory timing protocol in rewards.py: FRESH random inputs for EVERY timed iteration (new seed per call), never a reused warmup tensor. `for i in range(M): inp = torch.randn(shape, device='cuda', generator=torch.Generator('cuda (residual 2)
- GRPO GRADIENT COLLAPSE / compile-only variance: a group where 1/8 compiles and 0/8 passes allclose has variance>0, passe → ACCEPTED — add a SECONDARY calibration gate beyond compile/allclose/variance: require >=1 rollout with NONZERO SPEEDUP reward (not just compile/allclose partial credit) in >=2 of the 5-10 calibration tasks before kicking overnight GRPO. Esc (residual 3)
- CTO KILL SHOT / DIFFERENTIATION: integrated plan reads as 3 papers glued together; the one genuine contribution (held-ou → ACCEPTED — demo narrative reorganized around the single moat. Money slide: 'Train shapes M in {256,512,1024,2048}; Test shapes M in {384,768,1536} — NEVER seen in training. Base reward on test shapes: X%. Trained: Y%. The gap proves general (residual 2)
- SFT COLD-START SCHEMA GAMBLE: filter assumes a column literally named final_speedup, 'verified' only from the HF dataset → ACCEPTED — SFT cold-start CUT from critical path (already removed under the scope-creep fix). If revisited only as deep contingency: gate the ENTIRE SFT branch behind a 10-line local schema probe (print all column names, dtypes, target-colu (residual 2)

**Protean reward decisions — final KEEP/CUT for daVinci features + canonical reward spec (lead-engineer revision)** (hrs now: 2)
- SCOPE CREEP / HOUR BUDGET: daVinci integration adds >=6.5h critical-path work, missing 8AM GRPO kick (severity 5, worst) → ACCEPT in full. FINAL daVinci KEEP/CUT list: KEEP exactly TWO items, both reward-hardening, both <90min total. (1) PR-as-binary-launch-gate in rewards.py (~20 lines, 30min). (2) Post-training static skill prefix for the hero demo kernel ONL (residual 2)
- PR-TERM GAMING + PR-TERM GAMABILITY: additive PR=T_gen/T_total is exploitable (wrap torch in a trivial @triton.jit so la → ACCEPT both. Verified arithmetically: adding a near-constant to all rewards in a group leaves GRPO advantages unchanged to 2e-16 (single-op T_total≈T_generated => PR≈1.0 constant => zero training signal). FINAL: CUT the continuous additive  (residual 2)
- REWARD FORMULA DIVERGENCE / rewards.py-is-aspirational: shared source-of-truth import mechanism unspecified; hidden grad → ACCEPT in full — verified ground truth: HUD grader.py loads grade.py via importlib.util.spec_from_file_location from per-task donotaccess/grade.py into its OWN module namespace (no shared import, runs inline like verilog-template). FINAL me (residual 2)
- SKILLS_V0.JSONL COLD-START CIRCULARITY: ablation gain requires SFT-on-skill-injected-prompts; a base Qwen2.5-Coder-7B no → ACCEPT. CUT skills_v0.jsonl + inject.py from the hour-0-8 critical path entirely. The daVinci Ablation-1 gain (L2 Fast1 44.8%->20.6%) is non-reproducible without skill-conditioned SFT, which is itself cut. FINAL: skill injection becomes a P (residual 2)
- SFT COLD-START IS A 2H GAMBLE ON UNVERIFIED SCHEMA: 'final_speedup' column name asserted from the HF card, not a local r → ACCEPT. SFT stays CUT from the default path (calibration-gated). FINAL guard if SFT is ever attempted: a 10-line schema probe runs FIRST against the actually-downloaded Parquet — print all column names, dtypes, and target-column distributio (residual 2)
- ALLCLOSE+TIMING RACE / SCRATCHPAD CACHING: kernel seeds a global-memory scratchpad on the correctness pass, returns cach → ACCEPT. FINAL timing protocol mandated in rewards.py and asserted in grade.py: FRESH random inputs for EVERY timed iteration, never a fixed reused tensor — `for i in range(M): inp = torch.randn(shape, device='cuda', generator=torch.Generato (residual 2)
- GRPO GRADIENT COLLAPSE / DEGENERATE-VARIANCE CALIBRATION: gate (>=5% compile, >=2% allclose, nonzero variance) passes on → ACCEPT. FINAL secondary calibration gate before the overnight kick: require >=1 rollout per group to earn nonzero SPEEDUP reward (not merely compile/allclose credit) on >=2 of the 5-10 calibration tasks. Tuning ladder if it fails: lower P_T (residual 3)
- CTO KILL SHOT / DIFFERENTIATION: integrated plan reads as three papers glued together; the one genuine contribution (hel → ACCEPT. FINAL demo narrative reorganized around the single moat. Money slide: 'Train shapes M in {256,512,1024,2048}; TEST shapes M in {384,768,1536} — never seen in training. Base reward on test shapes: X%. Trained: Y%. Gap = generalizatio (residual 2)

## Round 3 attacks

**Target: Protean — solo 24h build, HUD x YC Frontier RL Environments Hackathon, Sat 12:30 to Sun 13:00** — worst: THROUGHPUT / OVERNIGHT STEP COUNT: the 40-100 realistic GRPO steps may be below the signal-emergence threshold for a 7B model on a novel reward formula without SFT warmup, producing a flat or noise-level reward curve that cannot support the primary 'base vs trained delta on held-out shapes' claim. The plan's own math confirms this risk and then dismisses it as a fallback scenario rather than treating it as the primary kill condition. This is the most likely single reason there is no visible delta at 1PM Sunday.
- [5] THROUGHPUT / OVERNIGHT STEP COUNT: The plan's own math yields 40-100 GRPO steps. daVinci Figure 3 shows gains emerge between steps 50-150 on a 14B model. At 7B the curve is unknown. 40 steps on a 7B with a novel reward formula and no SFT warmup may produce a reward curve tha  → demand: Move the GRPO kick to Saturday night, not Sunday 03:00. If calibration can clear by 21:00 Saturday, kicking at 21:30 buys 15h of GPU time and ~225 steps at 4 mi
- [5] REWARD FORMULA DIVERGENCE (SILENT GHOST SIGNAL): The plan builds rewards.py as a shared import between grade.py (baked into the Modal image at Docker build time) and kernel_env.py (loaded at GRPO runtime from the live filesystem). The sha256 hash assertion is correct in principle, but the  → demand: Do NOT bake rewards.py into the image as a static copy. Instead mirror the verilog pattern exactly: mount or volume-map the live rewards.py into the container a
- [4] CALIBRATION GATE FAILURE / GRPO ADVANTAGE COLLAPSE: The plan's calibration gate (Stage A: >=5% compilable, >=2% allclose, nonzero variance; Stage B: >=1 nonzero-speedup rollout in >=2 tasks) is sound, but the escalation ladder (P_TARGET 1.5->1.2->1.1, then L1 elementwise, then +0.1 compile c  → demand: Build the SFT warmup NOW, not as a contingency. Filter drkernel-coldstart-8k (8920 rows, MIT license, verified in the plan) to speedup >= 1.2, take 200 rows, an
- [4] PR GATE AS REWARD KILLER (ANTI-HACK BECOMES ANTI-LEARNING): The plan implements PR as a MULTIPLICATIVE binary gate: reward = 0 if pr_frac <= tau=0.5. This is more aggressive than daVinci Eq1, which uses PR as an ADDITIVE term (R = C(y) * (1 + speedup + PR)) and PRS as a SAMPLING weight (Eq4), not a   → demand: Start with the daVinci original: PR as an additive bonus (+ PR_BONUS * pr_frac) not a multiplicative gate. Keep the launch-count gate (launches > 0) as the hard
- [4] LAUNCHCOUNTER HOOK FRAGILITY VS. TRITON INTERNALS: The plan monkeypatches triton.runtime.driver's JITFunction.__call__ to count kernel launches. The plan itself flags this: 'tied to a specific triton.runtime.driver internal that can change across triton versions -> silently counts 0 (false   → demand: Validate the LaunchCounter hook end-to-end in the FIRST Modal H100 smoke test (Gate-0, hour 0), not in test_anti_hack.py which runs later. The smoke test should
- [3] HELD-OUT SHAPE GENERALIZATION CLAIM IS EMPIRICALLY WEAK AT 40-100 STEPS: The moat claim — 'train on M in {256,512,1024,2048}, test on M in {384,768,1536}, the gap proves generalization not memorization' — is scientifically defensible by construction (the disjointness is structural) but empirically weak at 40-100  → demand: Sharpen the held-out split to include at least one structurally different shape regime: for example, test shapes that cross a tile-count boundary (e.g., M=3072 
- [3] DAVINCI SKILL LIBRARY DELIVERS ZERO VALUE AT 40-100 STEPS WITHOUT SFT: The plan's skill-library-lite (BM25 retrieval from a static skills_v0.jsonl) is grounded in daVinci Ablation 1: removing skill injection drops Level 2 Fast1 from 44.8% to 20.6% on the 8B model. But daVinci's model has an SFT cold start that  → demand: Either (a) include at least a minimal skill-injection SFT pass (10-20 examples showing skill -> kernel usage) alongside the drkernel-coldstart warmup, which mak
- [3] DEMO HERO KERNEL MAY NOT EXIST AT 07:00 SUNDAY: The demo plan requires a 'best HELD-OUT allclose + speedup rollout' to exist by 07:00 Sunday. This requires: (1) at least one rollout achieving allclose=True AND speedup >= 1.1x on a held-out shape (M=768), (2) enough GRPO steps to have tra  → demand: Pre-generate the hero kernel from the BASE model on a train shape (not held-out) as a fallback, recorded at hour 0 before any training. This gives a baseline ke

**Target: Protean — HUD x YC Frontier RL Environments Hackathon, 24h solo build** — worst: REWARD-FORMULA DIVERGENCE BETWEEN DEPLOYED IMAGE AND LIVE TRAINING: The single most likely cause of losing is that rewards.py gets edited during a 2AM debug session AFTER the Modal image has been built and deployed, creating a silent divergence between the formula grade.py runs (baked at image-build time via Dockerfile COPY rewards.py /donotaccess/rewards.py) and the formula kernel_env.py imports at training time from the live file. The grader does not crash — it silently scores rollouts with a stale formula. GRPO trains on one reward signal, the demo grader evaluates on another, and the base-vs-trained delta on the money slide is computed against yet a third version if the eval pass uses a freshly rebuilt image. The plan's own risk register lists this as R1-SEV5, the sha256 hash-check mitigation is correct in principle, but it requires discipline to re-run the preflight after EVERY edit AND to rebuild the image before the eval pass — at 3AM under deadline pressure, after the calibration gate has already failed twice, this chain of steps will be skipped. The verilog-template grader.py (the scaffold being copied) uses importlib.util.spec_from_file_location to load the hidden grader at runtime from a baked path; any live edit to the source file does NOT reach that baked path. This is not a theoretical risk: it is a structural divergence between 'the file I am editing' and 'the file the running container executes', and it requires an explicit image rebuild plus a known-good/known-bad smoke assertion to close. The plan has all the pieces but they must survive human error under sleep deprivation. If this fires, the overnight GRPO run produces a curve that cannot be trusted, and there is no Sunday morning to rebuild and re-train.
- [5] REWARD FORMULA DIVERGENCE (image vs live file): The Dockerfile bakes rewards.py into /donotaccess/rewards.py at build time. Any post-build edit to the live rewards.py — inevitable given the 2AM calibration-gate escalation ladder that adjusts P_TARGET, SPEEDUP_FLOOR, and CORRECT_FLOOR — c  → demand: Enforce a one-way immutability rule: rewards.py is FROZEN at Hour 4 (before red-team, before calibration). After that point, tuning knobs live ONLY in a separat
- [4] TRITON LAUNCH COUNTER HOOK FRAGILITY: The entire 4-layer anti-hack system's hardest gate — the no-Triton-launch cap — depends on monkeypatching triton.runtime.driver or triton.JITFunction.__call__ to count kernel invocations. This is an undocumented internal API that changes ac  → demand: Add a three-assertion smoke test that runs INSIDE the Modal container at image build time (not as a local pytest): (1) a hardcoded known-good elementwise Triton
- [4] GRPO THROUGHPUT: REALISTIC STEP COUNT IS 40-60, NOT 150: The plan's own budget math arrives at 40-100 GRPO steps overnight on a single H100, with the step-150 abort rule described as an insurance mechanism that 'likely never fires on step count.' This is honest but creates a different problem: 40  → demand: De-risk the curve legibility problem by pre-computing what 'directional' actually looks like at 40-60 steps. Before kicking GRPO, run 10 steps manually and meas
- [3] HUD SDK PACKAGE NAME MISMATCH AT DEPLOY TIME: The phase14 audit explicitly flags this: 'pip show hud' and 'pip show hud-sdk' may install different packages — one is a monitoring SDK (SOC 2, function tracking) that does not provide the RL environment framework at all. The verilog-templa  → demand: In Hour 0, before any other action: copy pyproject.toml verbatim from /starters/verilog-template/. Do not recreate it. The verilog-template's pyproject.toml alr
- [4] PR GATE AS MULTIPLICATIVE ZERO: TRAINING SIGNAL DESTRUCTION: The plan implements pr_gate as a MULTIPLICATIVE binary gate: reward = 0.0 if pr_frac <= TAU=0.5. This is correct for anti-hack purposes but creates a gradient cliff: any rollout where the Triton kernel covers less than 50% of measured GPU t  → demand: For the first 50 training steps, implement PR as a BONUS term (not a gate): reward = (CORRECT_FLOOR + speedup_score) * (1 + PR_BONUS * pr_frac), where PR_BONUS 
- [4] DEMO KILL SHOT: 'THIS IS KERNELGYM WITH A SHAPE SAMPLER': A CTO judge who has read the KernelGYM paper (hkust-nlp/KernelGYM, Feb 2026, open-source, MIT license) will note that KernelGYM already implements: subprocess isolation, Triton launch instrumentation for anti-hack, allclose correctness gati  → demand: The answer to this kill shot must be pre-rehearsed and specific, not generic. Prepare the exact sentence: 'KernelGYM evaluates on fixed KernelBench shapes that 
- [3] MULTI-TURN VIA environment_factory: UNVALIDATED IN TRL AT THIS SCALE: The plan designates multi-turn via trl's environment_factory as a 'stretch' with a 1-hour validation budget at Hour 8, with the fallback being single-turn GRPO. This is the correct risk posture but the consequences of the fallback are under  → demand: Validate environment_factory at Hour 1, not Hour 8. Build the SIMPLEST possible 2-turn loop immediately after the first single-turn reward flows: turn 1 yields 
- [4] CALIBRATION GATE STAGE B IMPOSSIBLE TO PASS AT BASE 7B WITHOUT SFT: Stage B of the calibration gate requires 'at least 1 rollout earning nonzero SPEEDUP reward on at least 2 of the 5-10 tasks.' Nonzero speedup reward requires: (1) allclose passes, (2) Triton launches > 0, (3) pr_frac > TAU, AND (4) speedup   → demand: Pre-build the SFT contingency on Friday night before the hackathon starts. Filter 200 rows from hkust-nlp/drkernel-coldstart-8k (final_speedup >= 1.2, messages 

**Target: Protean — daVinci integration honesty audit + full plan Devil's Advocate pass** — worst: CRITICAL PATH COLLISION — rewards.py HASH PREFLIGHT IS UNENFORCEABLE AT IMAGE BUILD TIME
- [4] DAVINCI CARGO-CULT #1 — PR REWARD TERM IS STRUCTURALLY MISPORTED: The plan ports daVinci Eq1 as R = CORRECT_FLOOR + speedup_score * pr_gate (multiplicative binary gate). The actual daVinci Eq1 is R = C(y) * (1 + speedup + PR) where PR is an ADDITIVE continuous term, not a multiplicative binary gate. The p  → demand: Either (a) implement the actual continuous PR additive term from Eq1 with a soft clip rather than a hard gate, OR (b) relabel it in the plan as 'daVinci-inspire
- [4] DAVINCI CARGO-CULT #2 — SKILL INJECTION CLAIM COLLAPSES TO A JSONL FILE READ: The plan's 'daVinci integration' for the skill library degrades across its own sections: the full plan calls for BM25 + LLM rerank + Summary Agent + execution verification (9h), then the KEEP-LITE decision cuts it to 'BM25-only, no LLM rera  → demand: Drop the Ablation 1 citation as justification for the static prefix. Either (a) do the SFT cold-start with skill-conditioned examples (as daVinci Section 3.6 ac
- [3] TRLOO vs GRPO MISMATCH — THE RL ALGORITHM CITED IS NOT THE ONE SHIPPED: The plan repeatedly cites daVinci's TRLOO (Turn-level REINFORCE Leave-One-Out) as the advantage estimator, including the LOO formula G_{i,t} - (1/(N_t-1)) * sum_{j!=i} G_{j,t} from Eq2. However, the actual training component ships trl GRPOT  → demand: Rename the RL algorithm section. State clearly: 'We use trl GRPO (group normalization), not daVinci TRLOO; daVinci's TRLOO is the stretch target for multi-turn.
- [4] DEMO RISK — THE LIVE HERO KERNEL DEPENDS ON A MODAL COLD-START THAT WILL TAKE 25-90 SECONDS ON CONFERENCE WIFI: Slide 4 plans to run 'bash demo/hero_run.sh' live on a held-out shape M=768. The plan's own throughput math (Budget section) shows COLD rollout = 35-130 seconds when the agent's kernel is novel. Even with TRITON_CACHE_DIR baked, the hero ke  → demand: Pre-warm the hero kernel IN the Modal container before the demo begins: run hero_run.sh once (off-screen) to populate the Triton cache for that exact source str
- [5] CRITICAL PATH COLLISION — rewards.py HASH PREFLIGHT IS UNENFORCEABLE AT IMAGE BUILD TIME: The plan's R1 risk mitigation (the highest-severity risk) relies on grade.py asserting sha256(rewards.py) == REWARDS_HASH at startup, where REWARDS_HASH is a constant baked into grade.py at image build. The Dockerfile line is 'COPY rewards.  → demand: The hash check must be end-to-end: at calibrate.py startup AND at each GRPO step, import rewards.py from the SAME path that grade.py uses (/donotaccess/rewards.
- [3] THROUGHPUT MATH IS OPTIMISTIC BY 3-5x — 40-100 STEPS CANNOT SHOW A STATISTICALLY MEANINGFUL CURVE: The Budget section honestly calculates 40-100 GRPO steps overnight. However, the plan's own money slide requires a 'train reward vs held-out reward curve' with enough steps to show a trend separable from noise. At 40 steps with num_generati  → demand: Add explicit confidence intervals or error bars to the money slide. Change the spoken numbers from 'Base: X%, Trained: Y%' to 'Base: X +/- sigma1, Trained: Y +/
- [3] FEASIBILITY — THE MULTI-TURN environment_factory PATH IS A 1H VALIDATION BUDGET AGAINST AN UNSTABLE API: The plan allocates 1 hour to validate trl's environment_factory multi-turn path (Block 4, '1h-BUDGET stretch: validate environment_factory multi-turn. RED by 23:30 => ship single-turn, no debate'). trl's environment_factory is documented as  → demand: Remove multi-turn from the critical path entirely. Plan single-turn GRPO as the ONLY path. Multi-turn via environment_factory is a post-submission stretch goal 
- [3] INVESTOR KILL SHOT — THE MOAT CLAIM ('GENERALIZATION VERIFIER NO ONE HAS') IS NARROWER THAN CLAIMED: The plan's investor pitch is 'the held-out continuous-shape verifier is the moat — KernelBench roadmap #74 lists shape-sweep as unshipped.' This is accurate as of the phase14 research date. However, daVinci-kernel (the very paper this plan   → demand: Choose one framing and commit to it. OPTION A (stronger, riskier): claim the train/test shape gap as evidence of generalization but only if you can show the gap
- [4] ANTI-HACK GUARD — THE LaunchCounter MONKEYPATCH HAS A DOCUMENTED SILENT FAILURE MODE THAT THE PLAN'S OWN MITIGATION DOES NOT FULLY CLOSE: The plan patches triton.runtime.driver internals to count Triton kernel launches. The plan acknowledges this is fragile (R4 risk: 'triton version drift => hook counts 0 OR always-pass') and mitigates with test_anti_hack.py asserting known-g  → demand: Test the launch counter on a PRE-CACHED kernel (one that has been compiled once and whose .cubin is in TRITON_CACHE_DIR). Add this as test_anti_hack.py case 5: 
- [3] CTO KILL SHOT — 'SO IT IS JUST KERNELBENCH WITH A SHAPE SAMPLER AND A PR GATE': A CTO judge who has read daVinci, Dr. Kernel, and KernelGYM (all published 2026, all in the plan's own bibliography) will observe: (1) KernelGYM already has subprocess isolation and launch-count anti-hacking; (2) Dr. Kernel already has TRLO  → demand: Prepare a one-paragraph pre-emptive answer to 'how is this different from KernelGYM + GRPO' that does NOT rely on the shape split (too thin) and does NOT rely o

## Round 3 defenses

**Protean — final delta-certainty revisions (HUD x YC Frontier, 24h solo, kick Sat-night, delta due Sun 13:00)** (hrs now: 15.5)
- REWARD-FORMULA DIVERGENCE / HASH PREFLIGHT UNENFORCEABLE — two files implement the formula; grade.py loads /donotaccess/ → STRUCTURAL collapse to one path, zero copies. (1) rewards.py is the ONLY formula file and lives ONLY at /donotaccess/rewards.py inside the image (HIDDEN_ROOT confirmed = /donotaccess in scenario_helpers.py:17). (2) kernel_env.py does NOT `i (residual 1)
- THROUGHPUT / OVERNIGHT STEP COUNT — own math = 40-100 steps; daVinci/Dr.Kernel needed hundreds. 40 steps on a cold 7B ma → Move the kick EARLIER and de-risk legibility. (1) Compress calibration into the harness/red-team block (parallel, not serial) so the GO/NO-GO lands by ~21:00 Sat; kick at 21:30 Sat buys ~15h GPU ≈ 200+ steps at ~4min/step. The 03:00/08:00 k (residual 3)
- CALIBRATION GATE STAGE B IMPOSSIBLE AT BASE 7B WITHOUT SFT / GRADIENT COLLAPSE — Stage B needs ≥1 nonzero-speedup rollou → Make SFT cold-start UNCONDITIONAL and PRE-BUILD it Friday night (off the clock). Filter hkust-nlp/drkernel-coldstart-8k (MIT, 8920 rows, verified) to final_speedup>=1.2, take ~200 rows whose messages contain a working Triton kernel, run 1 e (residual 2)
- PR GATE AS MULTIPLICATIVE BINARY ZERO = ANTI-LEARNING — reward=0 if pr_frac<=0.5 zeroes correct-but-partial kernels (Tri → Start with daVinci's actual Eq1 shape and the ROBUST hard gate only. reward = (CORRECT_FLOOR + speedup_score) * (1 + PR_BONUS*pr_frac), PR_BONUS=0.2, pr_frac clipped [0,1]. The ONLY hard zeros are the robust, unambiguous gates: allclose+dty (residual 2)
- LAUNCHCOUNTER HOOK FRAGILITY incl. CACHED-PATH SILENT FAILURE — monkeypatch of triton JITFunction.__call__ can count 0 o → (1) Validate the hook in the Hour-0 Modal H100 smoke test, before any other grader code: known-good kernel -> launches>0, never-launched -> launches==0, hook survives try/except wrapper. (2) ADD the cache-path case: compile a kernel once so (residual 2)
- MULTI-TURN environment_factory ON CRITICAL PATH — 1h validation budget at Hour 8 against an experimental trl API; if the → Remove multi-turn from the critical path. Single-turn GRPO via trl reward_funcs is the ONLY committed path (rock-solid). Build the simplest 2-turn loop (turn1: prompt->kernel; harness returns compile+allclose-diff as observation; turn2: rev (residual 2)
- GENERALIZATION-MOAT OVERCLAIM + CTO 'JUST KERNELGYM + SHAPE SAMPLER' — held-out {384,768,1536} are tile-aligned, near tr → (1) Add ≥1 structurally-disjoint held-out regime: a shape that crosses a tiling boundary (e.g. M=3072 where optimal block flips) plus an odd-prime off-grid point (383/769) provably absent from code corpora; keep _assert_split_disjoint at im (residual 3)
- DAVINCI SKILL-INJECTION CARGO-CULT — static skills_v0.jsonl prefix cites Ablation 1 (44.8%->20.6%), but that gain is on  → Conditioned on the now-UNCONDITIONAL SFT cold-start, include ~10-20 skill-conditioned examples (skill-block -> kernel) in the SFT mix so injection becomes load-bearing. If SFT slips, DROP the skill library entirely and present it as 'daVinc (residual 3)
- TRLOO vs GRPO MISLABEL — plan cites daVinci TRLOO/Eq2 LOO advantages but ships trl GRPO (mean/std group-norm). Claiming  → Rename the RL section and slide text: 'We use trl GRPO (group normalization); daVinci TRLOO is the stretch target.' Remove all 'LOO advantage' pseudocode/labels unless the Eq2 subtraction is actually implemented (it is not). No correctness/ (residual 1)
- LIVE HERO-KERNEL DEMO COLD-START — novel trained kernel cold-compiles 35-130s; Modal spin-up + conference WiFi blows the → (1) Pre-warm: run hero_run.sh once OFF-SCREEN to populate TRITON_CACHE for that exact source, THEN run live -> cache-warm 3-8s. (2) Raise the live cut threshold to 60s and keep hero_run.mov (two takes: 3x-speed + 1x reveal) as the default;  (residual 2)
- HUD SDK PACKAGE-NAME / @env.template TYPED-PARAM CRASH (baseline build risk) — wrong `hud` vs `hud-sdk` package silently → Hour 0, before anything else: copy pyproject.toml + uv.lock VERBATIM from starters/verilog-template (has the correct package name + pinned SDK commit + working dep set); do not recreate. Verify `from hud import Environment` succeeds INSIDE  (residual 1)

**Protean — HUD x YC Frontier RL Environments Hackathon, 24h solo build (Sat 12:30 → Sun 13:00)** (hrs now: 24)
- REWARD-FORMULA DIVERGENCE / HASH PREFLIGHT UNENFORCEABLE (SEV5, worst). VERIFIED against scaffold: grader.py:13-20 loads → KILL the two-file architecture. There is exactly ONE reward authority: the hidden grade.py (baked, root:700). rewards.py becomes a thin module that ALSO lives at /donotaccess and is imported by grade.py ONLY. The trainer NEVER imports rewar (residual 2)
- THROUGHPUT / OVERNIGHT STEP COUNT below signal-emergence threshold (SEV5). Plan's own math: 40-100 steps; daVinci gains  → Three stacked changes. (1) MOVE THE KICK EARLIER, hard. New rule: calibration must clear by Sat 21:00 (run it IN PARALLEL with red-team during hours 4-6, not serially after — they touch different code: red-team hits anti_hack.py + grade.py  (residual 3)
- CALIBRATION GATE FAILURE / GRADIENT COLLAPSE at base 7B without SFT (SEV4). Stage B (>=1 nonzero-speedup rollout in >=2  → BUILD SFT WARM-START FRIDAY NIGHT, BEFORE THE HACKATHON CLOCK, and make it UNCONDITIONAL. Concrete: download hkust-nlp/drkernel-coldstart-8k (MIT, 8920 rows, verified), filter final_speedup>=1.2 AND messages contains a compiling Triton kern (residual 2)
- PR GATE AS MULTIPLICATIVE BINARY ZERO destroys training signal (SEV4, raised in 3 separate attacks). reward=0 if pr_frac → Stage the PR term and relabel it honestly. PHASE 1 (steps 0-50, the learning phase): PR is an ADDITIVE bonus matching daVinci Eq1's continuity: reward = CORRECT_FLOOR + speedup_score + PR_BONUS*clip(pr_frac,0,1), PR_BONUS=0.2. The HARD zero (residual 2)
- LAUNCHCOUNTER MONKEYPATCH FRAGILITY + CACHED-PATH SILENT FAILURE (SEV4, two attacks). The triton.runtime internal hook c → Validate the hook in the FIRST H100 smoke test at Hour 0 (Gate-0), not later in test_anti_hack.py. Five assertions, INSIDE the Modal container at image-build (build FAILS, not warns, on any failure): (1) known-good elementwise kernel -> lau (residual 2)
- HUD SDK PACKAGE-NAME / pyproject MISMATCH at deploy (SEV3) + @env.template typed-param crash. 'pip install hud' vs 'hud- → Do NOT author pyproject.toml or env.py boilerplate from scratch. Hour 0: `cp` pyproject.toml AND uv.lock verbatim from starters/verilog-template/ — they carry the correct package name, the pinned hud SDK, and the dep set proven to work with (residual 1)
- GENERALIZATION-MOAT OVERSELL + 'just KernelGYM + a shape sampler' CTO kill shot (SEV3-4, multiple attacks). Test shapes  → Two changes. (1) SHARPEN THE SPLIT: add at least one held-out shape that crosses a tile-count boundary where the optimal Triton BLOCK changes (e.g. M=3072 if train tops out at 2048), and include an odd/prime off-grid dim (383, 769) provably (residual 3)
- MULTI-TURN environment_factory on the critical path + refactor risk to frozen rewards.py (SEV3, two attacks). trl's mult → REMOVE multi-turn from the critical path entirely. Single-turn GRPO (yield prompt -> one completion -> grade via reward_funcs, which trl supports rock-solid) is the ONLY shipped path. Delete the environment_factory validation block from the (residual 2)
- SKILL-LIBRARY-LITE DELIVERS ZERO VALUE on an un-SFT'd-for-skills base + Ablation-1 citation misapplied (SEV3, two attack → Since SFT warm-start is now mandatory (revision 3), fold ~20 skill-conditioned examples into the SFT mix (skill markdown -> kernel that uses it) so the model actually learns to consume injected skills — this makes the static BM25 prefix gen (residual 3)
- DEMO HERO KERNEL MAY NOT EXIST AT 07:00 + LIVE MODAL COLD-START 25-90s on conference wifi (SEV3-4, two attacks). A held- → (1) PRE-WARM, never first-shot-live: in the Modal container before the demo begins, run hero_run.sh ONCE off-screen to populate TRITON_CACHE for that exact source, THEN run it live (cache-warm, 3-8s). Raise the fallback threshold to 60s. Re (residual 3)
- TRLOO vs GRPO MISLABEL (SEV3). Plan cites daVinci TRLOO/LOO advantages but ships trl GRPOTrainer (group mean/std normali → Rename the RL section and slide text: 'We use trl GRPO (group-relative, mean/std normalization), not daVinci TRLOO. TRLOO + per-agent LOO (Eq7-9) is the deferred multi-turn/multi-agent stretch.' Remove all pseudocode/text presenting the adv (residual 1)
- DOLLAR-CEILING / RUNAWAY CONTAINER. Earlier 21:30 kick + Friday SFT + pre-warm runs increase total GPU hours; a forgotte → Recompute and cap. Friday SFT ~$8, hackathon spine ~$33, earlier 14h overnight run ~$55 (14h*$3.95), eval+demo+pre-warm ~$10 => ~$106 total, well under the $200 spendable ceiling ($50 reserve). Keep the CostAbortCallback (rate=3.95, ceiling (residual 1)