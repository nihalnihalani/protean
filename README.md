# Protean

Small overnight GPU-kernel optimizer.

## Goal

Protean's goal is to build a coding agent that gets better at optimizing GPU kernels by trying edits, benchmarking them, learning from the results, and improving its next attempts.

You give Protean a GPU kernel. Protean keeps modifying it, tests every version for correctness and speed, rejects bad versions, keeps better versions, and logs every trial, speedup, failure, cost, and decision.

For the hackathon, the goal is deliberately lean:

1. A coding agent proposes kernel edits.
2. A verifier grades correctness, speed, held-out shape behavior, and anti-hack checks.
3. A 1M-parameter learned policy head trains from those verifier traces.
4. The learned policy changes which edits the agent tries next.

Long term, Protean should run overnight and wake you up with a faster, correct, well-tested kernel plus a full audit trail.

## Promise

> Start from a working GPU kernel, keep editing the current best version overnight, and wake up with the fastest correct kernel plus a full optimization trace.

The verifier is the measurement core. The product is the improvement loop around it.

## What Works Now

- One op: `elementwise_add_relu`, equivalent to `torch.relu(x + y)`.
- Second op: `rmsnorm`, equivalent to vector RMS normalization with learned weight.
- Train shapes: `1024`, `2048`, `4096`.
- Held-out shapes: `1536`, `3072`, `5632`.
- Static anti-hack checks for PyTorch passthrough and no-`@triton.jit` submissions.
- GPU grader that checks correctness, dtype, shape, `@triton.jit` usage, and CUDA timing.
- Demo scripts that output JSON and Markdown benchmark artifacts.
- Iterative optimizer that edits the current best kernel, evaluates each candidate, accepts improvements, and logs every trial.
- One model-agent folder, `src/protean/model/`, for the edit policy, RL layer, harness policy, prompts, and model config.
- HUD dashboard proof with a deterministic Protean demo agent that submits known-good kernels through the same HUD grader.

## Install

Core package and tests:

```bash
python -m pip install -e ".[test]"
```

GPU verifier dependencies:

```bash
python -m pip install -e ".[gpu,test]"
```

HUD integration is optional:

```bash
python -m pip install -e ".[hud]"
```

## Run

CPU-safe checks:

```bash
python -m pytest -q
python scripts/check_redteam.py
```

GPU smoke test:

```bash
python scripts/smoke_verifier.py
```

Generate the demo benchmark:

```bash
python scripts/run_demo_benchmark.py
```

Run the optimizer loop:

```bash
python scripts/run_optimizer.py --max-rounds 1
```

Train the v1 learned controller from the optimizer trace:

```bash
python scripts/train_tiny_policy.py --trace runs/protean-overnight/trials.jsonl
python scripts/run_optimizer.py --max-rounds 1 --edit-policy learned --policy-path runs/protean-overnight/tiny_policy.json
```

Ask Fireworks `gpt-oss-120b` for a model-generated kernel edit:

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --max-rounds 1 --edit-policy fireworks
```

Run the HUD task wrapper:

```bash
python -m protean.env
python scripts/smoke_verifier.py --op elementwise_add_relu
python scripts/smoke_verifier.py --op rmsnorm
PYTHONPATH=src hud task list --source src/protean/env.py
HUD_API_KEY=... PYTHONPATH=src python scripts/run_hud_demo_agent.py
```

HUD exposes four task ids:

- `elementwise_add_relu_train`
- `elementwise_add_relu_held_out`
- `rmsnorm_train`
- `rmsnorm_held_out`

### HUD Dashboard Results

Protean's demo agent (deterministic, submits known-good kernels) creates a real
job on the HUD platform with non-zero rewards:

- **Passing demo job**: https://hud.ai/jobs/8a8c3bfcf5904b9f8181ff13f3f309a7
- **Integration job** (Fireworks openai_compatible): https://hud.ai/jobs/3af4548f0afe4f449b5245a2809ae0e0

Per-task results (Spark GB10, hand-optimized Triton kernels):

| Task                          | Reward | Correct | Speedup  | Caps |
|-------------------------------|--------|---------|----------|------|
| elementwise_add_relu_train    | 1.300  | True    | 1.55x    | []   |
| elementwise_add_relu_held_out | 1.300  | True    | 2.08x    | []   |
| rmsnorm_train                 | 1.300  | True    | 6.38x    | []   |
| rmsnorm_held_out              | 1.300  | True    | 6.87x    | []   |

Run the demo agent yourself:

```bash
export HUD_API_KEY=...
python scripts/run_hud_demo_agent.py
```

### Fireworks gpt-oss-120b Integration

Protean's optimizer supports Fireworks as a model-backed edit policy. The model
receives the current best kernel and proposes improvements as JSON.

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 5
```

Verified live: gpt-oss-120b produces valid Triton kernel edits with
`@triton.jit` and `solution()` on both ops, with real token counts and cost
tracking. The crash-safe optimizer logs compile errors as rejected trials
instead of aborting the run.

Outputs:

- `demo/protean-demo-results.md`
- `demo/protean-demo-results.json`
- `demo/hud-demo-agent-results.json`
- `runs/protean-overnight/best_kernel.py`
- `runs/protean-overnight/trials.jsonl`
- `runs/protean-overnight/summary.json`
- `runs/protean-overnight/tiny_policy.json`

Each optimizer trial logs the implementation path, edit reason, harness settings, score before/after, delta versus the current best, acceptance decision, elapsed time, and model cost. The current deterministic policy has `model_cost_usd: 0.0`; model-backed edits should fill that field later.

If CUDA, PyTorch, or Triton are missing, GPU scripts fail closed with `cuda_unavailable`.

## Verified Artifacts

Latest Spark + HUD run:

- HUD integration job with generic one-step agent: https://hud.ai/jobs/22314aa9438c4d41bb98edeadea29913
- HUD passing demo job with Protean demo agent: https://hud.ai/jobs/813e572399c842c78d5a515f7644b4ae
- HUD passing demo artifact: `demo/hud-demo-agent-results.json`

Passing HUD demo results from Spark GB10:

| Task | Split | Shape | Reward | Correct | Speedup |
|---|---|---:|---:|---|---:|
| `elementwise_add_relu` | held-out | 1536 | 1.3 | true | 2.08x |
| `elementwise_add_relu` | train | 1024 | 1.3 | true | 1.55x |
| `rmsnorm` | held-out | 1536 | 1.3 | true | 6.87x |
| `rmsnorm` | train | 1024 | 1.3 | true | 6.38x |

Spark optimizer preflight before the HUD run:

| Op | Best score | Accepted |
|---|---:|---:|
| `elementwise_add_relu` | `(1.998708, 1.294553, 3)` | 1/5 |
| `rmsnorm` | `(8.730407, 1.3, 3)` | 0/5 |

## Reward Output

The grader returns a plain dict:

```json
{
  "reward": 1.3,
  "correct": true,
  "speedup": 1.62,
  "t_eager_ms": 0.018,
  "t_kernel_ms": 0.011,
  "split": "held_out",
  "caps": []
}
```

Hard failures get zero reward. Slow-but-correct kernels can report correctness, but do not earn speedup reward below the floor.

## Add Next

1. Use the 1M policy head as the default learned edit-ordering policy after enough traces exist.
2. Let the model improve `src/protean/model/rl_layer.py` and `src/protean/model/harness.py`, with every change logged.
3. Make Fireworks-generated edits robust enough to run unattended overnight.
4. Add training/RL only after the optimizer loop is stable on two ops.

KERNEL-FORGE notes live in `docs/strategy/KERNEL_FORGE_AUDIT.md`; they are background, not the build path.
The daVinci-kernel paper is included under `docs/papers/` with `davinci-kernel-2606.16497.llm.txt` as the LLM-first reading companion.
For implementation structure, read `docs/ARCHITECTURE.md`.
