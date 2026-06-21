# Protean

Small overnight GPU-kernel optimizer.

Protean starts with one promise:

> Start from a working GPU kernel, keep editing the current best version overnight, and wake up with the fastest correct kernel plus a full optimization trace.

The verifier is the measurement core. The product is the improvement loop around it.

## What Works Now

- One op: `elementwise_add_relu`, equivalent to `torch.relu(x + y)`.
- Train shapes: `1024`, `2048`, `4096`.
- Held-out shapes: `1536`, `3072`, `5632`.
- Static anti-hack checks for PyTorch passthrough and no-`@triton.jit` submissions.
- GPU grader that checks correctness, dtype, shape, `@triton.jit` usage, and CUDA timing.
- Demo scripts that output JSON and Markdown benchmark artifacts.
- Iterative optimizer that edits the current best kernel, evaluates each candidate, accepts improvements, and logs every trial.

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

Outputs:

- `demo/protean-demo-results.md`
- `demo/protean-demo-results.json`
- `runs/protean-overnight/best_kernel.py`
- `runs/protean-overnight/trials.jsonl`
- `runs/protean-overnight/summary.json`

Each optimizer trial logs the implementation path, edit reason, harness settings, score before/after, delta versus the current best, acceptance decision, elapsed time, and model cost. The current deterministic policy has `model_cost_usd: 0.0`; model-backed edits should fill that field later.

If CUDA, PyTorch, or Triton are missing, GPU scripts fail closed with `cuda_unavailable`.

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

1. Replace the deterministic edit policy with a model-backed edit policy.
2. Let the model tune the harness and reward settings, with every change logged.
3. Add `rmsnorm` as the second optimization target.
4. Add training/RL only after the optimizer loop is stable on two ops.

KERNEL-FORGE notes live in `docs/strategy/KERNEL_FORGE_AUDIT.md`; they are background, not the build path.
The daVinci-kernel paper is included under `docs/papers/` with `davinci-kernel-2606.16497.llm.txt` as the LLM-first reading companion.
For implementation structure, read `docs/ARCHITECTURE.md`.
