# Protean

Small verifier-first GPU-kernel RL environment.

Protean starts with one promise:

> Produce one credible PyTorch-eager vs hand-written Triton delta on shapes the kernel did not see during development.

No GRPO is required for v1. Training can be added after the verifier is trustworthy.

## What Works Now

- One op: `elementwise_add_relu`, equivalent to `torch.relu(x + y)`.
- Train shapes: `1024`, `2048`, `4096`.
- Held-out shapes: `1536`, `3072`, `5632`.
- Static anti-hack checks for PyTorch passthrough and no-`@triton.jit` submissions.
- GPU grader that checks correctness, dtype, shape, `@triton.jit` usage, and CUDA timing.
- Demo scripts that output JSON and Markdown benchmark artifacts.

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

Outputs:

- `demo/protean-demo-results.md`
- `demo/protean-demo-results.json`

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

1. Run and record the benchmark on `ssh spark`.
2. Add `rmsnorm` only after `elementwise_add_relu` is green.
3. Add training only after two ops are stable.

KERNEL-FORGE notes live in `docs/strategy/KERNEL_FORGE_AUDIT.md`; they are background, not the build path.
The daVinci-kernel paper is included under `docs/papers/` with `davinci-kernel-2606.16497.llm.txt` as the LLM-first reading companion.
For implementation structure, read `docs/ARCHITECTURE.md`.
