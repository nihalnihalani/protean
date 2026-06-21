# Protean Architecture

Protean is intentionally small. The v1 architecture has one job: grade a candidate Triton kernel against PyTorch eager on train and held-out shapes, then emit a plain reward report.

## Flow

```text
candidate source
  -> static anti-hack checks
  -> CUDA benchmark
  -> structured reward
  -> demo JSON/Markdown
```

## Components

- `src/protean/grader.py`
  - Main entrypoint: `grade_source(...)`.
  - Runs static checks first.
  - Calls the CUDA benchmark only when the candidate passes the cheap gates.

- `src/protean/anti_hack.py`
  - Rejects obvious PyTorch passthrough calls such as `torch.relu`.
  - Requires a real `@triton.jit` function.
  - Keeps v1 simple; deeper profiling can be added later.

- `src/protean/bench_core.py`
  - Imports the candidate from a real temporary `.py` file because Triton 3.7 requires JIT functions to be file-backed.
  - Runs correctness on fresh random inputs.
  - Times PyTorch eager and candidate Triton with CUDA events.
  - Re-checks correctness after timing on a separate seed.

- `src/protean/rewards.py`
  - Builds the public reward payload.
  - Hard-fails incorrect output, dtype mismatch, shape mismatch, and missing JIT usage.
  - Reports slow-but-correct kernels separately from fast kernels.

- `src/protean/splits.py`
  - Freezes the generalization split.
  - Train shapes: `1024`, `2048`, `4096`.
  - Held-out shapes: `1536`, `3072`, `5632`.

- `src/protean/kernels.py`
  - Known-good hand Triton kernel.
  - Red-team examples used by tests and scripts.

## Scripts

- `scripts/check_redteam.py`
  - Verifies passthrough, no-JIT, and bad-shape submissions score zero.

- `scripts/smoke_verifier.py`
  - Runs the known-good Triton kernel on one held-out shape.

- `scripts/run_demo_benchmark.py`
  - Runs train and held-out shapes.
  - Writes `demo/protean-demo-results.md` and `.json`.

## Spark Runtime

The verified GPU target is `ssh spark`.

```bash
cd /home/alhinai/protean
. /home/alhinai/.venvs/protean/bin/activate
python -m pytest -q
python scripts/check_redteam.py
python scripts/smoke_verifier.py
python scripts/run_demo_benchmark.py
```

Spark has already produced a real held-out delta for `elementwise_add_relu`.

## Additions Later

Add features in this order only:

1. `rmsnorm` as the second op.
2. Profiler-based runtime attribution if needed.
3. HUD remote packaging.
4. GRPO training.

Do not add training code before the verifier has at least two stable ops.
