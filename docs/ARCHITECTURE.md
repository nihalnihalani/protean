# Protean Architecture

Protean is intentionally small, but the product is an optimizer, not just a grader. The v1 architecture starts from a working kernel, edits the current best implementation, grades each candidate, accepts improvements, and logs the full trace.

## Flow

```text
candidate source
  -> static anti-hack checks
  -> CUDA benchmark
  -> structured reward
  -> accept/reject as current best
  -> trial log + best kernel
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

- `src/protean/optimizer.py`
  - Runs the improvement loop.
  - Starts from the current best kernel.
  - Calls the model-agent layer for edits and acceptance policy.
  - Evaluates candidates, accepts improvements, and writes logs.
  - Saves every candidate source file and records score deltas against the current best.

- `src/protean/model/`
  - Single home for model-related hackathon files.
  - `policy.py` proposes kernel edits.
  - `rl_layer.py` scores and accepts candidates.
  - `harness.py` owns benchmark knobs the agent can tune.
  - `fireworks_policy.py` asks Fireworks `gpt-oss-120b` for model-generated kernel edits.
  - `tiny_policy.py` trains the v1 1M-parameter learned policy head from verifier traces.
  - `prompts/` and `configs/` hold the model prompt contract and active policy config.
  - The current edit policy is deterministic; a model-backed policy should replace it next.

## Scripts

- `scripts/check_redteam.py`
  - Verifies passthrough, no-JIT, and bad-shape submissions score zero.

- `scripts/smoke_verifier.py`
  - Runs the known-good Triton kernel on one held-out shape.

- `scripts/run_demo_benchmark.py`
  - Runs train and held-out shapes.
  - Writes `demo/protean-demo-results.md` and `.json`.

- `scripts/run_optimizer.py`
  - Runs the iterative optimizer.
  - Writes `runs/protean-overnight/best_kernel.py`, `trials.jsonl`, and `summary.json`.

- `scripts/train_tiny_policy.py`
  - Trains the v1 1M-parameter policy head from `trials.jsonl`.
  - Writes `runs/protean-overnight/tiny_policy.json`.

## Spark Runtime

The verified GPU target is `ssh spark`.

```bash
cd /home/alhinai/protean
. /home/alhinai/.venvs/protean/bin/activate
python -m pytest -q
python scripts/check_redteam.py
python scripts/smoke_verifier.py
python scripts/run_demo_benchmark.py
python scripts/run_optimizer.py --max-rounds 1
python scripts/train_tiny_policy.py --trace runs/protean-overnight/trials.jsonl
python scripts/run_optimizer.py --max-rounds 1 --edit-policy learned --policy-path runs/protean-overnight/tiny_policy.json
python scripts/run_optimizer.py --max-rounds 1 --edit-policy fireworks
```

Spark has already produced a real held-out delta for `elementwise_add_relu`.

## Additions Later

Add features in this order only:

1. Use the 1M policy head as the default learned edit policy after enough traces exist.
2. Model-backed kernel edit policy in `src/protean/model/policy.py`.
3. Self-improvement policy for `src/protean/model/rl_layer.py` and `src/protean/model/harness.py`.
4. Expand optimizer edit policies beyond `elementwise_add_relu`.
5. Profiler-based runtime attribution if needed.
6. HUD remote packaging.
7. GRPO training.

Do not add training code before the optimizer loop is stable on at least two ops.
