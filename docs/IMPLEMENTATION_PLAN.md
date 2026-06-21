# Protean Lean Implementation Plan

## Rule

Keep the repo small. The goal is an overnight optimizer, but each layer must work on Spark before adding the next layer.

## Layer 1: Verifier

- Grade one op: `elementwise_add_relu`.
- Compare candidate Triton against PyTorch eager.
- Return structured reward with correctness, speedup, timing, split, and caps.
- Reject obvious hacks: PyTorch passthrough, no `@triton.jit`, dtype mismatch, shape mismatch.

## Layer 2: Demo

- Run train and held-out shapes.
- Produce `demo/protean-demo-results.md` and `.json`.
- Accept v1 when at least one held-out row is correct and faster than PyTorch eager.

## Layer 3: Iterative Optimizer

- Start from the current best kernel.
- Generate candidate edits by modifying that kernel.
- Grade each candidate.
- Accept only candidates that improve held-out speed/reward.
- Log each trial as JSONL with candidate source path, edit reason, harness settings, score, delta versus the current best, acceptance, elapsed time, model cost, and per-shape results.
- Write the best kernel to `runs/protean-overnight/best_kernel.py`.

## Layer 4: Add More

Only after Layer 1 and 2 pass on Spark:

1. Replace deterministic edits with a model-backed edit policy.
2. Let the model propose harness and reward mutations.
3. Add `rmsnorm`.
4. Add HUD remote packaging.
5. Add GRPO training.

Use `docs/papers/davinci-kernel-2606.16497.llm.txt` before the PDF when adding training features. The paper is context, not v1 scope.

## Current Target

Run on `ssh spark`, an NVIDIA GB10 host. Spark must have a user-local Python environment with PyTorch, Triton, and pytest.
