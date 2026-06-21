# Protean Lean Implementation Plan

## Rule

Keep the repo small. Add one layer only after the previous layer works on Spark.

## Layer 1: Verifier

- Grade one op: `elementwise_add_relu`.
- Compare candidate Triton against PyTorch eager.
- Return structured reward with correctness, speedup, timing, split, and caps.
- Reject obvious hacks: PyTorch passthrough, no `@triton.jit`, dtype mismatch, shape mismatch.

## Layer 2: Demo

- Run train and held-out shapes.
- Produce `demo/protean-demo-results.md` and `.json`.
- Accept v1 when at least one held-out row is correct and faster than PyTorch eager.

## Layer 3: Add More

Only after Layer 1 and 2 pass on Spark:

1. Add `rmsnorm`.
2. Add HUD remote packaging.
3. Add GRPO training.

Use `docs/papers/davinci-kernel-2606.16497.llm.txt` before the PDF when adding training features. The paper is context, not v1 scope.

## Current Target

Run on `ssh spark`, an NVIDIA GB10 host. Spark must have a user-local Python environment with PyTorch, Triton, and pytest.
