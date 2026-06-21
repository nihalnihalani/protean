# KERNEL-FORGE Audit Lineage

KERNEL-FORGE was the internal research plan. Protean is the cleaned-up public implementation.

## What Protean Keeps

- The held-out shape generalization moat.
- The verifier-first product story.
- The anti-hack framing: correctness, real Triton launch, dtype/shape integrity, and measured speed.
- The Modal/HUD sponsor alignment.

## What Protean Cuts From The Critical Path

- Overnight GRPO as a required success condition.
- Multi-op graph generation.
- Multi-agent daVinci-style skill selection and summarization.
- Large claims about being first or complete relative to every kernel-RL system.

## Best-Of-Both-Worlds Decision

Protean uses the KERNEL-FORGE adversarial analysis, but narrows the build to the artifact judges can inspect:

1. A working verifier.
2. A train-vs-held-out shape split.
3. A PyTorch eager vs hand-optimized Triton delta.
4. Red-team examples that score zero.

If training succeeds later, it strengthens the story. It is not required for the v1 demo to be credible.
