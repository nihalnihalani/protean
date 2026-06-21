# Protean Model Layer

This folder is the single home for model-related hackathon files.

The hackathon goal is not just "benchmark one kernel." The goal is to show a coding agent that becomes a better GPU-kernel optimizer: it edits the current best kernel, learns which edits and harness settings produce real held-out speedup, and eventually improves its own RL layer and evaluation harness.

## Files

- `policy.py`: proposes kernel edits. Today it is deterministic; next it becomes model-backed.
- `rl_layer.py`: scores candidates and decides whether an edit becomes the new best.
- `harness.py`: owns benchmark knobs the agent can tune later.
- `prompts/kernel_optimizer.md`: prompt contract for the future model-backed coding agent.
- `configs/local_deterministic.json`: current no-model policy config.

## Boundary

Keep model-agent code here. The rest of Protean should call this layer instead of scattering prompt, RL, harness, and model-policy logic across the repo.
