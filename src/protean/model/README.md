# Protean Model Layer

This folder is the single home for model-related hackathon files.

The hackathon goal is not just "benchmark one kernel." The goal is to show a coding agent that becomes a better GPU-kernel optimizer: it edits the current best kernel, learns which edits and harness settings produce real held-out speedup, and eventually improves its own RL layer and evaluation harness.

## Files

- `policy.py`: proposes kernel edits. Today it is deterministic; next it becomes model-backed.
- `rl_layer.py`: scores candidates and decides whether an edit becomes the new best.
- `harness.py`: owns benchmark knobs the agent can tune later.
- `fireworks_policy.py`: optional Fireworks `gpt-oss-120b` kernel-edit backend.
- `tiny_policy.py`: v1 1M-parameter learned policy head trained from verifier traces.
- `prompts/kernel_optimizer.md`: prompt contract for the future model-backed coding agent.
- `configs/local_deterministic.json`: current no-model policy config.
- `configs/tiny_policy_head.json`: config/spec for the live-demo learned layer.

## V1 Learned Layer

Run an optimizer trace, train the 1M policy head, then run the optimizer with the learned policy:

```bash
python scripts/run_optimizer.py --max-rounds 1
python scripts/train_tiny_policy.py --trace runs/protean-overnight/trials.jsonl
python scripts/run_optimizer.py --max-rounds 1 --edit-policy learned --policy-path runs/protean-overnight/tiny_policy.json
```

Use Fireworks for a model-generated edit:

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --max-rounds 1 --edit-policy fireworks
```

The default small head has 1,000,005 trainable parameters over 10 state features. It is intentionally much smaller than the later 100M controller; the point is to prove that verifier traces can train a policy that changes the coding agent's edit behavior.

## Boundary

Keep model-agent code here. The rest of Protean should call this layer instead of scattering prompt, RL, harness, and model-policy logic across the repo.
