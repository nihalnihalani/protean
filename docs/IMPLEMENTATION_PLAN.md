# Implementation Plan

Protean is built in layers. Each layer must be demonstrable before the next one matters.

## Current North Star

> Throw a GPU kernel at Protean, let it run overnight, and wake up with a faster correct kernel plus a complete trace of every edit, benchmark, failure, token, and decision.

For the hackathon, the proof is smaller:

1. Show a trustworthy verifier.
2. Show non-zero HUD reward on real GPU kernels.
3. Show an optimizer loop that saves and logs every candidate.
4. Show the path for Fireworks/model-backed edits.

## Layer 1: Verifier

Status: implemented and verified.

- Two ops: `elementwise_add_relu`, `rmsnorm`.
- PyTorch eager reference for each op.
- Known-good hand-written Triton implementation for each op.
- Fresh random inputs for correctness.
- CUDA-event timing with warmup and median timing.
- Structured reward JSON.
- Static anti-hack checks.
- Held-out shapes disjoint from train shapes.

Acceptance:

```bash
python -m pytest -q
python scripts/check_redteam.py
python scripts/smoke_verifier.py --op elementwise_add_relu
python scripts/smoke_verifier.py --op rmsnorm
```

## Layer 2: HUD Proof

Status: implemented and verified.

The HUD wrapper exposes four tasks and calls the direct Protean grader. The deterministic demo agent submits known-good kernels so the dashboard shows the verifier working, not the randomness of a weak one-step generic model.

Passing job:

https://hud.ai/jobs/5a3ddc3f24a748d9abda38866bccb503

Acceptance:

```bash
HUD_API_KEY=... PYTHONPATH=src python scripts/run_hud_demo_agent.py
```

## Layer 3: Optimizer Loop

Status: implemented and verified.

- Starts from current best kernel.
- Generates candidate edits.
- Writes every candidate to disk.
- Evaluates train and held-out shapes.
- Scores by held-out speed first, reward second, held-out correctness count third.
- Accepts only strict improvements.
- Logs every trial to JSONL.
- Converts verifier crashes into rejected trial records.

Acceptance:

```bash
python scripts/run_optimizer.py --all-ops --max-rounds 1
```

## Layer 4: Model-Backed Edits

Status: wired, requires `FIREWORKS_API_KEY` for new runs.

Fireworks path:

- Uses OpenAI-compatible chat completions.
- Requests strict JSON with `response_format={"type": "json_object"}`.
- Uses `reasoning_effort="low"`.
- Falls back from `content` to `reasoning_content` for `gpt-oss`.
- Tracks token count.
- Produces the same `CandidateEdit` record as local policies.

Next run:

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --out-dir runs/protean-fireworks-overnight
```

## Layer 5: Learned Policy Head

Status: implemented as v1 1M-parameter controller.

Purpose:

- Learn which edit to try next from verifier traces.
- Reorder deterministic edit actions.
- Stay small enough to train quickly during the hackathon.

Commands:

```bash
python scripts/train_tiny_policy.py --trace runs/protean-overnight/trials.jsonl
python scripts/run_optimizer.py --edit-policy learned --policy-path runs/protean-overnight/tiny_policy.json
```

## Stretch

Only after the above is stable:

1. Let the model edit `src/protean/model/harness.py`.
2. Let the model propose changes to `src/protean/model/rl_layer.py`.
3. Add more ops.
4. Add GRPO/LoRA training.
5. Compare deterministic, Fireworks, learned-policy, and trained-agent curves.

## Do Not Do Yet

- Do not make GRPO required for the demo.
- Do not claim trained model superiority without a real held-out benchmark.
- Do not add unrelated abstractions before Fireworks overnight traces exist.
- Do not create a second HUD-specific grader.
