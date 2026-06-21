# Implementation Plan

Protean is built in layers. Each layer must be demonstrable before the next one matters.

## Current North Star

> Throw a GPU kernel at Protean, let it run overnight, and wake up with a faster correct kernel plus a complete trace of every edit, benchmark, failure, token, and decision.

For the hackathon, the proof is smaller:

1. Show a trustworthy verifier.
2. Show non-zero HUD reward on real GPU kernels.
3. Show an optimizer loop that saves and logs every candidate.
4. Use HUD as the eval/training control plane for optimizer runs.
5. Show the path for Fireworks/model-backed edits.

## Layer 1: Verifier

Status: implemented and verified.

- Three public ops: `elementwise_add_relu`, `rmsnorm`, `softmax_rows`.
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
python scripts/smoke_verifier.py --op softmax_rows
```

## Layer 2: HUD Proof

Status: implemented and verified.

The HUD wrapper exposes six tasks and calls the direct Protean grader. The deterministic demo agent submits known-good kernels so the dashboard shows the verifier working, not the randomness of a weak one-step generic model.

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
- Optionally streams every trial candidate into one HUD job/session.

Acceptance:

```bash
python scripts/run_optimizer.py --all-ops --max-rounds 1
```

HUD streaming acceptance:

```bash
HUD_API_KEY=... python scripts/run_optimizer.py --op elementwise_add_relu --max-rounds 1 --stream-hud --hud-job-name protean-smoke
```

This opens one HUD job, appends each trial candidate under that job, and records either `hud_stream.job_url` or `hud_stream_error` per trial. HUD failures do not kill the optimizer.

## Layer 3.5: HUD Control Plane

Status: implemented.

- Stable task rows expose the Protean benchmark to HUD.
- One optimizer run maps to one HUD job.
- Each candidate creates HUD traces for train and held-out tasks.
- Trace steps show model response, 1M controller decision when available, saved candidate, AST check, compile status, correctness, timing, reward, and accept/reject.
- HUD rewards/subscores are normalized to `0..1`.
- Raw Protean reward remains in `info.protean_reward_raw` and `trials.jsonl`.
- `--hud-group N` repeats each task to measure reward spread for trainability.

Commands:

```bash
hud deploy . --no-env
hud sync tasks protean-kernel-optimizer src/protean/env.py --yes
hud eval protean-kernel-optimizer claude --full --group 3 --max-concurrent 4
python scripts/run_optimizer.py --op elementwise_add_relu --max-rounds 1 --stream-hud --hud-group 3
python scripts/run_hud_optimizer_agent.py --policy local --all-ops --max-rounds 1 --group 2 --job-name protean-live-fallback
```

Use `hud set HUD_API_KEY=...` or `export HUD_API_KEY=...`; do not source the project `.env`.

Verified platform artifacts:

- requested environment: https://hud.ai/environments/9907b272-ef58-4f57-9cd3-5dbcb37dd51e
- requested taskset: https://hud.ai/tasksets/6d2feb10-b23c-4928-a1f9-e8b53db364d7
- active verified environment: https://hud.ai/environments/32bb1f0c-0737-4a58-8a5e-5c9ec8a2f01b
- active verified taskset: https://hud.ai/tasksets/3f2d2423-72d4-4541-bb18-b78e31151676
- active all-ops grouped live job: https://hud.ai/jobs/3eda0cb665df40f6a3f25a89460819ae

## Layer 4: Model-Backed Edits

Status: wired, requires `FIREWORKS_API_KEY` for new runs.

Fireworks path:

- Uses OpenAI-compatible chat completions.

## Layer 5: GRPO Stretch Controls

Status: scaffolded and test-covered; not required for the guaranteed demo.

- Frozen manifest: `manifest_v1.jsonl` is committed and SHA256-pinned.
- Curriculum: early GRPO steps sample a prefix of train shapes only; held-out shapes are never mixed into training.
- Calibration: optional real rollout calibration checks compile rate, allclose rate, reward spread, and speedup before training.
- Safeguards: step-75 warning, step-150 hard abort, cost/time limits, and 30-step flatline detection.
- Curve logging: `RewardCurveLogger` writes `outputs/train_history.json`; `scripts/plot_curve.py` plots that real file.
- Requests strict JSON with `response_format={"type": "json_object"}`.
- Uses `reasoning_effort="low"`.
- Falls back from `content` to `reasoning_content` for `gpt-oss`.
- Tracks token count.
- Produces the same `CandidateEdit` record as local policies.

Next run:

```bash
export FIREWORKS_API_KEY=...
export HUD_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --stream-hud --hud-job-name protean-fireworks-overnight --out-dir runs/protean-fireworks-overnight
python scripts/run_hud_optimizer_agent.py --policy fireworks --controller outputs/policy_head.pt --all-ops --max-rounds 50 --group 4 --job-name protean-live-kernel-optimizer
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
6. Keep `demo/powered-eval-200.json` current as the statistical held-out moat artifact.

## Do Not Do Yet

- Do not make GRPO required for the demo.
- Do not claim trained model superiority without a real held-out benchmark.
- Do not add unrelated abstractions before Fireworks overnight traces exist.
- Do not create a second HUD-specific grader.
