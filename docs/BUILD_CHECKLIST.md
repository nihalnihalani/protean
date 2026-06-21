# Build Checklist

This checklist is for a judge or teammate trying to reproduce the demo quickly.

## Local CPU Checks

- [x] Package imports without CUDA.
- [x] Unit tests cover reward, split, HUD wrapper, Fireworks payload, optimizer crash logging, and tiny policy head.
- [x] Red-team examples fail closed.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/check_redteam.py
```

Expected current local result:

```text
49 passed, 1 skipped
```

## Spark GPU Checks

Verified target: `ssh spark`, Spark GB10.

- [x] PyTorch + Triton installed in `/home/alhinai/.venvs/protean`.
- [x] `elementwise_add_relu` known-good kernel passes held-out smoke.
- [x] `rmsnorm` known-good kernel passes held-out smoke.
- [x] HUD CLI and `hud-python` installed in the Protean venv.
- [x] HUD key configured.

```bash
ssh spark
cd /home/alhinai/protean
. /home/alhinai/.venvs/protean/bin/activate
python -m pytest -q
python scripts/smoke_verifier.py --op elementwise_add_relu
python scripts/smoke_verifier.py --op rmsnorm
python scripts/smoke_verifier.py --op softmax_rows
```

## HUD Dashboard Proof

- [x] HUD task discovery lists all six tasks.
- [x] Deterministic Protean demo agent creates a HUD job with non-zero reward.
- [x] Passing job: https://hud.ai/jobs/5a3ddc3f24a748d9abda38866bccb503

```bash
PYTHONPATH=src hud task list --source src/protean/env.py
HUD_API_KEY=... PYTHONPATH=src python scripts/run_hud_demo_agent.py
```

Expected output shape:

```text
mean_reward=0.894
elementwise_add_relu_held_out: reward=0.628 correct=True speedup=2.08x caps=[]
elementwise_add_relu_train:    reward=0.482 correct=True speedup=1.56x caps=[]
rmsnorm_held_out:              reward=1.255 correct=True speedup=6.97x caps=[]
rmsnorm_train:                 reward=1.211 correct=True speedup=6.40x caps=[]
```

## Optimizer Preflight

- [x] Optimizer runs all ops.
- [x] Candidates are written to `runs/.../candidates/`.
- [x] Trial records are written to `trials.jsonl`.
- [x] Compile/runtime failures are logged as rejected trials.
- [x] Optional HUD streaming records `hud_stream.job_url` or `hud_stream_error` per trial.
- [x] One optimizer run streams into one HUD job/session.
- [x] HUD trace steps include model response, candidate save, AST, compile, correctness, timing, reward, and accept/reject.
- [x] HUD trace steps include the 1M controller decision when a controller artifact is present.
- [x] HUD reward/subscores are normalized to `0..1`; raw Protean reward is in metadata.

```bash
python scripts/run_optimizer.py --all-ops --max-rounds 1 --out-dir runs/protean-hud-preflight
python scripts/run_optimizer.py --op elementwise_add_relu --max-rounds 1 --stream-hud --hud-job-name protean-hud-stream-smoke --out-dir runs/protean-hud-stream-smoke
python scripts/run_hud_optimizer_agent.py --policy local --all-ops --max-rounds 1 --group 2 --job-name protean-live-fallback
```

HUD auth should come from `hud set HUD_API_KEY=...` or `export HUD_API_KEY=...`. Do not source the project `.env`.

Verified Spark output:

```text
elementwise_add_relu  accepted=1/5
rmsnorm               accepted=0/5
```

Verified HUD control-plane smoke on Spark:

```text
job_url: https://hud.ai/jobs/53014ddee1c34b60b229e700532793d3
op: elementwise_add_relu
trials: 5
accepted: 1/5
group: 1
auth_errors: 0
held_out_correct: 3/3
mean_held_out_speedup: 1.915966
```

Verified all-ops grouped live run on Spark:

```text
job_url: https://hud.ai/jobs/3eda0cb665df40f6a3f25a89460819ae
group: 2
elementwise_add_relu: accepted 1/5
rmsnorm: accepted 0/5
softmax_rows: accepted 2/5
auth_errors: 0
```

## Fireworks Overnight Run

Requires `FIREWORKS_API_KEY`. Add `HUD_API_KEY` and `--stream-hud` when every trial should appear in the HUD dashboard.

```bash
export FIREWORKS_API_KEY=...
export HUD_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --duration-hours 8 --stream-hud --hud-job-name protean-fireworks-overnight --out-dir runs/protean-fireworks-overnight
python scripts/run_hud_optimizer_agent.py --policy fireworks --controller outputs/policy_head.pt --all-ops --max-rounds 50 --group 4 --job-name protean-live-kernel-optimizer
DURATION_HOURS=8 POLICY=fireworks HUD_GROUP=1 scripts/start_overnight_hud_optimizer.sh
```

Acceptance criteria:

- [ ] Every Fireworks candidate source is saved.
- [ ] Every trial is logged.
- [ ] Time-budgeted runs stop with `stop_reason=duration_reached` or the configured round cap.
- [ ] `run_complete` is appended to `trials.jsonl` and `improvements_<op>.jsonl`.
- [ ] Compile/runtime errors appear as `eval_error` and rejected, not lost.
- [ ] Every trial has either `hud_stream.job_url` or `hud_stream_error`.
- [x] All streamed trials share one HUD job URL for the optimizer session in local-policy smoke.
- [ ] All streamed trials share one HUD job URL for the overnight Fireworks session.
- [ ] Summary shows accepted count, best score, tokens, and model cost metadata.

## HUD Platform Taskset

Use this when the goal is a reusable HUD benchmark, not only local Spark streaming:

```bash
hud deploy . --no-env
hud sync tasks protean-kernel-optimizer src/protean/env.py --yes
hud eval protean-kernel-optimizer claude --full --group 3 --max-concurrent 4
```

Acceptance criteria:

- [x] HUD environment deploy succeeds.
- [x] Taskset appears on the HUD dashboard as `protean-kernel-optimizer`.
- [x] The expanded HUD grid is present: 12 ops x 2 splits x 42 shape variants = 1008 rows.
- [ ] `--group 3` remote eval completes without operator interruption.

Verified HUD platform artifacts:

```text
environment: https://hud.ai/environments/9907b272-ef58-4f57-9cd3-5dbcb37dd51e
taskset:     https://hud.ai/tasksets/6d2feb10-b23c-4928-a1f9-e8b53db364d7
active env:  https://hud.ai/environments/32bb1f0c-0737-4a58-8a5e-5c9ec8a2f01b
active set:  https://hud.ai/tasksets/3f2d2423-72d4-4541-bb18-b78e31151676
deploy:      image version 1, v6 control channel introspection OK
sync:        taskset source now defines 1008 rows under the active HUD key
```

## GRPO Stretch Controls

Imported from `feature/step-5-6-7-grpo-vllm-pr`, adapted to current Protean `shape` tasks:

- [x] `RewardCurveLogger` writes `outputs/train_history.json` with train and held-out reward curves.
- [x] `scripts/plot_curve.py` consumes real history data and refuses missing history instead of inventing mock curves.
- [x] Calibration supports real base-model rollout checks and `PROTEAN_SKIP_CALIBRATION=1` is the dev escape hatch.
- [x] L1 curriculum restricts early train shapes while held-out shapes always use the full moat split.
- [x] Cost abort callback includes step-75 warning, step-150 hard review, and 30-step flatline detection.
- [x] `manifest_v1.jsonl` is frozen, committed, and pinned by SHA256 in `src/protean/tasks.py`.
- [x] Dynamic manifest generation requires `PROTEAN_ALLOW_DYNAMIC_MANIFEST=1`.
