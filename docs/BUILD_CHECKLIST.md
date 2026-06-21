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
47 passed, 1 skipped
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
```

## HUD Dashboard Proof

- [x] HUD task discovery lists all four tasks.
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

```bash
python scripts/run_optimizer.py --all-ops --max-rounds 1 --out-dir runs/protean-hud-preflight
HUD_API_KEY=... python scripts/run_optimizer.py --op elementwise_add_relu --max-rounds 1 --stream-hud --out-dir runs/protean-hud-stream-smoke
```

Verified Spark output:

```text
elementwise_add_relu  accepted=1/5
rmsnorm               accepted=0/5
```

Verified HUD stream smoke:

```text
trial 1 -> https://hud.ai/jobs/3203cf74fb314cb29b32389e1a22531d
trial 2 -> https://hud.ai/jobs/f58885c88d77479ab182eb9dd123651f
trial 3 -> https://hud.ai/jobs/ae685277145d472c9030386957be8ce6
trial 4 -> https://hud.ai/jobs/42e800d337b545ac90021be2b08c3cfd
trial 5 -> https://hud.ai/jobs/41b4bc655ac34c2487f435c19c389054
```

## Fireworks Overnight Run

Requires `FIREWORKS_API_KEY`. Add `HUD_API_KEY` and `--stream-hud` when every trial should appear in the HUD dashboard.

```bash
export FIREWORKS_API_KEY=...
export HUD_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --stream-hud --out-dir runs/protean-fireworks-overnight
```

Acceptance criteria:

- [ ] Every Fireworks candidate source is saved.
- [ ] Every trial is logged.
- [ ] Compile/runtime errors appear as `eval_error` and rejected, not lost.
- [ ] Every trial has either `hud_stream.job_url` or `hud_stream_error`.
- [ ] Summary shows accepted count, best score, tokens, and model cost metadata.
