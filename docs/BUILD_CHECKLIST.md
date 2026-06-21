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

```bash
python scripts/run_optimizer.py --all-ops --max-rounds 1 --out-dir runs/protean-hud-preflight
```

Verified Spark output:

```text
elementwise_add_relu  accepted=1/5
rmsnorm               accepted=0/5
```

## Fireworks Overnight Run

Blocked until `FIREWORKS_API_KEY` is present on Spark.

```bash
export FIREWORKS_API_KEY=...
python scripts/run_optimizer.py --edit-policy fireworks --all-ops --max-rounds 20 --out-dir runs/protean-fireworks-overnight
```

Acceptance criteria:

- [ ] Every Fireworks candidate source is saved.
- [ ] Every trial is logged.
- [ ] Compile/runtime errors appear as `eval_error` and rejected, not lost.
- [ ] Summary shows accepted count, best score, tokens, and model cost metadata.
