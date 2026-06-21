# Build Checklist

## Local

- [x] Core package imports without Torch/Triton installed.
- [x] CPU-safe tests cover reward, split, and static red-team checks.
- [x] GPU scripts fail closed with `cuda_unavailable` when no GPU stack exists.

## Spark

- [ ] Create user-local venv.
- [ ] Install `.[gpu,test]`.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python scripts/check_redteam.py`.
- [ ] Run `python scripts/smoke_verifier.py`.
- [ ] Run `python scripts/run_demo_benchmark.py`.
- [ ] Keep demo artifacts only if they contain real Spark CUDA timings.

## Then Add

- [ ] `rmsnorm`.
- [ ] HUD packaging.
- [ ] Training.
