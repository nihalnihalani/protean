# Contributing to Protean

Thanks for contributing. Protean is a verifier-first GPU-kernel optimization RL environment.
The core invariant is **the verifier must stay trustworthy**: changes are additive and surgical,
the public API does not drift, and every claim is backed by a test or labeled as unverified.

## Development setup

Protean targets Python `>=3.11,<3.13`. The development virtualenv is **Python 3.12** and lives
at `.venv/`.

```bash
# If you use uv (recommended): create the env and install everything against the lockfile.
uv sync --locked --all-extras --dev

# Or with the committed venv directly:
.venv/bin/python -m pip install -e ".[test]"
```

Most development and the entire test suite run **CPU-only** — no GPU is required. GPU
dependencies (`torch`, `triton`) and the training stack are installed only on a CUDA host:

```bash
# CUDA host only
python -m pip install -e ".[gpu,test,hud]"
```

## Running tests, lint, and type checks

The single source of truth for "is it green" is the full suite on a CPU host:

```bash
.venv/bin/python -m pytest -q
```

The expected baseline is **`180 passed, 3 skipped`**. The three skipped tests are CUDA-only
correctness/timing checks and are expected to skip on a CPU host. **Do not let the passing count
drop and do not add a test that requires CUDA to pass.**

Lint and format (these gate merges in CI):

```bash
uv run ruff check .
uv run ruff format --check .
```

Type checks:

```bash
uv run mypy src/protean
```

If pre-commit hooks are installed they run ruff (with `--fix`), ruff-format, basic file hygiene,
and mypy on commit:

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

## Hard constraints

Every change must satisfy all of these. CI and review will check them.

1. **Keep all 180 tests green** (`180 passed, 3 skipped` on CPU). New behavior gets new tests.
2. **No public-API drift.** Do not change the `grade_source` signature, the `splits` public
   symbols, or the `eval_protocol` public functions. New parameters must be keyword-only with
   defaults, and new dict keys must be additive (never remove existing keys).
3. **CPU-only tests.** No unit test may require CUDA. The three GPU correctness/timing tests are
   already marked to skip on CPU; do not add more required-GPU tests.
4. **Additive and surgical.** Prefer small, targeted edits to existing files over new
   abstractions. No cargo-cult complexity.
5. **No non-determinism in tests.** No wall-clock (`Date.now`-style time reads) and no unseeded
   randomness in tests. Seed everything; derive seeds deterministically.
6. **Ground your choices.** Tie non-trivial design decisions to the current code and to a cited
   source (see `docs/IMPROVEMENT_RESEARCH.md` for the ranked roadmap and references).

## Reproducibility rules

- The task manifest (`manifest_v1.jsonl`) is frozen and pinned by SHA-256 in
  `tasks.MANIFEST_SHA256`. If you change the manifest you must regenerate the hash with
  `scripts/freeze_manifest.py` and update the pinned constant in the same change.
- Dependencies are pinned in `uv.lock`. If you change `pyproject.toml` dependencies, run
  `uv lock` and commit the updated lockfile; `uv sync --locked` and `uv lock --check` must still
  succeed. Install against the lockfile (`uv sync --locked` / `uv pip install --locked`) — a plain
  `uv pip install` ignores `uv.lock` and breaks the reproducibility guarantee.
- Eval artifacts must carry their provenance envelope and integrity hashes. Do not commit
  measured-looking numbers that were generated on CPU — label synthetic artifacts
  (`synthetic=true`) as the existing demo artifact does.

## Honesty rules

This project is deliberately conservative about claims.

- Label anything that needs a GPU, load testing, or human review as such — see
  `docs/PRODUCTION_READINESS.md`. Do not present synthetic plumbing as measured results.
- The committed powered-eval artifact is synthetic; do not change its labels to imply it is real.
- If a feature is wired but unproven (e.g. the learned controller beating the deterministic
  baseline), say so in the PR rather than overstating it.

## Pull request conventions

- **Branch** off the default branch; do not commit directly to it.
- **Scope**: keep PRs focused. One concern per PR makes review and rollback easier.
- **Title**: concise and imperative, e.g. `anti_hack: ban concurrency import roots`.
- **Description** should state:
  - what changed and why (link the relevant `docs/IMPROVEMENT_RESEARCH.md` item if applicable),
  - the test result line (`180 passed, 3 skipped`),
  - any new public dict keys / keyword-only params (confirming no API drift),
  - whether any path is GPU-blocked and therefore only CPU-tested via mocks.
- **Before requesting review**, confirm locally: tests green, `ruff check`/`ruff format --check`
  clean, `mypy src/protean` clean, and `uv sync --locked` succeeds.
- CI must be green before merge. Lint failures and lockfile drift are hard failures.

## Where things live

| Path | Role |
|---|---|
| `src/protean/grader.py` | `grade_source` — the direct verifier entrypoint. |
| `src/protean/anti_hack.py` | Static AST anti-hack checks. |
| `src/protean/bench_core.py` | CUDA correctness + timing harness (GPU). |
| `src/protean/rewards.py` | Reward math (log-speedup, profiling-ratio shaping). |
| `src/protean/splits.py` | Train vs off-grid held-out shape splits. |
| `src/protean/eval_protocol.py` | Paired delta, bootstrap CI, sign test, power/MDE. |
| `tests/` | CPU-only test suite (180 pass, 3 skip). |
| `scripts/` | Smoke verifiers, optimizer runner, powered-eval, manifest freeze. |
| `docs/PRODUCTION_READINESS.md` | Honest production-readiness accounting + runbook. |
| `docs/IMPROVEMENT_RESEARCH.md` | Ranked roadmap with cited sources. |
