# Production Readiness

This document is an honest accounting of what is production-ready in Protean today versus
what still requires a GPU, load testing, or human review before it can be trusted in a real
production deployment. The guiding rule is the same one the project applies to its reward
artifacts: **claim only what is verified, label everything else.**

The test baseline this document is written against is `180 passed, 3 skipped` (the three
skipped tests are CUDA-only correctness/timing checks that cannot run on a CPU host). Run it
yourself:

```bash
.venv/bin/python -m pytest -q
```

---

## Readiness summary

| Axis | Status | Notes |
|---|---|---|
| CPU verifier surface (AST anti-hack, reward math, splits, eval statistics) | **Production-ready** | Fully unit-tested on CPU; no GPU required. |
| Static sandboxing of untrusted kernel source (import bans, Triton-launch requirement) | **Production-ready (static layer)** | AST-level; see [Sandboxing](#sandboxing-untrusted-code) for the in-process caveat. |
| Reward determinism & reproducibility | **Production-ready** | Frozen manifest + pinned SHA-256; seeded eval; pinned `uv.lock`. |
| Public API stability (`grade_source`, splits, eval_protocol) | **Production-ready** | Covered by the no-API-drift constraint and tests. |
| CI / regression gate | **Production-ready** | GitHub Actions on Python 3.11 + 3.12; lint + tests gate merges. |
| Packaging & dependency pinning | **Production-ready** | `uv.lock`, dependency groups, hatch wheel targets. |
| Eval integrity / provenance | **Production-ready (CPU plumbing)** | Run-provenance envelope + manifest/reward-config hashes; **the powered numbers themselves are GPU-only.** |
| Runtime sandboxing (subprocess isolation, rlimits) | **NOT production-ready** | In-process exec today; subprocess isolation is the GPU-infra capstone. See below. |
| GPU correctness & timing | **NOT validated in CI** | Requires CUDA host; covered by the 3 skipped tests + manual smoke. |
| Money-figure / powered eval numbers | **NOT measured** | Committed artifact is `synthetic=true`; real numbers need a GPU run. |
| Learned controller beating the deterministic baseline | **NOT demonstrated** | Trains from traces; before/after curve unrun. |
| Load / concurrency / multi-tenant throughput | **NOT tested** | No load test exists; see [Open before production](#open-before-production). |

---

## What is production-ready (no GPU needed)

These components are fully exercised on a CPU host and are safe to depend on.

### CPU verifier surface
- **AST anti-hack** (`src/protean/anti_hack.py`): static rejection of banned imports, banned
  call prefixes, and missing `@triton.jit`. Pure string-set / AST-node checks — zero runtime
  cost, fully deterministic, tested in `tests/test_hardening.py` and
  `tests/test_splits_and_redteam.py`.
- **Reward math** (`src/protean/rewards.py`): log-scaled speedup reward, correctness floor,
  hard-fail capping, optional profiling-ratio shaping. Tested in `tests/test_rewards.py`.
- **Splits** (`src/protean/splits.py`): train (`1024, 2048, 4096`) vs off-grid held-out
  (`1535, 3073, 6143`); the held-out / train disjointness is asserted at import time.
- **Eval statistics** (`src/protean/eval_protocol.py`): paired delta, hierarchical + BCa
  bootstrap CI, across-op sign test, MDE / power calculators. All pure-Python and seeded;
  tested in `tests/test_eval_protocol.py` and `tests/test_powered_eval_wiring.py`.

### Reproducibility
- **Frozen manifest**: `manifest_v1.jsonl` is pinned and its SHA-256 is checked against
  `tasks.MANIFEST_SHA256`. A JSON sidecar (`manifest_v1.jsonl.meta.json`) records the
  schema version, content SHA-256, row count, and provenance. Regenerate/verify with
  `scripts/freeze_manifest.py`.
  - **Integrity scope (be precise):** the sidecar protects against *accidental* drift — a
    bit-flip, a botched merge, CRLF mutation, or an out-of-band edit to the data file. It is
    **not** an anti-tamper boundary against an adversary who can write *both* files: the sidecar
    is a plain JSON file in the repo root (not under the `/donotaccess` root-700 moat), so an
    attacker with filesystem write can recompute the SHA and update `n_rows` to match a tampered
    data file. Adversarial integrity for task data is the job of the deployment moat (image
    baking + `/donotaccess`), not the sidecar.
  - **Committed-sidecar provenance is a bootstrap case:** the shipped
    `manifest_v1.jsonl.meta.json` has null `created_at` / `platform` / `python_version` /
    `protean_version` and `generator: "committed (pinned manifest_v1.jsonl)"`. This is expected —
    the sidecar was generated for the pre-existing pinned manifest rather than by a fresh
    `freeze()` on a clean host. The hash, row count, and schema version are real and verified;
    the null provenance fields mean this artifact cannot *prove* it was produced by the canonical
    freeze pipeline. To get machine-verifiable provenance, regenerate both files together via
    `scripts/freeze_manifest.py` on a clean machine and commit them atomically.
- **Deterministic seeding**: samplers derive seeds from SHA-256; eval uses
  `random.Random(seed)`; tests use no wall-clock or unseeded randomness.
- **Pinned environment**: `uv.lock` captures the full transitive tree. The local
  reproducibility gate is `uv sync --locked` (and `uv lock --check`), which fail if the lock
  has drifted from `pyproject.toml`. **Lockfile enforcement is only as strong as the install
  command actually run:** a plain `uv pip install` *without* `--locked` resolves fresh versions
  and ignores `uv.lock` entirely, so reproducibility holds only when CI and contributors install
  with `uv sync --locked` (or `uv pip install --locked`). Verify the CI install step uses one of
  those before relying on this guarantee — see the pre-deploy checklist.

### Public API stability
`grade_source(src, *, op, split, shape, reps, warmup)` is the single direct entrypoint and is
held stable by the no-API-drift constraint. The splits public symbols and the `eval_protocol`
public functions are likewise contract-tested.

### CI / packaging
- GitHub Actions runs lint + the full test suite on Python 3.11 and 3.12. The cache is keyed on
  `uv.lock`; the reproducibility guarantee above is only realized when the install step uses
  `uv sync --locked` (or `uv pip install --locked`). A plain `uv pip install` resolves fresh
  versions and does *not* enforce the lockfile — confirm the workflow's install command before
  trusting CI as a lock-drift gate.
- The project caps Python `>=3.11,<3.13` (`pyproject.toml`); the development venv is **3.12**.
- The wheel ships `src/protean` (and `train/`, which is GPU-only — see the deployment notes).

### Eval integrity (CPU plumbing)
Every eval report carries a run-provenance envelope (run id, timestamps, Python/package
versions, platform, manifest SHA-256, reward-config SHA-256, seed, `n_per_op`, bootstrap `B`,
`argv`). This makes any committed artifact auditable without re-running the pipeline. The
plumbing is production-ready; **the powered numbers it wraps are GPU-only and the committed
artifact is explicitly synthetic** (see below).

---

## Sandboxing untrusted code

Protean grades **LLM-generated kernel source**, which is untrusted input. The sandboxing story
has two layers with very different guarantees — be explicit about which is which.

**Static layer (production-ready):** `anti_hack.ast_clean` parses the candidate and rejects
banned import roots (including the concurrency and binary-embedding families), banned call
prefixes (`torch.ops`, `aten`, `torch.jit.fork`, …), banned builtins by name *and* by
namespace-reconstruction route (e.g. `globals()`, `vars()`, `__builtins__[...]`,
`builtins.__import__`, dotted `*.getattr` / `*.eval` attribute calls), relative imports, and any
source missing a real `@triton.jit`. This is a static, deterministic, CPU-testable gate and it is
the first thing every candidate hits. It closes the obvious obfuscation routes at zero runtime
cost, but it cannot close *every* in-process escape (e.g. CPython subclass-traversal tricks); that
class of attack is only made consequenceless by the runtime layer below and, ultimately, by
subprocess isolation.

**Runtime layer (defense-in-depth only, NOT a hard boundary):**
- A PEP 578 `sys.addaudithook` import backstop catches naive runtime-obfuscated imports that
  slip past static AST analysis. **This is explicitly not a security boundary** — audit hooks
  added from Python (rather than via the C API before interpreter init) can be bypassed by
  adversarial code reaching the C layer (CPython issue #87604). It is documented as such in the
  code.
- A strict `type(out) is torch.Tensor` identity check rejects `FakeTensor` / tensor-subclass
  outputs that would pass `isinstance`.
- A `sys.modules` tombstone + tempfile unlink prevents a candidate's monkey-patches and globals
  from leaking into the next evaluation.

**What is still NOT production-ready:** the candidate is currently executed **in-process** via
`importlib.util.exec_module` (`src/protean/bench_core.load_solution`). A candidate that triggers
a CUDA OOM, corrupts the CUDA context, overwrites a timing primitive, or calls `os._exit()` can
kill or corrupt the entire evaluator session and affect all subsequent candidates in the run. No
in-process Python mechanism can fully prevent this class of failure.

The correct fix — and the explicit production capstone — is **subprocess isolation**: run
`bench_source` in a fresh spawned subprocess (the KernelGym / SOL-ExecBench pattern), set
`RLIMIT_AS` / `RLIMIT_CPU` / `RLIMIT_CORE` / `RLIMIT_NOFILE` in a Linux `preexec_fn`, and enforce
a wall-clock timeout in the parent with SIGKILL + `caps=['timeout']`. The `bench_source` return
dict is already JSON-serializable, so the IPC boundary is clean and the wrapper is CPU-testable
by mocking `bench_source`. **Real execution of this path is GPU-blocked and must be validated on
a CUDA host before production.**

---

## What requires GPU validation, load testing, or human review

These are not faked and not claimed as done. Each must be exercised on the right infrastructure
before a real production launch.

### Requires a CUDA host
- **GPU correctness & timing** — the three skipped tests are CUDA-only. Real validation is the
  manual smoke path: `scripts/check_redteam.py`, `scripts/smoke_verifier.py --op <op>`.
- **Powered held-out eval numbers** — the committed `demo/powered-eval-200.json` is
  `synthetic=true`, `powered_real=false`, `cuda_unavailable=null`. It exercises the *same*
  statistics over a labeled synthetic effect so the report shape is real, but the numbers are
  not measured GPU data. Regenerate the real version with
  `scripts/run_powered_eval.py --run-dir <run>` on CUDA. The `-200` suffix is the target scale
  (5 ops × 40), not the current count (3 ops × 40 = 120).
- **Money-figure speedups** — the README table predates the current off-grid held-out split and
  must be re-captured on `1535, 3073, 6143`. `softmax_rows` has no measured numbers yet.
- **Subprocess-isolation execution path** — see [Sandboxing](#sandboxing-untrusted-code).

### Requires load / concurrency testing
- No load test, soak test, or concurrency test exists. Throughput, the safety of concurrent
  graders sharing a GPU, and behavior under memory pressure are all unmeasured.

### Requires human review before launch
- **Learned controller**: not yet shown to beat the deterministic baseline. Do not present it
  as the core result without an actual before/after curve.
- **GRPO stretch path**: opt-in and gated behind env vars; treat as experimental.
- **Hidden-verifier moat** (`Dockerfile.hud`): the `/donotaccess` root-700 layout and the baked
  `rewards.py` SHA should be re-reviewed against the live HUD threat model before each release.

---

## Deployment / runbook

### Deploy the HUD environment (GPU)
`Dockerfile.hud` builds the CUDA image used by HUD:
- Base `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Python 3.11 venv, Triton cache at `/triton-cache`.
- Installs the training stack (`requirements-train.txt`) and `hud-python[agents]`.
- Bakes the `rewards.py` SHA-256 into the hidden graders and copies the canonical verifier to
  `/donotaccess` (root-owned, `chmod 700`) so HUD agents (uid 1000) cannot read it. The build
  asserts the moat is unreadable by the `agent` user before the image is accepted.
- In-tree `donotaccess` dirs are re-locked to `700`/`600` and chowned to root.

Sync and run against HUD:
```bash
hud deploy . --no-env
hud sync tasks protean-kernel-optimizer src/protean/env.py --yes
hud eval protean-kernel-optimizer claude --full --group 3 --max-concurrent 4
```
HUD auth via `export HUD_API_KEY=...` or `hud set HUD_API_KEY=...`. Do **not** `source .env`
(it may contain non-shell-safe notes).

### Pre-deploy checklist
1. `.venv/bin/python -m pytest -q` is green (`180 passed, 3 skipped` on CPU).
2. `uv sync --locked` and `uv lock --check` succeed (lockfile not drifted), and the CI install
   step uses `uv sync --locked` / `uv pip install --locked` (a plain `uv pip install` would
   silently ignore `uv.lock`).
3. Lint passes (`ruff check .`, `ruff format --check .`).
4. Manifest integrity: `scripts/freeze_manifest.py` SHA matches `tasks.MANIFEST_SHA256`.
5. On a CUDA host: `scripts/check_redteam.py` and `scripts/smoke_verifier.py` pass for every op.
6. Re-capture money-figure numbers on the current off-grid held-out shapes.
7. Regenerate the **real** powered eval (`scripts/run_powered_eval.py --run-dir <run>`) and
   confirm `synthetic=false`, `powered_real=true`, and a non-null CUDA status.

### Operate
- **Audit trail**: the optimizer writes an append-only `trials.jsonl` (one complete JSON object
  per line) plus per-candidate sources. Final artifacts (`summary_*.json`, `best_kernel_*.py`)
  use atomic writes (`os.replace`) and carry a `best_kernel_sha256` for downstream verification.
- **Structured logs**: graders/optimizer emit JSON-keyed log events under the `protean.*`
  loggers, correlated by `run_id`. To trace a run end-to-end:
  `grep run_id=<id> <logs>` across the JSONL trial log and the log stream.
- **Provenance**: every eval report embeds the provenance envelope; an auditor can verify the
  manifest, reward-config, and package versions from the artifact alone.

### Roll back / recover
- The hidden verifier is baked into the image; roll back by deploying the previous image tag.
- `trials.jsonl` is append-only and line-atomic — a partial final line is safe to discard; all
  earlier trials remain recoverable.
- If `summary_*.json` / `best_kernel_*.py` look truncated, the atomic-write design means the
  previous good version was never partially overwritten; re-derive from `trials.jsonl` if needed.

---

## Open before production

A short, honest list of what a real production launch still needs and which axis owns it:

- [ ] Subprocess isolation for `bench_source` validated on a CUDA host (runtime sandbox).
- [ ] GPU correctness/timing tests run in a CUDA CI lane (the 3 currently-skipped tests).
- [ ] Real powered-eval numbers captured (`synthetic=false`).
- [ ] Money-figure speedups re-measured on the current off-grid split; `softmax_rows` benchmarked.
- [ ] Load / concurrency / memory-pressure testing of the grader.
- [ ] Human review of the learned controller before it is presented as a result.
- [ ] Human review of the `/donotaccess` moat against the live HUD threat model per release.
