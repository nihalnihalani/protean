# HUD Integration

Protean is a verifier-first GPU-kernel optimization environment served on
HUD v6 (`hud-python[agents]==0.6.6`). This document describes the now-complete
wiring — env, serve, deploy, eval, and streaming — the deploy-bomb fixes that
were applied, and an honest accounting of what still requires the GPU/platform
host before the integration is "full".

For the bomb-by-bomb fix log see
[`HUD_INTEGRATION_AUDIT.md`](./HUD_INTEGRATION_AUDIT.md).

---

## 1. Topology

```
agent  ──prompt──▶  @env.template (env.py)  ──source──▶  grade_source (grader.py)
                          ▲                                      │
                          │                                      ▼
                  EvaluationResult  ◀── to_eval_result ──  compute_reward (rewards.py)
                                                            against the hidden
                                                            /donotaccess answer key
```

- **`src/protean/env.py`** — the HUD env file. Constructs
  `Environment(name="protean")` and registers three generative templates.
- **`src/protean/grader.py`** — `grade_source` (verifier) and `to_eval_result`
  (HUD adapter). Public verifier API; unchanged by the deploy-bomb work.
- **`src/protean/rewards.py`** — the reward function, baked root:700 into
  `/donotaccess` at image build (the verifier moat).
- **`src/protean/hud_stream.py`** — optional client-side helpers to stream
  optimizer trials to the HUD dashboard.

---

## 2. Env file (`src/protean/env.py`)

### Templates

Three ops are each registered with an explicit **string-literal** id:

```python
@_template("elementwise_add_relu")
async def elementwise_add_relu(split: str = "train", shape: int | None = None): ...

@_template("rmsnorm")        async def rmsnorm(...): ...
@_template("softmax_rows")   async def softmax_rows(...): ...
```

Each is an async generator: it `yield`s the prompt (op-specific instructions +
grading metadata) and receives the agent's kernel source back, then `yield`s the
graded `EvaluationResult`. Pre-bound task instances expose the `train` and
`held_out` splits as six discoverable tasks (slug + `columns`), confirmed by
`hud task list --source src/protean/env.py`.

### Why no `from __future__ import annotations`

The file deliberately omits the future-import (see the block comment at the top).
Under PEP 563 every `@env.template` parameter annotation becomes a string
forward-ref, and the HUD server runs `TypeAdapter` on it at task-start, raising
`PydanticUserError` (JSON-RPC `-32000`) for `int | None`. PEP 604 `int | None` /
`str | None` unions evaluate natively from Python 3.10 onward, so the import is
unnecessary and harmful here on both the 3.11 deploy image and the 3.12 dev venv
(see the audit's Python-version note). This mirrors the `verilog-template`
guidance.

`grader.py` carries the same future-import removal (and an explanatory comment)
for symmetry. It defines no `@env.template` functions, so the TypeAdapter crash
does not apply there today; keeping annotation evaluation symmetric across the
two modules closes the surface for a future contributor who copies a graded
function into an `@env.template`.

### Dual-mode import + loud-fail serve guard

`from hud import Environment` is wrapped in `try/except` so the package imports
without the `hud` extra (CI's `test` extra omits it; the full suite runs hud-less).
When hud is absent, `env is None` and the template names fall back to plain
dicts — a non-fatal import path.

That same guard could silently deploy a **zero-template** env if the hud import
failed inside the container. To prevent that without breaking the hud-less path,
a serve/deploy-only guard fails loud:

- `assert_templates_registered()` raises `RuntimeError` if `env is None` or any
  of `EXPECTED_TEMPLATE_IDS` is missing.
- It runs (a) at **import time** when a serve verb (`serve`/`dev`) is detected on
  `sys.argv` or `HUD_SERVE` is set, and (b) from an **`@env.initialize`**
  serve-start hook (before the control channel accepts connections).
- Under `pytest` neither trigger fires, so the test suite is unaffected.

The `@env.initialize` hook also warms the Triton cache and best-effort probes
CUDA (a missing GPU is logged, not fatal, so the env boots on CPU-only smoke
hosts). A near-no-op `@env.shutdown` hook is registered for symmetry.

---

## 3. Serve & deploy

### `Dockerfile.hud` CMD

```dockerfile
CMD ["python3", "-m", "hud", "serve", "src/protean/env:env", "--host", "0.0.0.0", "--port", "8765"]
```

- `env.run()` (the previous CMD) is a deprecated v5 `LegacyEnvMixin` shim and is
  **invisible** to `hud deploy`'s static `CMD` parser, which only recognizes
  `hud serve` / `hud dev` tokens. The new CMD is parseable and matches the
  reference starters.
- The module spec is the **path form** `src/protean/env:env` (not the dotted
  `protean.env:env`): hud's serve loader treats the pre-`:` token as a filesystem
  path. With `WORKDIR=/app` and the source copied to `/app`, this resolves to
  `/app/src/protean/env.py`, attribute `env`.
- `--host 0.0.0.0` is required for container reachability.

`hud deploy` resolves the env name by AST-parsing `Environment(name="protean")`
(a string literal) — deterministic.

### Pre-deploy validation (no `--dry-run` exists)

```
hud task list --source src/protean/env.py
```

This imports the module from source and prints every registered task without
building or deploying. Expect six rows (train + held_out for each of the three
ops). This is the canonical pre-deploy smoke check.

---

## 4. Eval contract (`grader.to_eval_result`)

`grade_source` returns the raw verifier dict; `to_eval_result` adapts it to a
HUD `EvaluationResult`:

- `reward` is normalized to `[0, 1]` via `max(0, min(raw / _reward_ceiling(), 1.0))`.
  `_reward_ceiling()` reads `max_reward` from `reward_config.json` (falling back to
  `DEFAULT_CONFIG.max_reward` if the on-disk config is missing/unreadable, and to
  `DEFAULT_CONFIG.max_reward` again if the configured ceiling is non-positive).
  The default `max_reward` is `2.0` (`correct_floor 0.3 + speedup_reward_weight 1.5
  + pr_bonus 0.2`), so by default `/_reward_ceiling()` maps `[0, 2.0] -> [0, 1.0]`
  exactly — but a production config that changes `max_reward` is honoured, instead
  of a hardcoded `/2.0` divisor silently breaking the `[0, 1]` contract.
- One weight-1.0 subscore `hud_reward` carries the score (satisfies the
  `sum(value*weight) == reward` validator); five weight-0 diagnostic subscores
  (`correctness`, `speedup`, `held_out`, `anti_hack`, `compile_success`) surface
  detail in the dashboard without affecting the weighted sum.
- `info` carries the full grade dict plus the raw/normalized rewards for
  debugging.

---

## 5. Streaming (`src/protean/hud_stream.py`)

Optional client-side helpers to push optimizer trials to the HUD dashboard:

- Async core: `_start_session_async`, `_stream_candidate_async` — call directly
  from an existing event loop.
- Sync shims: `start_hud_stream_session`, `stream_candidate_to_hud` — drive the
  coroutines to completion; they only call `asyncio.run` when no loop is running,
  and otherwise dispatch to a worker-thread loop (so they never raise
  "asyncio.run() cannot be called from a running event loop").
- Auth: `resolve_hud_api_key` / `assert_hud_auth` read the HUD user env / API
  key before streaming.

The `group` argument is **logging/bookkeeping only** today: it tags trials so
they group visually in the dashboard. It is **not** wired into any RL advantage
computation — see the gap below.

---

## 6. What remains for FULL integration (honest)

These require the GPU/platform host and are intentionally **not** completed in
this PR (which was scoped to defusing deploy bombs and deepening the serve/eval
wiring):

1. **Training tier (HUD `TrainingClient`).** The streaming layer's `group`
   argument is a dashboard tag, not a GRPO-style group-relative advantage
   signal. Connecting Protean's reward to a HUD `TrainingClient` rollout/advantage
   loop is unbuilt and needs a multi-GPU host to be meaningful.
2. **Grouped platform-eval completion.** The local `hud task list` path is
   verified, but a full grouped eval run on the HUD platform (concurrent rollouts
   across the six tasks, dashboard aggregation) has not been executed end-to-end
   here — it requires the deployed GPU image and platform credentials.
3. **CUDA-real grading on the platform.** Local verification ran on CPU/eager
   (the `@env.initialize` CUDA probe logs "not available"). Speedup numbers and
   anti-hack timing caps are only meaningful on the GPU image.

Everything in sections 2–5 (env registration, serve/deploy contract, eval
adapter, streaming, loud-fail guards) is wired and verified; the items above are
the remaining work to reach a fully exercised training + platform-eval loop.

---

## 7. Verification snapshot

```
.venv/bin/python -m pytest -q             -> 368 passed, 5 skipped
.venv/bin/python -m mypy src/protean      -> Success: no issues found in 29 source files
.venv/bin/ruff check src/protean/env.py   -> All checks passed!
.venv/bin/python -m hud task list --source src/protean/env.py
                                          -> 6 task instances (3 ops x train/held_out)
protean.env.registered_template_ids()     -> ('elementwise_add_relu','rmsnorm','softmax_rows')
```
