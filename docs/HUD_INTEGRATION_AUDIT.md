# HUD Integration Audit

Verifier-first GPU-kernel optimization HUD env. This audit tracks the depth and
correctness of Protean's HUD v6 integration against the idiomatic reference
starters (`verilog-template`, `coding-template`) and the installed SDK
(`hud-python[agents]==0.6.6`).

---

## Resolved in this PR

The audit graded the integration at depth **2.5 / 5** and flagged four
deploy-time failure modes ("deploy bombs"). All blocking bombs are now defused.
Fixes are surgical and additive; the public API (`grade_source` / `splits` /
`eval_protocol`) is unchanged. Verification commands and results are at the
bottom of this section.

| # | Bomb | Status | Fix |
|---|------|--------|-----|
| 1 | `Dockerfile.hud` `CMD` used `env.run()` — a deprecated v5 `LegacyEnvMixin` shim, invisible to `hud deploy`'s static `CMD` parser | **Resolved** | Replaced with the v6 serve entrypoint `python3 -m hud serve src/protean/env:env --host 0.0.0.0 --port 8765` (matches the reference starters). |
| 2 | `env.py` had `from __future__ import annotations`, which turns every `@env.template` parameter annotation (e.g. `shape: int \| None`) into a string forward-ref; the HUD server then runs `TypeAdapter` on the string and raises `PydanticUserError` (JSON-RPC `-32000`) at serve/task-start | **Resolved** | Removed the future-import; replaced with a block comment forbidding its return. PEP 604 `int \| None` / `str \| None` is native from Python 3.10 on, so removal is runtime-safe on both the 3.11 deploy image and the 3.12 dev venv (see the Python-version note below). `grader.py` carries the **same** future-import removal + comment for symmetry: it defines no `@env.template` functions so the TypeAdapter crash does not apply there today, but keeping annotation evaluation symmetric closes the surface if a graded function is ever decorated and moved into `env.py`. |
| 3 | Dynamic template registration via a `for`-loop with a variable `id` was suspected unsafe for `hud deploy` static parsing | **Resolved / not a bomb** | The current code already uses three explicit string-literal decorators (`@_template("elementwise_add_relu")`, etc.). `hud deploy` does **not** statically parse `@env.template(id=...)` — template ids are pure runtime; only `Environment(name="protean")` must be a string literal, and it is. No change needed beyond confirming. |
| 4 | The guarded `from hud import Environment` import means a *failed* hud import in the deploy image would silently register **zero** templates and serve an empty env | **Resolved** | Kept the guarded import (CI's `test` extra omits `hud`, so the package must still import without it). Added a serve/deploy-only loud-fail guard: `assert_templates_registered()` runs (a) at import time when a serve verb is detected via `sys.argv`/`HUD_SERVE`, and (b) from an `@env.initialize` serve-start hook. Both raise `RuntimeError` if `env is None` or any expected template is missing. The hud-less import path is untouched. |

### Module-path nuance (folded into Fix 1)

The reference starters keep `env.py` at the image `WORKDIR` root, so their spec
is `env:env`. Protean's env file lives at `src/protean/env.py`, and hud's serve
loader (`_load_environment`) treats the token before `:` as a **filesystem
path**, not a dotted module. The dotted form `protean.env:env` would resolve to
the nonexistent `Path("protean.env.py")`, so the correct spec is the path form
`src/protean/env:env` (resolved against `WORKDIR=/app` -> `/app/src/protean/env.py`).

### Config-driven reward ceiling (applied)

`grader.to_eval_result` previously normalized with a hardcoded `/ 2.0`. That was
mathematically correct only for `max_reward=2.0`; a production `reward_config.json`
changing `max_reward` would have silently broken the `[0, 1]`
`EvaluationResult.reward` contract. This has been fixed: normalization now uses
`raw / _reward_ceiling()`, where `_reward_ceiling()` reads `max_reward` from the
config (falling back to `DEFAULT_CONFIG.max_reward` when the config is
missing/unreadable or the configured ceiling is non-positive). The default
ceiling is still `2.0`, so default behaviour is unchanged. See
[`HUD_INTEGRATION.md`](./HUD_INTEGRATION.md) §4.

### Python-version note (deploy image vs dev venv)

`Dockerfile.hud` builds on `nvidia/cuda:12.4.1-devel-ubuntu22.04` and installs
`python3.11` (Ubuntu 22.04's system Python); the dev venv runs Python 3.12.
`pyproject.toml` declares `requires-python = ">=3.11,<3.13"`, so both are
in-range and the image builds. The Bomb 2 fix is safe on **both**: PEP 604
`int | None` unions evaluate natively from Python 3.10 onward, so removing the
future-import does not depend on the 3.12-specific behaviour the dev venv uses.
If the deploy image is ever rebuilt against a 3.12 base
(`nvidia/cuda:12.x-*-ubuntu24.04` ships 3.12), the `python3.11`/`python3.11-venv`
package names and the `python3.11 -m venv` line must be bumped to `python3.12`
to keep the runtime aligned with the dev venv. Tracked as runtime-alignment
hygiene, not a deploy bomb.

### Verification

```
.venv/bin/python -m pytest -q          # 368 passed, 5 skipped
.venv/bin/python -m mypy src/protean   # Success: no issues found in 29 source files
.venv/bin/ruff check src/protean/env.py# All checks passed!
.venv/bin/python -m hud task list --source src/protean/env.py
#   -> lists 6 task instances across elementwise_add_relu / rmsnorm / softmax_rows
```

Registration also confirmed programmatically:
`protean.env.registered_template_ids()` returns
`('elementwise_add_relu', 'rmsnorm', 'softmax_rows')` when hud is installed, and
`()` (non-fatal) on the hud-less import path. The serve guard was exercised by
blocking the `hud` import under `HUD_SERVE=1`: it raised the expected
`RuntimeError`; without the serve context the same blocked import succeeds with
`env is None`.

---

## Reference contract notes (for future audits)

- **Serve entrypoint:** `hud serve <module>:<attr>` (or `python3 -m hud serve …`,
  which dispatches via `hud/__main__.py`). `--host 0.0.0.0` is required in a
  container (default `127.0.0.1` is unreachable).
- **`Environment(name=...)`** must be a string literal — `hud deploy` extracts
  it by AST-parsing the constructor call. `@env.template(id=...)` ids have no
  such constraint (runtime-only).
- **`EvaluationResult`** fields: `reward: float` (≈0..1), `done: bool = True`,
  `content: str | None = None`, `info: dict = {}`, `isError: bool = False`,
  `subscores: list[SubScore] | None`. A model-validator warns (does not error)
  if positive subscore weights don't sum to ~1.0 or if `sum(value*weight) !=
  reward`. Protean's single weight-1.0 `hud_reward` subscore satisfies this; the
  remaining subscores are weight-0 diagnostics.
