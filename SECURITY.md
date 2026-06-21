# Security Policy

Protean is a verifier-first RL environment that grades **LLM-generated GPU-kernel
source**. Candidate source is **untrusted input by definition**: it is produced by an
arbitrary policy whose objective is to maximize the reward, including by faking it. This
document states the threat model, the controls in place today, the boundaries those
controls do **not** provide, and how to report a vulnerability.

The concrete design for the hard isolation boundary that closes the gaps below lives in
[`docs/SANDBOX_DESIGN.md`](docs/SANDBOX_DESIGN.md).

---

## Threat model

**Asset under protection.** The integrity of the reward signal and the secrecy of the
hidden grader. Specifically:

- The reference outputs, reward math (`rewards.py`), and reward config that, on the HUD
  image, live behind the root-700 `/donotaccess` moat (`Dockerfile.hud`).
- The evaluator process and the shared CUDA context: a candidate must not be able to
  corrupt or crash the session for subsequent candidates.
- The host: a candidate must not read or write files outside its sandbox, open network
  sockets, spawn processes, or escalate privileges.

**Adversary.** A reward-maximizing policy emitting Python/Triton source that is executed
by the grader. The adversary can submit any string. It will try to:

1. **Fake correctness/timing** — return a `FakeTensor`/meta/subclass output that passes
   `isinstance`, cache an output keyed on the fixed seed and replay it, or hide the real
   GPU work on an un-timed background thread.
2. **Delegate off the `@triton.jit` path** — call `torch.ops.aten.*`, `torch.nn.functional.*`,
   `torch.jit.fork`, cuBLAS via `ctypes.CDLL("libcublas.so")`, or decode and `dlopen` a
   precompiled `.cubin` — keeping a decorative kernel only to satisfy the launch check.
3. **Read the hidden grader** — open `/donotaccess` via `io.open`/`builtins.open` aliases
   or read the reward config off disk.
4. **Escape the process** — `os._exit()`, a CUDA OOM that corrupts the shared context, an
   infinite-loop kernel that hangs the evaluator, or namespace/`ctypes` escapes into the
   C layer.
5. **Exfiltrate data** — open a socket and ship the recovered reference outputs out.

This taxonomy follows the exploit families enumerated in SOL-ExecBench (arXiv:2603.19173):
concurrency injection, binary-embedding, state-caching, FakeTensor/tensor-subclass, and
environment manipulation.

---

## Current controls

Protean defends in two layers with **very different guarantees**. The first is a hard,
deterministic gate; the second is best-effort defense-in-depth.

### Layer 1 — static AST admission (production-ready)

`anti_hack.ast_clean` (`src/protean/anti_hack.py`) parses every candidate and rejects it
before any execution if it contains:

- **Banned import roots** (`BANNED_IMPORT_ROOTS`): the FS/process-escape family
  (`os`, `sys`, `subprocess`, `importlib`, `pathlib`, `io`, `builtins`), GPU-delegation
  backends (`ctypes`, `cffi`, `cupy`, `pycuda`, `cuda`, `numpy`), the concurrency family
  (`threading`, `concurrent`, `multiprocessing`, `_thread`, `asyncio`), the
  network/exfiltration family (`socket`, `urllib`, `http`, `requests`, `ssl`, …), the
  binary-embedding family (`base64`, `binascii`, `tempfile`, `shutil`, `zipfile`,
  `tarfile`), and the deserialization family (`pickle`, `shelve`).
- **Banned import modules / submodules** (`BANNED_IMPORT_MODULES`): `torch.nn`,
  `torch.ops`, `torch.jit`, `torch._C`, including the `from torch import nn` form and
  relative imports (`from . import x`).
- **Banned calls and prefixes** (`BANNED_CALLS`, `BANNED_PREFIXES`): direct eager/fused
  ops (`torch.add`, `torch.relu`, `torch.matmul`, `torch.compile`, …) and dispatch chains
  (`torch.ops.*`, `torch.nn.*`, `torch.jit.fork`, `torch.cuda.Stream`, `torch._C`).
- **Banned builtins by name and by reconstruction route** (`BANNED_NAMES`,
  `BANNED_CALL_LEAVES`): `eval`, `exec`, `compile`, `__import__`, `open`, `getattr`,
  `setattr`, `globals`, `locals`, `vars`, plus dotted/leaf forms
  (`builtins.__import__`, `mod.getattr`, `*.eval`) and subscript reconstruction
  (`__builtins__["eval"]`, `*.__dict__[...]`).
- **The swallow-and-return timing-shell** (`_handler_swallows`): a `try/except` whose
  handler reaches a `return` (the documented pattern that falls back to a no-op/cached
  tensor when timing is about to be measured). Handlers that only re-raise are allowed.
- **Missing `@triton.jit`** — checked separately via `contains_triton_jit`.

This layer is pure AST string-set / node analysis: zero runtime cost, fully deterministic,
and CPU-unit-tested (`tests/test_hardening.py`, `tests/test_splits_and_redteam.py`). It is
the first thing every candidate hits.

### Layer 2 — runtime defense-in-depth (NOT a hard boundary)

These run during/around in-process execution in `bench_core` and raise the cost of the
attacks the AST layer cannot statically see. **None of them is a security boundary.**

- **PEP 578 import audit hook** (`anti_hack._audit_import_hook`, armed by
  `install_audit_hook` at the `load_solution` chokepoint): blocks runtime-obfuscated
  imports (`__import__("o" + "s")`) that evade static analysis. It mirrors the same ban
  lists so the two stay in lockstep.
- **Real-launch counting** (`_TritonLaunchCounter`): hooks Triton's launch path so a
  count of `0` over the timed window flags a decorative kernel that was never executed
  (`no_kernel_launched` cap). Best-effort and version-guarded; falls back to the
  source-grep heuristic with a `launch_count_unverified` advisory cap.
- **Strict tensor-identity gate** (`_is_strict_tensor`): `type(out) is torch.Tensor`
  (identity, not `isinstance`) rejects `FakeTensor`/meta/subclass outputs — applied both
  pre- and post-trial, because a candidate can return a real tensor first and a fake one
  later.
- **Multi-init correctness** (`_check_correct_multi_init`): re-checks `allclose` across
  several independent random seeds to defeat per-seed output caching.
- **Session hygiene** (`load_solution`): a `sys.modules` tombstone plus tempfile unlink so
  a candidate's monkey-patches, globals, and on-disk source do not leak into the next
  evaluation.

---

## Explicit non-guarantees

**In-process `exec` is NOT a hard sandbox.** The candidate is executed in the evaluator's
own process via `importlib.util.exec_module` (`bench_core.load_solution`). The following
are known, accepted gaps of the current in-process design — they are mitigated, not
closed, by Layers 1–2, and are the explicit subject of `docs/SANDBOX_DESIGN.md`:

- **Audit-hook bypass via the C layer.** An audit hook added from Python (via
  `sys.addaudithook`, not the C API `PySys_AddAuditHook` before `Py_Initialize`) can be
  bypassed by adversarial code that reaches the C layer directly. `ctypes.CDLL` /
  `cffi` can `dlopen` `libcublas.so` and call CUDA without firing any Python-level audit
  event; `_ctypes.dlsym` does not reliably fire audit hooks (CPython issue #87604, Python
  bug tracker 38622). The `ctypes`/`cffi` import *roots* are banned at the AST layer, but
  the audit hook itself is not a boundary against code that has already reached C.
- **Static-analysis evasion.** The AST walk can be evaded by string-concat obfuscation of
  `__import__`, by C-extension side channels that emit no AST nodes, or by bytecode
  injection post-parse. Layer 2 catches the naive runtime forms; it does not catch all of
  them.
- **Process integrity.** A candidate that triggers a CUDA OOM, corrupts the shared CUDA
  context, overwrites a timing primitive in the shared process, or calls `os._exit()` can
  kill or corrupt the evaluator session and affect all subsequent candidates. No
  in-process Python mechanism can fully prevent this.
- **Driver hang.** An infinite-loop GPU kernel hangs the evaluator with no automatic
  recovery (Linux has no Windows-TDR equivalent). There is no in-process watchdog today.
- **GPU DRAM remanence.** NVIDIA driver DRAM pages are not zeroed on CUDA-context
  teardown. Without per-process context isolation (or MIG hardware partitioning), a
  candidate that knows the grader's tensor shapes can `cudaMalloc` and read uninitialized
  buffers containing a previous candidate's tensors or the reference outputs — with zero
  exploit code. In-process evaluation has no defense against this.
- **Triton JIT cache poisoning.** `bench_core._ensure_triton_cache_dir` and
  `env.configure_triton_cache_dir` select a single shared cache path
  (`/triton-cache` or `~/.cache/protean-triton`) for all evaluations. A candidate that
  writes a crafted `.cubin` under a collision-prone key could have it loaded by a later
  candidate's JIT. This is a Protean-specific supply-chain vector, not a Linux boundary
  bypass; the fix is a per-candidate cache dir.

`docs/PRODUCTION_READINESS.md` marks **"Runtime sandboxing (subprocess isolation,
rlimits)"** as **NOT production-ready**, consistent with the above.

### Deployment moat (HUD image)

On the HUD GPU image (`Dockerfile.hud`), the canonical verifier, reward math, reward
config, and references are copied to `/donotaccess` (root-owned, `chmod 700`) and the
build asserts the `agent` user (uid 1000) cannot read them. The `rewards.py` SHA-256 is
baked into the hidden graders for integrity. This is a **UNIX-permission moat against the
agent uid**, not a sandbox against in-process code running *as the grader*: it protects
the secret from the agent, but does not contain a candidate that escapes into the grader
process. The two are complementary — the moat hides the secret; the sandbox design
contains execution.

---

## Reporting a vulnerability

If you find a way to (a) make `grade_source` / `bench_source` report a reward not earned
by genuine on-GPU kernel work, (b) read anything under `/donotaccess` or otherwise recover
the reference outputs or reward config, (c) crash, hang, or corrupt the evaluator session
from candidate source, or (d) escape the process/host:

- **Do not** open a public issue with a working exploit.
- Report privately to the maintainers (the address in `pyproject.toml` / repository
  metadata). Include a minimal reproducer (the candidate source string), the op/split, and
  the observed vs. expected reward/caps.
- For static-layer bypasses (a candidate that `ast_clean` should reject but does not), a
  failing test case in the style of `tests/test_hardening.py` is the most useful form.

We aim to acknowledge within a few business days. Static-layer fixes (new ban entries,
widened guards) ship as ordinary patches; runtime-isolation gaps are tracked against the
phased rollout in `docs/SANDBOX_DESIGN.md`.

---

## References

- SOL-ExecBench, arXiv:2603.19173 (exploit taxonomy; subprocess-isolation pattern).
- CPython issue #87604; Python bug tracker 38622 (audit-hook / `dlsym` bypass).
- `docs/PRODUCTION_READINESS.md` (readiness accounting).
- `docs/SANDBOX_DESIGN.md` (hard-isolation design).
- `src/protean/anti_hack.py`, `src/protean/bench_core.py` (current controls).
