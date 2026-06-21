# Sandbox Design — Hard Isolation for Untrusted GPU-Kernel Evaluation

This document specifies a concrete, implementable design for executing untrusted
LLM-generated kernel candidates under a **hard isolation boundary** on a Linux GPU host.
It is the engineering capstone behind the non-guarantees listed in
[`../SECURITY.md`](../SECURITY.md) and the "Runtime sandboxing" row that
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) marks **NOT production-ready**.

The static AST layer (`anti_hack.ast_clean`) and the runtime defense-in-depth in
`bench_core` remain in place and are **not replaced** by this design. They are the cheap
first filters; this design is the boundary that makes the residual in-process escapes
*inconsequential*.

---

## 1. Why in-process `exec` cannot be the boundary

`bench_core.load_solution` executes candidate source in the evaluator's own process via
`importlib.util.exec_module`. As documented in `anti_hack._audit_import_hook` and
`SECURITY.md`:

- A Python-installed `sys.addaudithook` can be bypassed by code reaching the C layer;
  `ctypes.CDLL("libcublas.so")` / `cffi` call CUDA with no audit event
  (CPython #87604, bug 38622).
- `os._exit()` kills the evaluator with no cleanup; a CUDA OOM or driver fault corrupts the
  shared CUDA context for every later candidate.
- GPU DRAM is not zeroed on context teardown — a later candidate can read a prior
  candidate's tensors or the reference outputs (DRAM remanence; CVE-2023-0184 class).
- An infinite-loop kernel hangs the evaluator (no Linux TDR watchdog).

None of these has an in-process Python fix. The boundary must be **a separate process
with reduced privileges, a wall-clock-enforced lifetime, and an isolated CUDA context.**

---

## 2. Design overview

Run `bench_source` for **each candidate in a fresh subprocess**. The
`bench_source` return value is already a JSON-serializable `dict` (verified in
`bench_core.py`), so the IPC boundary is clean: the child writes `json.dumps(result)` to a
pipe, the parent reads `json.loads(...)`. This matches the KernelGym / SOL-ExecBench
(arXiv:2603.19173) per-candidate subprocess pattern referenced in
`IMPROVEMENT_RESEARCH.md` item #15.

```
parent (grader)                         child (per candidate)
---------------                         ---------------------
grade_source(src, ...)
  └─ spawn child  ── src, op, split, shape ──▶  bench_source(src, ...)
     threading.Timer(deadline, kill)                 │ fresh CUDA context
     read stdout                                     │ per-candidate TRITON_CACHE_DIR
        ◀──────── json.dumps(result) ───────────────┘ atexit: zero device buffers
  └─ json.loads(result)
       on timeout  → caps=['timeout']
       on crash    → caps=['sandbox_crash']
```

**Primary isolation layer: nsjail subprocess per candidate** (used by Windmill and Google
CTF infra). For an MVP without nsjail packaging, a plain `subprocess.Popen` with a
`preexec_fn` (rlimits + namespaces) provides most of the boundary and is CPU-testable by
mocking. Prefer nsjail over firejail: firejail has documented compatibility issues with
proprietary NVIDIA drivers and lacks nsjail's Kafel BPF policy language for precise
`ioctl` filtering. Treat **gVisor/nvproxy as a future upgrade**, not an MVP dependency —
it introduces a hard NVIDIA-driver-version alignment constraint that is fragile across CUDA
releases.

---

## 3. The five GPU-specific non-negotiables

These are the additions specific to a GPU host, beyond a generic code sandbox.

### 3.1 Bind-mount GPU devices read-write; omit `/donotaccess` entirely

The single most critical correctness fix. The candidate needs the GPU device nodes but
must never see the hidden grader.

- Bind-mount **read-write** (CUDA writes to these): `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm` (and `/dev/nvidia-uvm-tools`, `/dev/dri/*` as the driver requires).
- Bind-mount **read-only**: the interpreter and site-packages
  (`/usr/local/lib/python3.12` or the venv).
- **Do NOT bind-mount `/donotaccess`.** The current in-process path exposes the grader's
  CUDA context and reward config to every candidate; omitting it from the sandbox's mount
  list is what actually closes that exposure. The Dockerfile root-700 moat is a permission
  fallback, not the boundary.

nsjail form:

```
-B /dev/nvidia0::/dev/nvidia0
-B /dev/nvidiactl::/dev/nvidiactl
-B /dev/nvidia-uvm::/dev/nvidia-uvm
-R /opt/venv::/opt/venv          # interpreter + packages, read-only
# (no -B / -R for /donotaccess — simply absent)
```

### 3.2 Per-subprocess `TRITON_CACHE_DIR` on tmpfs

Prevents Triton JIT cache poisoning (a Protean-specific vector, §SECURITY non-guarantees).
Today `bench_core._ensure_triton_cache_dir` and `env.configure_triton_cache_dir` pick one
shared path for all evaluations. In the child, set before exec:

```
TRITON_CACHE_DIR=/tmp/protean-{uuid}/triton-cache
```

with `/tmp` as a fresh tmpfs (`--tmpfsmount /tmp --tmpfssize 64m` in nsjail). Each
candidate's compiled cubins are then isolated; one candidate cannot overwrite a cached
cubin loaded by another.

### 3.3 Parent-side SIGKILL watchdog (NOT `RLIMIT_CPU` alone)

`RLIMIT_CPU` bounds CPU time but **not GPU time**, and async CUDA errors on modern GPUs
(Blackwell; pytorch/pytorch #178491) can hang the subprocess silently. The parent owns a
wall-clock deadline (~60 s/candidate):

```python
proc = subprocess.Popen(argv, ..., preexec_fn=_child_setup, start_new_session=True)
timer = threading.Timer(WALL_CLOCK_S, lambda: os.killpg(os.getpgid(proc.pid), signal.SIGKILL))
timer.start()
try:
    out, _ = proc.communicate()
finally:
    timer.cancel()
```

`start_new_session=True` puts the child in its own process group so `killpg` reaps any
grandchildren. Set `CUDA_LAUNCH_BLOCKING=1` in the child so launches are synchronous and
CPU-time limits act as a usable proxy for GPU time. After a kill, the parent must call
`torch.cuda.synchronize()` and drain pending async errors **in a fresh child** before the
next candidate (the grader process's own context, if any, stays clean because compute
happens in children).

### 3.4 No-network namespace + seccomp denylist that does not break CUDA

- **No network:** `--iface_no_lo` (nsjail) or `CLONE_NEWNET` in `preexec_fn` — no
  loopback, closing socket-based exfiltration and the fetch-a-cubin-over-network vector.
- **seccomp (Kafel) — deny, without breaking CUDA:**
  - `ptrace` (blocks `/proc/$pid/mem` reads of the parent),
  - `clone` with `CLONE_NEWUSER` (blocks user-namespace escape),
  - `mount` / `umount2` (blocks overmounting the read-only binds),
  - `kexec_load` / `kexec_file_load`, `keyctl`,
  - `perf_event_open` (timing side channels against co-located GPUs).
  - **Allow `ioctl` only on fds pointing at `/dev/nvidia*` and `/dev/dri/*`, deny
    otherwise.** NVIDIA `ioctl` codes are not publicly enumerable, so a numeric whitelist
    is unmaintainable; gate by fd path instead. `open`/`read`/`write` need no syscall ban —
    the read-only FS binds and absent network namespace already contain them.

### 3.5 Zero device buffers before child exit (non-MIG DRAM remanence)

On hardware without MIG, register a child `atexit` handler that **zeroes device memory
before freeing it.** This is critical and easy to get wrong: `torch.cuda.empty_cache()`
returns the caching allocator's blocks to the driver but **does not zero the pages** — a
sibling process's subsequent `cudaMalloc` can still read the old contents. The scrub must
*write zeros into each live buffer, synchronize, then free*, and only after that call
`empty_cache`:

```python
import atexit, gc, torch

def _scrub():
    # 1) Zero every live CUDA tensor in-place (overwrite the actual DRAM pages),
    #    then drop our references so the allocator can release the now-zeroed blocks.
    try:
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                try:
                    obj.detach().zero_()   # in-place write of 0x00 over the pages
                except Exception:
                    pass
        torch.cuda.synchronize()           # ensure the zero-writes actually landed
    finally:
        # 2) Only now release the (zeroed) blocks back to the driver.
        gc.collect()
        torch.cuda.empty_cache()

atexit.register(_scrub)
```

This `gc`-walk is a **best-effort heuristic**, not a guarantee: it cannot reach buffers
already freed-but-not-zeroed earlier in the run, pages held by cuBLAS/cuDNN workspaces, or
DRAM the driver reassigns out from under the process. The **only reliable control is MIG**
(below). On A100/H100/H200, **MIG** (hardware-enforced DRAM partitions) is the correct
primary control when co-tenanting untrusted code; per-process CUDA context isolation is the
minimum acceptable floor, and the `atexit` zero-then-free scrub is a defense-in-depth layer
*under* that floor, not a substitute for it. **MPS must not be used** with untrusted code:
a CUDA error in one MPS client kills the whole shared MPS context and all co-located
clients.

---

## 4. Resource limits

### 4.1 rlimits (child `preexec_fn`, before exec)

```python
import resource
resource.setrlimit(resource.RLIMIT_CPU,   (CPU_S, CPU_S))      # CPU-time ceiling
resource.setrlimit(resource.RLIMIT_AS,    (AS_BYTES, AS_BYTES))# host address space
resource.setrlimit(resource.RLIMIT_CORE,  (0, 0))             # no core dumps
resource.setrlimit(resource.RLIMIT_NOFILE,(64, 64))           # fd ceiling
resource.setrlimit(resource.RLIMIT_NPROC, (2, 2))             # anti fork-bomb
```

These complement, never replace, the §3.3 parent watchdog.

### 4.2 cgroups v2

On cgroups-v2 hosts (Ubuntu 20.04+), place the child in a dedicated slice:

- `memory.max` — hard host-RAM ceiling so the OOM killer fires inside the child's cgroup,
  not against the grader.
- `cpu.max` — bound CPU fraction.
- `pids.max = 1` or `2` — fork-bomb defense.

nsjail integrates this via `--cgroup_mem_max` / `--cgroup_cpu_ms_per_sec`. Without nsjail,
the parent writes the child PID to `/sys/fs/cgroup/protean-sandbox/cgroup.procs` after fork,
before exec.

**GPU memory has no cgroup controller** in the mainline kernel; it is governed only by the
MIG DRAM partition or per-context limits. The CPU-side proxy is application-level
`tensor.storage().nbytes()` tracking with an early-abort cap.

### 4.3 Dropped capabilities and read-only FS

- Drop all Linux capabilities (`--cap_bset 0`). `CAP_SYS_PTRACE` and `CAP_SYS_RESOURCE`
  stay dropped (rlimits are set by the parent before exec). Evaluate `CAP_IPC_LOCK` only if
  the Triton path actually needs pinned memory (`cudaHostAlloc`).
- FS layout: interpreter + site-packages read-only (`-R`); `/tmp` as a 64 MiB tmpfs with
  `TRITON_CACHE_DIR` underneath it; `/proc` read-only or omitted so `/proc/self/mem` is
  unavailable; `/donotaccess` absent (§3.1).

---

## 5. IPC contract and failure mapping

The child runs a thin entrypoint. **This entrypoint is not implemented yet** —
`protean._sandbox_child` does **not** exist in `src/protean/` today; the snippet below is
**target pseudocode** for the Phase-0 deliverable in §6, not runnable code. (`bench_source`
and `task_catalog.get_op` referenced in it *do* already exist and the call shape below
matches the real `bench_source` signature in `src/protean/bench_core.py`.) Attempting to run
`python -m protean._sandbox_child` today raises `ModuleNotFoundError` by design until
Phase 0 lands. See `SECURITY.md` "Explicit non-guarantees" — runtime isolation is **NOT
production-ready**.

```python
# TARGET PSEUDOCODE — NOT YET IMPLEMENTED (Phase 0 deliverable, §6).
# Tracked as the first rollout step; no module protean._sandbox_child exists today.
# child: python -m protean._sandbox_child  (reads a JSON job on stdin)
import json, sys
from protean.bench_core import bench_source     # exists today
from protean.task_catalog import get_op         # exists today
job = json.loads(sys.stdin.read())
spec = get_op(job["op"])
result = bench_source(job["src"], n=job["shape"], split=job["split"], spec=spec,
                      reps=job["reps"], warmup=job["warmup"])
sys.stdout.write(json.dumps(result))
```

The parent maps child outcomes to the same `caps` vocabulary the reward layer already
understands:

| Child outcome                         | Parent result                          |
|---------------------------------------|----------------------------------------|
| clean exit, valid JSON                | the parsed `bench_source` dict         |
| killed by watchdog (§3.3)             | synthetic dict, `caps=['timeout']`     |
| non-zero exit / unparseable stdout    | synthetic dict, `caps=['sandbox_crash']`|
| seccomp/`SIGSYS` kill                 | synthetic dict, `caps=['sandbox_crash']`|

The synthetic dicts mirror the shape of `bench_source._non_tensor_result` (zeroed
timings/speedup, `correct=False`) so the reward gate treats a timeout/crash as a hard fail
without special-casing. **Public APIs are unchanged**: `grade_source` /
`bench_source` signatures are preserved; the subprocess wrapper sits *inside*
`grade_source`'s call to `bench_source` and returns the same dict shape.

---

## 6. Phased rollout

1. **Phase 0 — wrapper + IPC (CPU-testable, no GPU).** Create the child entrypoint module
   `src/protean/_sandbox_child.py` (the §5 pseudocode, which does not exist yet) and the
   parent-side spawn/timeout/JSON-parse wrapper. Test by mocking `bench_source` with a
   fake subprocess that emits a fixed JSON dict, a hanging process (assert `timeout` cap),
   and a crashing process (assert `sandbox_crash` cap). Keep all existing tests green; no
   CUDA-required unit test. This phase lands the boundary plumbing with zero GPU access.
2. **Phase 1 — `preexec_fn` hardening (Linux, GPU host).** Add rlimits, `start_new_session`,
   `CLONE_NEWNET`, per-candidate `TRITON_CACHE_DIR`, and the `atexit` scrub. Validate on a
   CUDA host that legitimate Triton kernels still run and that a `socket` import / `os._exit`
   candidate is contained.
3. **Phase 2 — nsjail + seccomp.** Replace `preexec_fn` with an nsjail invocation carrying
   the §3.1 binds, §3.4 Kafel policy, and §4 cgroup limits. Validate the same corpus plus
   `ptrace`/`mount`/`kexec` denial.
4. **Phase 3 — hardware GPU isolation.** Enable MIG on A100/H100/H200 (or document the
   non-MIG `atexit`-scrub floor on consumer cards). Confirm no cross-candidate DRAM read.
5. **Phase 4 — gVisor/nvproxy (optional, higher assurance).** Only after pinning the host
   NVIDIA driver to an nvproxy-supported version; ~50 ms startup overhead is negligible for
   multi-second benchmark runs but the driver-version coupling is a maintenance cost.

---

## 7. How to test

- **CPU / CI (Phase 0):** mock the subprocess to assert the three caps mappings (success,
  `timeout`, `sandbox_crash`) and that the parsed dict equals the mocked `bench_source`
  output. These are deterministic and require no CUDA — consistent with the project's
  no-CUDA-unit-test constraint.
- **GPU smoke (Phases 1–3):** extend `scripts/check_redteam.py` / `scripts/smoke_verifier.py`
  to run each red-team candidate (`socket` open, `os._exit`, infinite-loop kernel,
  shape-known DRAM read, JIT-cache-poison) through the sandbox and assert containment
  (`timeout`/`sandbox_crash`/reject) plus a clean evaluator afterward.
- **Negative control:** a legitimate `@triton.jit` kernel must still produce a real,
  non-zero speedup through the sandbox — proving the boundary does not break CUDA.

---

## References

- SOL-ExecBench, arXiv:2603.19173 (exploit taxonomy; per-candidate subprocess pattern).
- `IMPROVEMENT_RESEARCH.md` item #15 (subprocess isolation; wrapper is CPU-testable).
- CPython issue #87604; Python bug tracker 38622 (audit-hook / `dlsym` bypass).
- pytorch/pytorch #178491 (silent async-CUDA hang on Blackwell → parent SIGKILL required).
- nsjail: github.com/google/nsjail, nsjail.dev (Kafel BPF policy, GPU device binds).
- gVisor nvproxy: gvisor.dev/docs/user_guide/gpu/ (future upgrade path, driver coupling).
- NVIDIA MIG User Guide (hardware DRAM partitioning); MPS unsafe-for-untrusted rationale.
- cgroups v2: docs.kernel.org/admin-guide/cgroup-v2.html.
- `src/protean/bench_core.py`, `src/protean/anti_hack.py`, `src/protean/env.py`,
  `Dockerfile.hud`, `docs/PRODUCTION_READINESS.md`.
