from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from protean import rewards
from protean.anti_hack import ast_clean
from protean.grader import grade_source


def test_rewards_hash_placeholder_present_for_docker_bake():
    placeholder = "<sha256 baked at image build>"
    paths = [
        Path("src/protean/tasks/elementwise_add_relu/donotaccess/grade.py"),
        Path("src/protean/tasks/rmsnorm/donotaccess/grade.py"),
        Path("train/calibrate.py"),
    ]
    for path in paths:
        assert placeholder in path.read_text(), f"{path} is not hash-bakeable"


def test_dockerfile_bakes_hash_and_builds_donotaccess_moat():
    dockerfile = Path("Dockerfile.hud").read_text()
    assert "sha256sum /app/src/protean/rewards.py" in dockerfile
    assert "sed -i" in dockerfile
    assert "mkdir -p /donotaccess" in dockerfile
    assert "cp /app/src/protean/rewards.py /donotaccess/rewards.py" in dockerfile
    assert "cp /app/src/protean/reward_config.json /donotaccess/reward_config.json" in dockerfile
    assert "chmod -R 700 /donotaccess" in dockerfile
    assert 'su agent -c "cat /donotaccess/rewards.py"' in dockerfile


def test_production_donotaccess_permissions_when_present():
    if not Path("/donotaccess").exists():
        pytest.skip("/donotaccess only exists inside the built HUD image")
    assert oct(Path("/donotaccess").stat().st_mode & 0o777) == "0o700"
    assert Path("/donotaccess/rewards.py").is_file()
    assert Path("/donotaccess/reward_config.json").is_file()
    assert Path("/donotaccess/elementwise_add_relu/grade.py").is_file()
    assert Path("/donotaccess/rmsnorm/grade.py").is_file()


def test_triton_cache_dir_matches_dockerfile_and_env():
    dockerfile = Path("Dockerfile.hud").read_text()
    docker_match = re.search(r"^ENV\s+TRITON_CACHE_DIR=(\S+)", dockerfile, re.MULTILINE)
    assert docker_match is not None

    env_py = Path("src/protean/env.py").read_text()
    assert docker_match.group(1) == "/triton-cache"
    assert 'Path("/triton-cache")' in env_py
    assert "protean-triton" in env_py


def test_reward_config_loads_from_local_dev_copy(tmp_path, monkeypatch):
    cfg_path = tmp_path / "reward_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "P_TARGET": 1.2,
                "SPEEDUP_FLOOR": 1.05,
                "SPEEDUP_CAP": 9.0,
                "CORRECT_FLOOR": 0.4,
                "SPEEDUP_REWARD_WEIGHT": 1.4,
                "PR_BONUS": 0.1,
                "MAX_REWARD": 1.8,
            }
        )
    )
    monkeypatch.setattr(rewards, "CANONICAL_CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(rewards, "LOCAL_CONFIG_PATH", cfg_path)

    config = rewards.load_reward_config()
    assert config.p_target == 1.2
    assert config.speedup_floor == 1.05
    assert config.speedup_reward_weight == 1.4
    assert config.pr_bonus == 0.1


def test_pr_frac_moves_reward():
    base = dict(correct=True, speedup=1.5, launches_timed=1, dtype_ok=True, shape_ok=True, split="held_out")
    low = rewards.compute_reward(**base, pr_frac=0.1)["reward"]
    high = rewards.compute_reward(**base, pr_frac=0.9)["reward"]
    assert high > low


# --- G5: hardened anti-hack AST bans (all CPU-only, no CUDA needed) ----------


_TORCH_OPS_ATEN_DISPATCH = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    # Delegates the real compute to the aten dispatcher, bypassing torch.relu /
    # torch.nn.functional.relu exact-match bans.
    return torch.ops.aten.relu.default(torch.ops.aten.add.default(x, y))
"""


_TORCH_OPS_RSQRT_DISPATCH = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, weight):
    return torch.ops.aten.rsqrt.default(x)
"""


_TORCH_NN_FUNCTIONAL_CHAIN = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    fn = torch.nn.functional.relu
    return fn(x + y)
"""


_TRY_EXCEPT_TIMING_SHELL = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    try:
        return _decoy(x)
    except Exception:
        return torch.empty_like(x)
"""


def test_torch_ops_aten_dispatch_banned_by_ast():
    ok, reason = ast_clean(_TORCH_OPS_ATEN_DISPATCH)
    assert ok is False
    assert reason.startswith("ast_ban:torch.ops")


def test_torch_ops_rsqrt_dispatch_banned_by_ast():
    ok, reason = ast_clean(_TORCH_OPS_RSQRT_DISPATCH)
    assert ok is False
    assert reason.startswith("ast_ban:torch.ops")


def test_torch_nn_functional_chain_banned_by_ast():
    # Referenced as a bare attribute (not a direct call target) - still banned.
    ok, reason = ast_clean(_TORCH_NN_FUNCTIONAL_CHAIN)
    assert ok is False
    assert reason.startswith("ast_ban:torch.nn.")


def test_try_except_timing_shell_banned_by_ast():
    ok, reason = ast_clean(_TRY_EXCEPT_TIMING_SHELL)
    assert ok is False
    assert reason == "ast_ban:try_except"


def test_legitimate_triton_kernel_not_banned_by_new_rules():
    from protean.kernels import (
        HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
        HAND_OPTIMIZED_RMSNORM,
        HAND_OPTIMIZED_SOFTMAX_ROWS,
    )

    for src in (
        HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
        HAND_OPTIMIZED_RMSNORM,
        HAND_OPTIMIZED_SOFTMAX_ROWS,
    ):
        ok, reason = ast_clean(src)
        assert ok is True, reason


def test_torch_ops_dispatch_rejected_through_grader():
    grade = grade_source(_TORCH_OPS_ATEN_DISPATCH)
    assert grade["reward"] == 0.0
    assert any(cap.startswith("ast_ban") for cap in grade["caps"])


def test_try_except_rejected_through_grader():
    grade = grade_source(_TRY_EXCEPT_TIMING_SHELL)
    assert grade["reward"] == 0.0
    assert "ast_ban:try_except" in grade["caps"]


def test_launch_counter_fallback_when_hooks_unavailable(monkeypatch):
    # When the Triton runtime hook surface is absent, _TritonLaunchCounter must
    # report available() == False so bench_source falls back to the source-grep
    # heuristic - never silently zero-launches a real kernel. This is CPU-safe:
    # we only construct the counter object, never run CUDA.
    import protean.bench_core as bench_core

    monkeypatch.setattr(bench_core, "triton", None)
    counter = bench_core._TritonLaunchCounter()
    assert counter.available() is False
    with counter as c:
        assert c.count == 0


# --- G5 hardening: delegation-via-import bypasses (CPU-only) -----------------


_CTYPES_CUBLAS = """
import torch
import triton
import triton.language as tl
import ctypes


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    lib = ctypes.cdll.LoadLibrary("libcublas.so")
    return x
"""


_FUNCTIONAL_ALIASED_IMPORT = """
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return F.relu(x + y)
"""


_FROM_TORCH_NN_IMPORT = """
import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return F.relu(x + y)
"""


_FROM_TORCH_IMPORT_NN = """
import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return nn.functional.relu(x + y)
"""


_NUMPY_HOST_ESCAPE = """
import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return torch.from_numpy(np.add(x.cpu().numpy(), y.cpu().numpy())).to(x.device)
"""


_FROM_TORCH_OPS_IMPORT_ATEN = """
import torch
import triton
import triton.language as tl
from torch.ops import aten


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return aten.add.default(x, y)
"""


_TRY_EXCEPT_RERAISE_OK = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    out = torch.empty_like(x)
    try:
        _decoy[(1,)](x)
    except Exception:
        raise
    return out
"""


_TRY_EXCEPT_HELPER_SWALLOW = """
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def _shape_fallback(x):
    try:
        return _decoy(x)
    except Exception:
        return torch.zeros_like(x)


def solution(x, y):
    return _shape_fallback(x)
"""


def test_ctypes_cublas_delegation_banned():
    ok, reason = ast_clean(_CTYPES_CUBLAS)
    assert ok is False
    assert reason == "ast_ban:import:ctypes"


def test_functional_aliased_import_banned():
    # `import torch.nn.functional as F; F.relu(...)` hides the call target behind
    # an alias the call-site prefix ban cannot see; the import statement is caught.
    ok, reason = ast_clean(_FUNCTIONAL_ALIASED_IMPORT)
    assert ok is False
    assert reason.startswith("ast_ban:import:torch.nn")


def test_from_torch_nn_import_functional_banned():
    ok, reason = ast_clean(_FROM_TORCH_NN_IMPORT)
    assert ok is False
    assert reason.startswith("ast_ban:import:torch.nn")


def test_from_torch_import_nn_banned():
    ok, reason = ast_clean(_FROM_TORCH_IMPORT_NN)
    assert ok is False
    assert reason == "ast_ban:import:torch.nn"


def test_numpy_host_escape_banned():
    ok, reason = ast_clean(_NUMPY_HOST_ESCAPE)
    assert ok is False
    assert reason == "ast_ban:import:numpy"


def test_from_torch_ops_import_aten_banned():
    ok, reason = ast_clean(_FROM_TORCH_OPS_IMPORT_ATEN)
    assert ok is False
    assert reason.startswith("ast_ban:import:torch.ops")


def test_try_except_reraise_is_allowed():
    # A handler that re-raises is benign error handling, not a timing-shell
    # exploit, and must NOT be banned (narrowed from the earlier blanket ban).
    ok, reason = ast_clean(_TRY_EXCEPT_RERAISE_OK)
    assert ok is True, reason


def test_try_except_swallow_in_helper_is_banned():
    # The swallow-and-return exploit is banned wherever it appears, including a
    # helper function called by solution() (not only directly inside solution).
    ok, reason = ast_clean(_TRY_EXCEPT_HELPER_SWALLOW)
    assert ok is False
    assert reason == "ast_ban:try_except"


def test_ctypes_delegation_rejected_through_grader():
    grade = grade_source(_CTYPES_CUBLAS)
    assert grade["reward"] == 0.0
    assert "ast_ban:import:ctypes" in grade["caps"]


def test_functional_aliased_import_rejected_through_grader():
    grade = grade_source(_FUNCTIONAL_ALIASED_IMPORT)
    assert grade["reward"] == 0.0
    assert any(cap.startswith("ast_ban:import:torch.nn") for cap in grade["caps"])


def test_launch_count_unverified_advisory_does_not_gate_reward():
    # On a Triton build without the runtime launch-hook surface, the fallback
    # must flag launch_count_unverified as an ADVISORY cap (not a reward-gating
    # cap), so a legit kernel is not zeroed merely because the count is
    # unverifiable. This is structural and CPU-safe: we exercise compute_reward
    # directly with a populated advisory cap kept out of the gating caps list.
    base = dict(correct=True, speedup=1.5, launches_timed=1, dtype_ok=True, shape_ok=True, split="held_out")
    # Advisory cap is intentionally NOT passed to compute_reward(caps=...).
    grade = rewards.compute_reward(**base, pr_frac=0.5, caps=None)
    assert grade["reward"] > 0.0
    assert grade["caps"] == []


# --- Production hardening: expanded import roots, relative-import ban, audit
# --- hook backstop, strict tensor-subclass identity, sys.modules tombstone -----


def _candidate_with_import(import_line: str) -> str:
    return (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        f"{import_line}\n\n\n"
        "@triton.jit\n"
        "def _decoy(x_ptr):\n"
        "    return\n\n\n"
        "def solution(x, y):\n"
        "    return x\n"
    )


@pytest.mark.parametrize(
    "root",
    [
        "threading",
        "concurrent",
        "multiprocessing",
        "_thread",
        "base64",
        "binascii",
        "tempfile",
        "shutil",
        "zipfile",
        "tarfile",
        "pickle",
        "shelve",
    ],
)
def test_expanded_banned_import_roots_rejected(root):
    # Concurrency family hides un-timed GPU work; binary-embedding family decodes
    # and dlopens a precompiled cubin; pickle/shelve close state-caching. Each new
    # root must be rejected by the static AST ban via a plain `import <root>`.
    src = _candidate_with_import(f"import {root}")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == f"ast_ban:import:{root}"


@pytest.mark.parametrize(
    "root",
    ["threading", "base64", "pickle", "concurrent", "multiprocessing"],
)
def test_expanded_banned_import_roots_rejected_via_from_import(root):
    # `from threading import Thread` / `from base64 import b64decode` must also be
    # caught: the ImportFrom path resolves node.module to the banned root.
    src = _candidate_with_import(f"from {root} import something")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == f"ast_ban:import:{root}"


def test_relative_import_banned():
    # Relative imports are abnormal in self-contained kernel code and could reach
    # the grader's own package namespace. Reject `from . import x`.
    src = _candidate_with_import("from . import sibling")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:relative_import"


def test_relative_import_with_package_banned():
    src = _candidate_with_import("from ..protean import rewards")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:relative_import"


def test_threading_delegation_rejected_through_grader():
    src = _candidate_with_import("import threading")
    grade = grade_source(src)
    assert grade["reward"] == 0.0
    assert "ast_ban:import:threading" in grade["caps"]


def test_audit_hook_installed_and_blocks_banned_import():
    # The PEP 578 audit hook is a defense-in-depth backstop for dynamic imports
    # that slip past static AST analysis. Importing bench_core installs it; the
    # hook raises ImportError for a banned root. CPU-safe: no CUDA used. We invoke
    # the hook directly (rather than via `import`) so the assertion does not depend
    # on whether a banned module happens to be cached from an earlier test, while
    # still exercising the exact code the interpreter calls on every import event.
    import sys

    import protean.bench_core as bench_core

    # The hook is armed lazily (at load_solution time) to avoid crashing the
    # interpreter bootstrap; install it explicitly here so the assertion does not
    # depend on whether an earlier test happened to call load_solution.
    from protean.anti_hack import set_audit_armed

    bench_core.install_audit_hook()
    assert getattr(sys, "_protean_audit_hook_installed", False) is True
    # The hook only enforces inside the candidate-exec window (set_audit_armed),
    # so the host process + torch threads run unimpeded outside it.
    set_audit_armed(True)
    try:
        for banned in ("base64", "threading", "pickle", "subprocess"):
            with pytest.raises(ImportError):
                bench_core._audit_import_hook("import", (banned, None, None, None))
        # dotted submodule of a banned module is also blocked
        with pytest.raises(ImportError):
            bench_core._audit_import_hook("import", ("torch.nn.functional", None, None, None))
    finally:
        set_audit_armed(False)
    # Dormant outside the window: a banned import does NOT raise (host/torch safe).
    bench_core._audit_import_hook("import", ("threading", None, None, None))


def test_audit_hook_allows_safe_import():
    # The backstop must not block a benign, allowed module, and must ignore
    # non-import audit events.
    import protean.bench_core as bench_core

    bench_core._audit_import_hook("import", ("json", None, None, None))
    bench_core._audit_import_hook("import", ("torch", None, None, None))
    bench_core._audit_import_hook("open", ("/tmp/x", "r", 0))  # non-import event ignored


def test_strict_tensor_identity_rejects_subclass():
    # type(x) is torch.Tensor must reject a subclass that passes isinstance. We
    # build a trivial torch.Tensor subclass; no CUDA is required to construct it.
    torch = pytest.importorskip("torch")
    import protean.bench_core as bench_core

    class _FakeTensor(torch.Tensor):
        pass

    real = torch.zeros(2)
    fake = torch.zeros(2).as_subclass(_FakeTensor)
    assert bench_core._is_strict_tensor(real) is True
    assert bench_core._is_strict_tensor(fake) is False
    assert isinstance(fake, torch.Tensor)  # confirms isinstance would have passed


def test_strict_tensor_identity_rejects_non_tensor():
    pytest.importorskip("torch")
    import protean.bench_core as bench_core

    assert bench_core._is_strict_tensor([1, 2, 3]) is False
    assert bench_core._is_strict_tensor(42) is False
    assert bench_core._is_strict_tensor(None) is False


def test_load_solution_tombstones_sys_modules(monkeypatch):
    # After load_solution returns, the candidate's module name must NOT remain in
    # sys.modules (so monkey-patches/globals cannot bleed into the next eval) and
    # its temp file must be removed. CPU-safe: we stub _require_torch and the
    # torch/triton globals so no CUDA is touched.
    import sys

    import protean.bench_core as bench_core

    monkeypatch.setattr(bench_core, "_require_torch", lambda: None)
    monkeypatch.setattr(bench_core, "torch", object())
    monkeypatch.setattr(bench_core, "triton", object())
    monkeypatch.setattr(bench_core, "tl", object())

    before = set(sys.modules)
    src = "def solution(x):\n    return x\n"
    module = bench_core.load_solution(src)
    assert module.solution("ok") == "ok"  # compiled function still callable

    new_candidate_names = [name for name in (set(sys.modules) - before) if name.startswith("protean_candidate_")]
    assert new_candidate_names == [], "candidate module leaked into sys.modules"


def test_legitimate_kernels_have_no_banned_imports():
    from protean.kernels import (
        HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
        HAND_OPTIMIZED_RMSNORM,
        HAND_OPTIMIZED_SOFTMAX_ROWS,
    )

    for src in (
        HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
        HAND_OPTIMIZED_RMSNORM,
        HAND_OPTIMIZED_SOFTMAX_ROWS,
    ):
        ok, reason = ast_clean(src)
        assert ok is True, reason


# --- Production hardening round 2: indirect-builtin reconstruction bypasses,
# --- io/builtins import roots, dotted-leaf dynamic-exec calls (all CPU-only) ---


def _candidate_with_body(body_line: str) -> str:
    return (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n\n"
        "@triton.jit\n"
        "def _decoy(x_ptr):\n"
        "    return\n\n\n"
        "def solution(x, y):\n"
        f"    {body_line}\n"
        "    return x\n"
    )


@pytest.mark.parametrize(
    "name",
    ["setattr", "globals", "locals", "vars", "__class__"],
)
def test_indirect_builtin_names_banned(name):
    # Without these in BANNED_NAMES a candidate can reconstruct any banned builtin
    # via `globals()["__builtins__"]["__import__"]("os")` or `setattr(...)` and slip
    # past the AST walk. Each must now be rejected as a bare-name call.
    src = _candidate_with_body(f"z = {name}()")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == f"ast_ban:{name}"


def test_globals_builtins_import_chain_banned():
    # The flagship sandbox-bypass vector. `globals()` is banned as a bare name; the
    # `["__builtins__"]` subscript is independently banned too. Either gate rejects.
    src = _candidate_with_body('globals()["__builtins__"]["__import__"]("os")')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason in ("ast_ban:globals", "ast_ban:builtins_subscript")


def test_builtins_subscript_eval_banned():
    src = _candidate_with_body('__builtins__["eval"]("1+1")')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:builtins_subscript"


def test_dunder_dict_subscript_banned():
    src = _candidate_with_body('torch.__dict__["__loader__"]')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:builtins_subscript"


def test_io_import_banned():
    # `io.open` is `builtins.open`; banning bare `open` without `io` left a direct
    # read of the hidden grader / reward config reachable.
    src = _candidate_with_import("import io")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:import:io"


def test_io_open_call_leaf_or_import_banned():
    # `io.open(...)` must be rejected -- either by the import-root ban or, if the
    # import were somehow hidden, by the io.open BANNED_CALLS entry.
    src = _candidate_with_body('z = io.open("/donotaccess/rewards.py")')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:io.open"


def test_builtins_import_root_banned():
    src = _candidate_with_import("import builtins")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:import:builtins"


def test_from_builtins_import_banned():
    src = _candidate_with_import("from builtins import __import__")
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason == "ast_ban:import:builtins"


def test_builtins_dotted_import_call_banned():
    # `builtins.__import__("os")` as a dotted call: bare-name BANNED_NAMES cannot
    # see it (node.func is an Attribute), but the dotted-leaf ban and the
    # builtins.__import__ prefix catch it.
    src = _candidate_with_body('z = builtins.__import__("os")')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason in ("ast_ban:builtins.__import__", "ast_ban:__import__")


def test_builtins_getattr_leaf_banned():
    # `builtins.getattr(x, "y")` reconstructs getattr past the bare-name ban; the
    # dotted-leaf ban catches getattr/setattr/eval/exec/__import__ regardless of
    # the receiving object.
    src = _candidate_with_body('z = builtins.getattr(torch, "relu")')
    ok, reason = ast_clean(src)
    assert ok is False
    assert reason in ("ast_ban:builtins.getattr", "ast_ban:getattr")


def test_globals_builtins_chain_rejected_through_grader():
    src = _candidate_with_body('globals()["__builtins__"]["__import__"]("os")')
    grade = grade_source(src)
    assert grade["reward"] == 0.0
    assert any(cap.startswith("ast_ban") for cap in grade["caps"])


def test_io_import_rejected_through_grader():
    src = _candidate_with_import("import io")
    grade = grade_source(src)
    assert grade["reward"] == 0.0
    assert "ast_ban:import:io" in grade["caps"]


def test_audit_hook_blocks_io_and_builtins_roots():
    # The audit-hook backstop mirrors the expanded import-root set: io and builtins
    # must be blocked at runtime too, not only statically.
    import protean.bench_core as bench_core
    from protean.anti_hack import set_audit_armed

    set_audit_armed(True)
    try:
        for banned in ("io", "builtins"):
            with pytest.raises(ImportError):
                bench_core._audit_import_hook("import", (banned, None, None, None))
    finally:
        set_audit_armed(False)


def test_audit_hook_lives_in_anti_hack_and_is_idempotent():
    # The hook now lives in anti_hack (imported eagerly by grader) and is armed
    # lazily by install_audit_hook at the load_solution chokepoint -- NOT at module
    # import, because a stdlib-root-banning hook installed during interpreter
    # bootstrap would crash on CPython's own importlib.machinery/os imports.
    # install_audit_hook must be idempotent and the hook must block banned roots.
    import sys

    import protean.anti_hack as anti_hack

    anti_hack.install_audit_hook()
    anti_hack.install_audit_hook()  # idempotent: second call is a no-op
    assert getattr(sys, "_protean_audit_hook_installed", False) is True
    anti_hack.set_audit_armed(True)
    try:
        with pytest.raises(ImportError):
            anti_hack._audit_import_hook("import", ("subprocess", None, None, None))
    finally:
        anti_hack.set_audit_armed(False)
