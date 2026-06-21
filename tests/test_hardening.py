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


_TORCH_OPS_ATEN_DISPATCH = '''
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
'''


_TORCH_OPS_RSQRT_DISPATCH = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, weight):
    return torch.ops.aten.rsqrt.default(x)
'''


_TORCH_NN_FUNCTIONAL_CHAIN = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    fn = torch.nn.functional.relu
    return fn(x + y)
'''


_TRY_EXCEPT_TIMING_SHELL = '''
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
'''


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


_CTYPES_CUBLAS = '''
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
'''


_FUNCTIONAL_ALIASED_IMPORT = '''
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return F.relu(x + y)
'''


_FROM_TORCH_NN_IMPORT = '''
import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return F.relu(x + y)
'''


_FROM_TORCH_IMPORT_NN = '''
import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return nn.functional.relu(x + y)
'''


_NUMPY_HOST_ESCAPE = '''
import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return torch.from_numpy(np.add(x.cpu().numpy(), y.cpu().numpy())).to(x.device)
'''


_FROM_TORCH_OPS_IMPORT_ATEN = '''
import torch
import triton
import triton.language as tl
from torch.ops import aten


@triton.jit
def _decoy(x_ptr):
    return


def solution(x, y):
    return aten.add.default(x, y)
'''


_TRY_EXCEPT_RERAISE_OK = '''
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
'''


_TRY_EXCEPT_HELPER_SWALLOW = '''
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
'''


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
