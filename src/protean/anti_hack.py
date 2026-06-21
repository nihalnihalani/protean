"""Static checks for obvious verifier bypass attempts."""

from __future__ import annotations

import ast

BANNED_CALLS = {
    "torch.add",
    "torch.relu",
    "torch.maximum",
    "torch.clamp",
    "torch.compile",
    "torch.matmul",
    "torch.bmm",
    "torch.einsum",
    "torch.mean",
    "torch.rsqrt",
    "torch.sum",
    "torch.nn.functional.relu",
}
BANNED_NAMES = {"eval", "exec", "compile", "__import__", "open", "getattr"}

# Import-root bans block whole packages that let a candidate delegate the real
# computation off the @triton.jit path. ctypes/cffi can dlopen libcublas.so and
# call cuBLAS directly; cupy/pycuda/cuda are GPU compute backends; numpy is a
# host-compute escape. importlib/subprocess/os/sys/pathlib are process/FS
# escapes. Source: GAP_ANALYSIS G5; CUDA-Agent / SOL-ExecBench (arXiv 2603.19173).
BANNED_IMPORT_ROOTS = {
    "importlib",
    "subprocess",
    "os",
    "sys",
    "pathlib",
    "ctypes",
    "cffi",
    "cupy",
    "pycuda",
    "cuda",
    "numpy",
}

# Import-module bans are matched as dotted-path prefixes against the FULL module
# path of an import, not just its root. They close the aliased-import bypass that
# the call-site prefix ban cannot see: `import torch.nn.functional as F` followed
# by `F.relu(x + y)` resolves the call target to "F.relu" (not the fully-spelled
# "torch.nn.functional.relu"), so the call-site ban misses it -- but the import
# statement still names torch.nn.* and is caught here. Likewise
# `from torch.ops import aten`. Source: GAP_ANALYSIS G5.
BANNED_IMPORT_MODULES = (
    "torch.nn",
    "torch.ops",
    "torch.jit",
    "torch._C",
)

# Prefix-based bans catch deep dispatch chains that exact-match cannot enumerate,
# e.g. torch.ops.aten.add.default(...), torch.ops.aten.rsqrt.default(...),
# torch.nn.functional.*, torch.jit.fork(...), torch.cuda.Stream(...). LLMs use
# these to delegate the real computation to a fused PyTorch/cuBLAS op while
# keeping a decorative @triton.jit function around to satisfy the launch check.
# Source: Dr. Kernel (arXiv 2602.05885), CUDA-Agent, GAP_ANALYSIS G5.
BANNED_PREFIXES = (
    "torch.ops",
    "torch.nn.",
    "torch.jit.fork",
    "torch.cuda.Stream",
    "torch._C",
)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _module_banned(module: str) -> bool:
    """True if an imported module path is itself banned (root or dotted prefix)."""
    if not module:
        return False
    root = module.split(".")[0]
    if root in BANNED_IMPORT_ROOTS:
        return True
    return any(module == m or module.startswith(m + ".") for m in BANNED_IMPORT_MODULES)


def _handler_swallows(node: ast.ExceptHandler) -> bool:
    """True if an except handler silently returns a value (the timing-shell exploit).

    The documented exploit (CUDA-Agent / SOL-ExecBench, arXiv 2603.19173) wraps
    the real compute in try/except and, when timing is about to be measured,
    silently falls back to a no-op / cached tensor via a `return` in the handler
    body. We ban exactly that shape -- a handler whose body (or any nested block
    of it) reaches a `return`. A handler that only re-raises (bare `raise` or
    `raise exc`) or re-logs is benign error handling and is allowed, so a kernel
    may legitimately validate shapes and re-raise. This narrows an earlier
    blanket ban on all try/except that produced false positives.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            return True
    return False


def ast_clean(src: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"

    for node in ast.walk(tree):
        # Ban only the swallow-and-return try/except timing-shell exploit; allow
        # benign handlers that re-raise. See _handler_swallows for rationale.
        if isinstance(node, ast.ExceptHandler):
            if _handler_swallows(node):
                return False, "ast_ban:try_except"

        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in BANNED_CALLS:
                return False, f"ast_ban:{name}"
            if name and any(name.startswith(p) for p in BANNED_PREFIXES):
                return False, f"ast_ban:{name}"
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"ast_ban:{node.func.id}"

        # Also catch banned dispatch chains referenced as bare attributes (not
        # only as the call target), e.g. handing torch.ops.aten.add to a helper.
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name and any(name.startswith(p) for p in BANNED_PREFIXES):
                return False, f"ast_ban:{name}"

        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_banned(alias.name):
                    return False, f"ast_ban:import:{alias.name}"

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _module_banned(module):
                return False, f"ast_ban:import:{module}"
            # `from torch import nn` / `from torch import ops`: the submodule is
            # named in the import targets, not in node.module.
            if module == "torch":
                for alias in node.names:
                    if _module_banned(f"torch.{alias.name}"):
                        return False, f"ast_ban:import:torch.{alias.name}"

    return True, ""


def contains_triton_jit(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if _dotted(decorator) in {"triton.jit", "jit"}:
                return True
            if isinstance(decorator, ast.Call) and _dotted(decorator.func) in {"triton.jit", "jit"}:
                return True
    return False
