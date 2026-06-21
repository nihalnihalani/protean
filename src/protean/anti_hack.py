"""Static checks for obvious verifier bypass attempts."""

from __future__ import annotations

import ast
import sys

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
    # Dotted aliases of banned builtins reached through the io/builtins modules.
    # `io.open` is `builtins.open`; `builtins.getattr`/`builtins.eval` reconstruct
    # the bare builtins past the BANNED_NAMES check (which only sees ast.Name
    # targets). The attribute-leaf check below catches these generically, but the
    # explicit entries keep the reported reason stable and self-documenting.
    "io.open",
    "builtins.open",
    "builtins.getattr",
    "builtins.setattr",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__import__",
    "builtins.vars",
    "builtins.globals",
    "builtins.locals",
}
# Bare-builtin call bans. eval/exec/compile/__import__/open/getattr are the
# classic dynamic-execution and file-access primitives. setattr/globals/locals/
# vars/__builtins__/__class__ close the indirect-builtin-reconstruction bypass:
# without them a candidate could write `globals()["__builtins__"]["__import__"]("os")`
# or `setattr(mod, "attr", bad_fn)` and slip past the AST walk entirely. The
# longer-term hard boundary is subprocess isolation (KernelGym / SOL-ExecBench
# arXiv:2603.19173), but each name here is a zero-cost AST string-set lookup that
# closes a today-exploitable static gap. Source: IMPROVEMENT_RESEARCH.md item #14.
BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "vars",
    "__builtins__",
    "__class__",
}

# Import-root bans block whole packages that let a candidate delegate the real
# computation off the @triton.jit path. ctypes/cffi can dlopen libcublas.so and
# call cuBLAS directly; cupy/pycuda/cuda are GPU compute backends; numpy is a
# host-compute escape. importlib/subprocess/os/sys/pathlib are process/FS
# escapes. Source: GAP_ANALYSIS G5; CUDA-Agent / SOL-ExecBench (arXiv 2603.19173).
#
# The concurrency family (threading/concurrent/multiprocessing/_thread) maps to
# the "concurrency exploit" category in SOL-ExecBench Table 3: hiding GPU work on
# un-timed Python threads or via torch.jit.fork (call-site already prefix-banned,
# but the import was not). The binary-embedding family
# (base64/binascii/tempfile/shutil/zipfile/tarfile) is used to decode and dlopen
# a precompiled cubin, bypassing the @triton.jit requirement entirely.
# pickle/shelve close the state-caching deserialization angle. Each entry is a
# pure AST string-set lookup with zero runtime cost.
# Source: SOL-ExecBench arXiv:2603.19173 Table 3; IMPROVEMENT_RESEARCH.md item #3.
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
    # `io` is the stdlib file-access alias around `open` (io.open == builtins.open),
    # so banning bare `open` without `io` left a direct read of the hidden grader /
    # reward config (/donotaccess) reachable on a misconfigured dev host. `builtins`
    # exposes __import__/getattr/eval as attributes (builtins.__import__("os")), which
    # the bare-name BANNED_NAMES check cannot see; banning the import root closes that
    # reconstruction path. Source: IMPROVEMENT_RESEARCH.md item #14.
    "io",
    "builtins",
    # concurrency exploit family (un-timed off-thread GPU work)
    "threading",
    "concurrent",
    "multiprocessing",
    "_thread",
    "asyncio",
    # network / exfiltration family (a kernel must never open a socket; closes the
    # data-exfiltration + fetch-precompiled-cubin-over-network vectors). SOL-ExecBench
    # arXiv:2603.19173 network category; devil's-advocate follow-up (socket/urllib were open).
    "socket",
    "urllib",
    "urllib3",
    "http",
    "httplib",
    "ftplib",
    "smtplib",
    "telnetlib",
    "requests",
    "ssl",
    # binary-embedding family (decode + dlopen a precompiled cubin)
    "base64",
    "binascii",
    "tempfile",
    "shutil",
    "zipfile",
    "tarfile",
    # state-caching / deserialization family
    "pickle",
    "shelve",
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
    "builtins.__import__",
)

# Leaf attribute names that are dangerous regardless of the object they are
# called on. The bare-name BANNED_NAMES check only fires when the call target is
# an ast.Name (`getattr(...)`); a dotted call like `builtins.getattr(...)` or
# `mod.__import__(...)` is an ast.Attribute and slips through. We ban the leaf
# (`node.func.attr`) for these dynamic-execution / file-access primitives. The
# set is deliberately narrow -- e.g. `compile` is excluded because `torch.compile`
# is already handled by BANNED_CALLS and a legitimate method named `compile` on a
# user object would false-positive; the genuinely exploitable leaves are the
# import/exec/eval/builtins-reconstruction ones.
BANNED_CALL_LEAVES = {
    "eval",
    "exec",
    "__import__",
    "getattr",
    "setattr",
}


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
    return any(isinstance(child, ast.Return) for child in ast.walk(node))


def ast_clean(src: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"
    except (ValueError, TypeError) as exc:
        # ast.parse raises ValueError on a NUL byte ("source code string cannot
        # contain null bytes") and TypeError on non-str input; fail closed rather
        # than letting the exception propagate out of the verifier boundary.
        return False, f"syntax_error:{exc}"
    except (RecursionError, MemoryError):
        # Pathological deeply-nested / oversized inputs can exhaust the C stack or
        # memory inside ast.parse. Treat them as invalid, never as a pass.
        return False, "syntax_error:input_too_complex"

    for node in ast.walk(tree):
        # Ban only the swallow-and-return try/except timing-shell exploit; allow
        # benign handlers that re-raise. See _handler_swallows for rationale.
        if isinstance(node, ast.ExceptHandler) and _handler_swallows(node):
            return False, "ast_ban:try_except"

        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in BANNED_CALLS:
                return False, f"ast_ban:{name}"
            if name and any(name.startswith(p) for p in BANNED_PREFIXES):
                return False, f"ast_ban:{name}"
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"ast_ban:{node.func.id}"
            # Dotted call whose LEAF is a dynamic-execution primitive, e.g.
            # `builtins.getattr(...)`, `mod.__import__(...)`, `x.eval(...)`. The
            # bare-name check above cannot see these because node.func is an
            # ast.Attribute, not an ast.Name.
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_CALL_LEAVES:
                return False, f"ast_ban:{node.func.attr}"

        # Subscript reconstruction of banned builtins, e.g.
        # `__builtins__["eval"]("...")` or `something.__dict__["__import__"]`. The
        # call target there is an ast.Subscript (not a Name/Attribute), so the
        # call-site checks miss it; we flag any subscript whose base resolves to
        # __builtins__ or ends in __dict__.
        if isinstance(node, ast.Subscript):
            base = _dotted(node.value)
            if base == "__builtins__" or base.endswith("__dict__") or base.endswith(".__builtins__"):
                return False, "ast_ban:builtins_subscript"

        # Default-argument capture of a banned dynamic-execution primitive, e.g.
        # `lambda v=eval: v(...)` or `def g(f=getattr): ...`. The banned name is an
        # ast.Name in a default *value* slot, never an ast.Call target, so every
        # call-site check above misses it -- yet at runtime the bound default lets
        # the body invoke eval/exec/__import__ indirectly through the parameter.
        # We walk the defaults / kw_defaults of every function and lambda and reject
        # any default expression that references a BANNED_NAMES entry.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                for sub in ast.walk(default):
                    if isinstance(sub, ast.Name) and sub.id in BANNED_NAMES:
                        return False, f"ast_ban:default_capture:{sub.id}"

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
            # Relative imports (`from . import x`, `from ..pkg import y`) are
            # abnormal in self-contained generated kernel code and could reach
            # the grader's own package namespace. Reject them outright.
            if node.level and node.level > 0:
                return False, "ast_ban:relative_import"
            if _module_banned(module):
                return False, f"ast_ban:import:{module}"
            # `from torch import nn` / `from torch import ops`: the submodule is
            # named in the import targets, not in node.module.
            if module == "torch":
                for alias in node.names:
                    if _module_banned(f"torch.{alias.name}"):
                        return False, f"ast_ban:import:torch.{alias.name}"

    return True, ""


def _audit_import_hook(event: str, args: tuple) -> None:
    """PEP 578 audit hook that blocks runtime imports of banned modules.

    Defense-in-depth backstop for the AST static ban: it catches dynamic-import
    bypasses that slip past static analysis (e.g. ``__import__("o" + "s")``
    string-concat obfuscation, or an import buried in a helper the AST walk did
    not reason about). It mirrors BANNED_IMPORT_ROOTS / BANNED_IMPORT_MODULES so
    the two stay in lockstep.

    This hook lives in anti_hack (not bench_core) on purpose: anti_hack is
    imported eagerly by grader at module top, whereas bench_core is imported
    lazily only behind the CUDA guard. Installing the hook here means it is armed
    on every code path that touches the grader -- including CPU-only entry points
    that never trigger a CUDA eval -- closing the "hook installed too late"
    window. The hard isolation boundary is still subprocess + RLIMIT (KernelGym),
    which is GPU-blocked here.

    CRITICAL caveat: audit hooks added from Python (via sys.addaudithook, not the
    C API PySys_AddAuditHook before Py_Initialize) CAN be bypassed by adversarial
    code that reaches the C layer directly -- the CPython docs and cpython issue
    #87604 are explicit about this. Treat this as defense-in-depth against naive
    runtime bypasses, NOT as a hard security boundary.
    Source: PEP 578; CPython issue #87604; IMPROVEMENT_RESEARCH.md item #14.
    """
    if event != "import":
        return
    module = args[0] if args else ""
    if not module:
        return
    if _module_banned(module):
        raise ImportError(f"Blocked import (audit hook): {module}")


def install_audit_hook() -> None:
    """Install the import audit hook once. Idempotent across repeated imports.

    Call this at the single chokepoint where untrusted candidate code is about to
    be executed (load_solution), NOT eagerly at module-import time. An import
    audit hook that bans stdlib roots (importlib/os/sys) cannot be armed during
    interpreter bootstrap: CPython itself lazily imports importlib.machinery,
    os, etc. while loading our own dependency tree, and a process-wide hook armed
    too early would raise ImportError on those legitimate host imports and crash
    the program. Arming it immediately before candidate exec is both correct (the
    host's own imports have already completed) and sufficient (no candidate runs
    before load_solution). Source: PEP 578; CPython issue #87604;
    IMPROVEMENT_RESEARCH.md item #14.
    """
    if not getattr(sys, "_protean_audit_hook_installed", False):
        sys.addaudithook(_audit_import_hook)
        sys._protean_audit_hook_installed = True  # type: ignore[attr-defined]


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
