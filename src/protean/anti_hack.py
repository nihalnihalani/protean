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
    "torch.nn.functional.relu",
}
BANNED_NAMES = {"eval", "exec", "compile", "__import__", "open", "getattr"}
BANNED_IMPORT_ROOTS = {"importlib", "subprocess", "os", "sys", "pathlib"}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def ast_clean(src: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in BANNED_CALLS:
                return False, f"ast_ban:{name}"
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"ast_ban:{node.func.id}"

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS:
                    return False, f"ast_ban:import:{alias.name}"

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_IMPORT_ROOTS:
                return False, f"ast_ban:import:{node.module}"

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
