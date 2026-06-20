"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# anti_hack.py
import ast

BANNED_CALLS = {"torch.matmul", "torch.bmm", "torch.einsum", "F.softmax",
                "torch.compile", "F.linear", "F.conv2d", "torch.nn.functional"}
BANNED_NAMES = {"eval", "exec", "__import__", "compile"}
BANNED_IMPORTS = {"importlib"}

def ast_clean(src: str) -> tuple[bool, str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in BANNED_CALLS: return False, f"ast_ban:{name}"
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"ast_ban:{node.func.id}"
            # dynamic attr laundering: getattr(torch, ...)
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                return False, "ast_ban:getattr"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.module or "") if isinstance(node, ast.ImportFrom) else node.names[0].name
            if mod.split(".")[0] in BANNED_IMPORTS: return False, f"ast_ban:import:{mod}"
    return True, ""

def delegates_to_matrix_unit(src: str, op: str) -> bool:
    """Layer for matmul-class ops: ban tl.dot / tl.math.* matrix intrinsics so the agent
    must write the MAC loop itself (closes the delegating-wrapper exploit)."""
    if op not in MATMUL_CLASS_OPS:  # no-op for elementwise/reduction
        return False
    return ("tl.dot" in src) or ("tl.math" in src)
