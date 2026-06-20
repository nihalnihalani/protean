"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# subprocess_runner.py — fail-closed, restricted globals
def run_bench(op, M, N, dtype, src, timeout_s=90):
    ok, why = ast_clean(src)
    if not ok: return {"status": why}
    if delegates_to_matrix_unit(src, op): return {"status": "delegating_wrapper"}
    # spawn subprocess; SIGKILL process group on timeout (verilog run() pattern)
    # inside child: exec(compile(src,...), {"triton":triton,"tl":tl,"torch":torch_ro,
    #                                       "__builtins__": SAFE_BUILTINS})
    # SAFE_BUILTINS excludes __import__/eval/exec/open
    ...
