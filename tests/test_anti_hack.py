from protean.anti_hack import ast_clean

def test_ast_clean_valid():
    src = """import triton
import triton.language as tl

@triton.jit
def solution_kernel(x):
    pass
"""
    ok, why = ast_clean(src)
    assert ok
    assert why == ""

def test_ast_clean_invalid_call():
    src = """import torch
def solution(x):
    return torch.matmul(x, x)
"""
    ok, why = ast_clean(src)
    assert not ok
    assert "ast_ban:torch.matmul" in why

def test_ast_clean_invalid_import():
    src = """import importlib
"""
    ok, why = ast_clean(src)
    assert not ok
    assert "ast_ban:import:importlib" in why
