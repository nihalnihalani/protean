"""Candidate kernels used by smoke tests and the guaranteed demo path."""

HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def solution(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _add_relu_kernel[grid](x, y, out, n_elements, block_size=1024)
    return out
'''


PYTORCH_PASSTHROUGH = r'''
import torch


def solution(x, y):
    return torch.relu(x + y)
'''


NO_LAUNCH = r'''
import torch


def solution(x, y):
    return torch.empty_like(x)
'''


BAD_SHAPE_TRITON = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _noop_kernel(x_ptr, out_ptr):
    return


def solution(x, y):
    out = torch.empty((1,), device=x.device, dtype=x.dtype)
    _noop_kernel[(1,)](x, out)
    return out
'''
