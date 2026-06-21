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


HAND_OPTIMIZED_RMSNORM = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr, eps: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_elements
    rstd = tl.rsqrt(mean_square + eps)
    out = x * rstd * weight
    tl.store(out_ptr + offsets, out, mask=mask)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(x, weight):
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = _next_power_of_2(n_elements)
    _rmsnorm_kernel[(1,)](x, weight, out, n_elements, block_size=block_size, eps=1e-5)
    return out
'''


PYTORCH_PASSTHROUGH = r'''
import torch


def solution(x, y):
    return torch.relu(x + y)
'''


PYTORCH_RMSNORM_PASSTHROUGH = r'''
import torch


def solution(x, weight):
    x_f32 = x.float()
    return (x_f32 * torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + 1e-5) * weight.float()).to(x.dtype)
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
