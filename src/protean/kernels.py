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


HAND_OPTIMIZED_SOFTMAX_ROWS = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    values = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(x):
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    block_size = _next_power_of_2(n_cols)
    _softmax_rows_kernel[(n_rows,)](x, out, n_cols, block_size=block_size)
    return out
'''


HAND_OPTIMIZED_MATMUL_TILE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_tile_kernel(a_ptr, b_ptr, out_ptr, k_size: tl.constexpr, block_k: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offsets = tl.arange(0, block_k)
    mask = offsets < k_size
    a = tl.load(a_ptr + row * k_size + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets * 16 + col, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(a * b, axis=0)
    tl.store(out_ptr + row * 16 + col, acc)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(a, b):
    out = torch.empty((16, 16), device=a.device, dtype=a.dtype)
    k_size = a.shape[1]
    block_k = _next_power_of_2(k_size)
    _matmul_tile_kernel[(16, 16)](a, b, out, k_size, block_k=block_k)
    return out
'''


HAND_OPTIMIZED_ATTENTION_SOFTMAX = HAND_OPTIMIZED_SOFTMAX_ROWS


HAND_OPTIMIZED_LAYERNORM = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr, eps: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_elements
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / n_elements
    out = centered * tl.rsqrt(variance + eps) * weight + bias
    tl.store(out_ptr + offsets, out, mask=mask)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(x, weight, bias):
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = _next_power_of_2(n_elements)
    _layernorm_kernel[(1,)](x, weight, bias, out, n_elements, block_size=block_size, eps=1e-5)
    return out
'''


HAND_OPTIMIZED_FUSED_MLP = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mlp_kernel(x_ptr, gate_ptr, bias_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    out = x * gate + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def solution(x, gate, bias):
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    _fused_mlp_kernel[(triton.cdiv(n_elements, block_size),)](x, gate, bias, out, n_elements, block_size=block_size)
    return out
'''


HAND_OPTIMIZED_QUANTIZE_DEQUANT = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _quant_dequant_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr, scale: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = x / scale
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    clipped = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    tl.store(out_ptr + offsets, clipped * scale, mask=mask)


def solution(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    _quant_dequant_kernel[(triton.cdiv(n_elements, block_size),)](x, out, n_elements, block_size=block_size, scale=0.1)
    return out
'''


HAND_OPTIMIZED_MOE_ROUTING = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_routing_kernel(logits_ptr, out_ptr, n_experts: tl.constexpr, block_size: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_experts
    logits = tl.load(logits_ptr + token * n_experts + offsets, mask=mask, other=-float("inf"))
    best = tl.max(logits, axis=0)
    tl.store(out_ptr + token, best)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(logits):
    n_tokens, n_experts = logits.shape
    out = torch.empty((n_tokens,), device=logits.device, dtype=logits.dtype)
    block_size = _next_power_of_2(n_experts)
    _moe_routing_kernel[(n_tokens,)](logits, out, n_experts, block_size=block_size)
    return out
'''


HAND_OPTIMIZED_EMBEDDING_LOOKUP = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_lookup_kernel(table_ptr, indices_ptr, out_ptr, dim: tl.constexpr, block_rows: tl.constexpr, block_dim: tl.constexpr):
    row_offsets = tl.arange(0, block_rows)
    dim_offsets = tl.arange(0, block_dim)
    token_mask = row_offsets < 64
    indices = tl.load(indices_ptr + row_offsets, mask=token_mask, other=0).to(tl.int64)
    values = tl.load(table_ptr + indices[:, None] * dim + dim_offsets[None, :], mask=token_mask[:, None])
    tl.store(out_ptr + row_offsets[:, None] * dim + dim_offsets[None, :], values, mask=token_mask[:, None])


def solution(table, indices):
    out = torch.empty((64, 32), device=table.device, dtype=table.dtype)
    _embedding_lookup_kernel[(1,)](table, indices, out, dim=32, block_rows=64, block_dim=32)
    return out
'''


HAND_OPTIMIZED_SUM_REDUCTION = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _sum_reduction_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr, total)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(x):
    out = torch.empty((1,), device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    block_size = _next_power_of_2(n_elements)
    _sum_reduction_kernel[(1,)](x, out, n_elements, block_size=block_size)
    return out
'''


HAND_OPTIMIZED_PREFIX_SCAN = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_scan_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, y, mask=mask)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def solution(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = _next_power_of_2(n_elements)
    _prefix_scan_kernel[(1,)](x, out, n_elements, block_size=block_size)
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

# Op-specific seed kernels for multi-op optimization
SEED_KERNELS = {
    "elementwise_add_relu": HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
    "rmsnorm": HAND_OPTIMIZED_RMSNORM,
    "softmax_rows": HAND_OPTIMIZED_SOFTMAX_ROWS,
    "matmul_tile": HAND_OPTIMIZED_MATMUL_TILE,
    "attention_softmax": HAND_OPTIMIZED_ATTENTION_SOFTMAX,
    "layernorm": HAND_OPTIMIZED_LAYERNORM,
    "fused_mlp": HAND_OPTIMIZED_FUSED_MLP,
    "quantize_dequant": HAND_OPTIMIZED_QUANTIZE_DEQUANT,
    "moe_routing": HAND_OPTIMIZED_MOE_ROUTING,
    "embedding_lookup": HAND_OPTIMIZED_EMBEDDING_LOOKUP,
    "sum_reduction": HAND_OPTIMIZED_SUM_REDUCTION,
    "prefix_scan": HAND_OPTIMIZED_PREFIX_SCAN,
}


def seed_kernel_for(op: str) -> str:
    """Return the best-known hand-optimized kernel for the given op."""
    return SEED_KERNELS.get(op, HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU)
