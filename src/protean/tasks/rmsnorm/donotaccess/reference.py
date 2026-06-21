"""HIDDEN reference for rmsnorm."""

import torch


def eager_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float()
    rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + 1e-5)
    return (x_f32 * rms * weight.float()).to(dtype=x.dtype)


def make_inputs(
    shape: int | tuple[int, ...],
    seed: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype, device=device)
    weight = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return (x, weight)
