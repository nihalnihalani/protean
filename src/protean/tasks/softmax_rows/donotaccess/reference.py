"""HIDDEN reference for softmax_rows — the allclose oracle + eager timing baseline.
Baked into the image as root:700; never visible to the agent. See IMPLEMENTATION_PLAN §4.3.
"""

import torch


def eager_fn(x: torch.Tensor) -> torch.Tensor:
    """Ground-truth: row-wise softmax over the last axis."""
    return torch.softmax(x, dim=-1)


def make_inputs(
    shape: int | tuple[int, ...],
    seed: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor]:
    """Fresh random inputs each grading call (defeats input-overfit).

    Note: softmax is invariant to translation, so we scale inputs by a small factor
    to keep them numerically well-behaved in fp16 (avoid all-zero exp underflow on
    extreme negative values).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype, device=device) * 2.0
    return (x,)  # single-arg tuple
