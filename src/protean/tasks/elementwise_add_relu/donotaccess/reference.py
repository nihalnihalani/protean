"""HIDDEN reference for elementwise_add_relu — the allclose oracle + eager timing baseline.
Baked into the image as root:700; never visible to the agent. See IMPLEMENTATION_PLAN §4.3.
"""
import torch


def eager_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ground-truth: relu(x + y)."""
    return torch.relu(x + y)


def make_inputs(shape, seed: int, dtype=torch.float16, device="cuda"):
    """Fresh random inputs each grading call (defeats input-overfit)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype, device=device)
    y = torch.randn(shape, generator=g, dtype=dtype, device=device)
    return (x, y)
