# Task: fused MLP elementwise Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(x, gate, bias):
    """Return relu(x * gate + bias) for CUDA float16 vectors."""
```

Requirements:
- Match `torch.relu(x * gate + bias)` within tolerance.
- Fuse multiply, add, and ReLU into one Triton kernel.
- Do not call PyTorch compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vector length.

