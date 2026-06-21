# Task: prefix scan Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(x):
    """Return cumsum(x) for one CUDA float16 vector."""
```

Requirements:
- Match `torch.cumsum(x.float(), dim=0).to(x.dtype)` within tolerance.
- Do not call PyTorch scan/reduction compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vector length.
- Optimize for correctness first, then speed.

