# Task: sum reduction Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(x):
    """Return a float32 tensor of shape (1,) containing sum(x)."""
```

Requirements:
- Match `torch.sum(x.float()).reshape(1)` within tolerance.
- Accumulate in fp32.
- Do not call PyTorch reduction compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vector length.

