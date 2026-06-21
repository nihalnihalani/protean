# Task: attention softmax Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(scores):
    """Return row-wise attention probabilities for scores of shape (64, N)."""
```

Requirements:
- Match `torch.softmax(scores, dim=-1)` within tolerance.
- Use a numerically stable softmax by subtracting the row maximum.
- Do not call `torch.softmax` or PyTorch compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out N value.

