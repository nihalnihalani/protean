# Task: MoE routing score Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(logits):
    """Return the top routing score per token for logits of shape (64, N)."""
```

Requirements:
- Match `torch.max(logits, dim=-1).values` within tolerance.
- One output value per token row.
- Do not call PyTorch max/reduction compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out expert count.

