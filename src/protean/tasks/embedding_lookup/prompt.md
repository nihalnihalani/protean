# Task: embedding lookup Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(table, indices):
    """Return table[indices] where table is (V, 32) and indices has length 64."""
```

Requirements:
- Match PyTorch embedding gather within tolerance.
- Return shape `(64, 32)` with the same dtype as `table`.
- Do not call PyTorch indexing/gather compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vocabulary size.

