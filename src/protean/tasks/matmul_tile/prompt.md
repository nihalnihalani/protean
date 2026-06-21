# Task: matmul tile Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(a, b):
    """Return a @ b where a is (16, K) and b is (K, 16), CUDA float16."""
```

Requirements:
- Match `torch.matmul(a, b)` within tolerance on fresh random inputs.
- Do not call PyTorch matrix multiply or other PyTorch compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out K value.
- Optimize for correctness first, then speed on train and held-out K sizes.

