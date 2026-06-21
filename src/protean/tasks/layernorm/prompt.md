# Task: layernorm Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(x, weight, bias):
    """Return layernorm(x) * weight + bias for one CUDA float16 vector."""
```

Requirements:
- Match a PyTorch fp32-accumulation layernorm reference within tolerance.
- Use epsilon `1e-5`.
- Do not call PyTorch normalization or reduction compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vector length.

