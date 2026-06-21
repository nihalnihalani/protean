# Task: quantize-dequant Triton kernel

Write a `@triton.jit` kernel exposing:

```python
def solution(x):
    """Return dequantized int8-style values using scale 0.1."""
```

Requirements:
- Match `round(clamp(x / 0.1, -127, 127)) * 0.1` within tolerance.
- Return the same dtype and shape as `x`.
- Do not call PyTorch quantization or rounding compute in `solution`.
- Use a real Triton launch during timed execution.
- Do not hardcode the held-out vector length.

