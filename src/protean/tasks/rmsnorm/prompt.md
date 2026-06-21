# Task: RMSNorm Triton kernel

Write a `@triton.jit` kernel in `/workdir/solution.py` exposing:

```python
def solution(x, weight):
    """Return RMSNorm(x) * weight for one CUDA float16 vector."""
```

Reference:

```python
x_f32 = x.float()
out = x_f32 * torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + 1e-5) * weight.float()
return out.to(x.dtype)
```

Requirements:
- The result must match the reference within tolerance on random inputs.
- It must be a real Triton kernel. Calling PyTorch to do the RMSNorm earns zero reward.
- You will be graded on withheld vector lengths you have not seen.
- Do not hardcode visible shapes.
