# Task: row-wise softmax Triton kernel

Write a `@triton.jit` kernel in `/workdir/solution.py` exposing:

```python
def solution(x):
    """Return row-wise softmax of x. x is a CUDA float16 tensor of shape (M, N)."""
```

Requirements:
- The result must match `torch.softmax(x, dim=-1)` within tolerance on random inputs.
- Must be numerically stable (subtract row max before exp).
- It must be a real Triton kernel (calling `torch.softmax` to do the work earns zero reward).
- You will be graded on a withheld tensor shape you have not seen — do not hardcode N.
- Larger `N` may require tiled reduction (cannot fit the whole row in `BLOCK_N`).
- Use `/workdir/bench.py` to self-check compile, correctness, and latency.

You have multiple turns: edit `solution.py`, run `bench.py`, improve while staying correct.
