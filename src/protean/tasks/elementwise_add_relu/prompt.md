# Task: fused `add + ReLU` Triton kernel

Write a `@triton.jit` kernel in `/workdir/solution.py` exposing:

```python
def solution(x, y):
    """Return relu(x + y), elementwise. x, y are CUDA float16 tensors of identical shape."""
```

Requirements:
- The result must match `torch.relu(x + y)` within tolerance on **random inputs**.
- It must be a real Triton kernel (calling `torch` to do the work earns **zero** reward).
- You will be graded on a **withheld tensor shape you have not seen** — do not hardcode shapes or sizes.
- Use `/workdir/bench.py` to self-check compile, correctness, and latency on the visible (training) shapes.

You have multiple turns: edit `solution.py`, run `bench.py`, and improve speed while staying correct.
