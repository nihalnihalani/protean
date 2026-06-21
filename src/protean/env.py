"""HUD template wrapper around the direct Protean grader."""

from __future__ import annotations

from protean.grader import grade_source, to_eval_result
from protean.splits import shapes_for_split
from protean.task_catalog import get_op

try:
    from hud import Environment
except Exception:  # pragma: no cover - local tests should not require HUD.
    Environment = None


PROMPT = """Write a Triton implementation of solution(x, y) for fused relu(x + y).

Requirements:
- x and y are CUDA float16 tensors with identical 1D shape.
- Return a tensor matching torch.relu(x + y).
- Use a real @triton.jit kernel. PyTorch passthrough earns zero reward.
- Do not hardcode the shape; held-out shapes are used for grading.
"""


env = Environment(name="protean") if Environment is not None else None


if env is not None:

    @env.template(name="elementwise_add_relu")
    async def kernel_opt(split: str = "train", shape: int | None = None):
        get_op("elementwise_add_relu")
        shape = shape or shapes_for_split(split)[0]
        source = yield PROMPT
        yield to_eval_result(grade_source(source or "", split=split, shape=shape))
