"""HUD template wrapper around the direct Protean grader."""

from __future__ import annotations

from protean.grader import grade_source, to_eval_result
from protean.splits import shapes_for_split
from protean.task_catalog import get_op

try:
    from hud import Environment
except Exception:  # pragma: no cover - local tests should not require HUD.
    Environment = None


PROMPTS = {
    "elementwise_add_relu": """Write a Triton implementation of solution(x, y) for fused relu(x + y).

Requirements:
- x and y are CUDA float16 tensors with identical 1D shape.
- Return a tensor matching torch.relu(x + y).
- Use a real @triton.jit kernel. PyTorch passthrough earns zero reward.
- Do not hardcode the shape; held-out shapes are used for grading.
""",
    "rmsnorm": """Write a Triton implementation of solution(x, weight) for RMSNorm.

Requirements:
- x and weight are CUDA float16 tensors with identical 1D shape.
- Return x * rsqrt(mean(x^2) + 1e-6) * weight (computed in float32 for stability).
- The result must match the PyTorch reference within tolerance.
- Use a real @triton.jit kernel. PyTorch passthrough earns zero reward.
- Do not hardcode the shape; held-out shapes are used for grading.
""",
}


env = Environment(name="protean") if Environment is not None else None


if env is not None:

    @env.template(id="elementwise_add_relu")
    async def elementwise_add_relu(split: str = "train", shape: int | None = None):
        get_op("elementwise_add_relu")
        shape = shape or shapes_for_split(split)[0]
        source = yield PROMPTS["elementwise_add_relu"]
        yield to_eval_result(grade_source(source or "", op="elementwise_add_relu", split=split, shape=shape))

    @env.template(id="rmsnorm")
    async def rmsnorm(split: str = "train", shape: int | None = None):
        get_op("rmsnorm")
        shape = shape or shapes_for_split(split)[0]
        source = yield PROMPTS["rmsnorm"]
        yield to_eval_result(grade_source(source or "", op="rmsnorm", split=split, shape=shape))

    # Expose runnable task instances for hud eval discovery
    t_ew_train = elementwise_add_relu(split="train")
    t_ew_held = elementwise_add_relu(split="held_out")
    t_rms_train = rmsnorm(split="train")
    t_rms_held = rmsnorm(split="held_out")
