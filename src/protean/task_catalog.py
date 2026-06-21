"""Task catalog for Protean verifier tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    name: str
    dtype: str
    rtol: float
    atol: float
    p_target: float
    speedup_floor: float
    prompt_path: str


OPS = (
    OpSpec(
        name="elementwise_add_relu",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.5,
        speedup_floor=1.1,
        prompt_path="src/protean/tasks/elementwise_add_relu/prompt.md",
    ),
    OpSpec(
        name="rmsnorm",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.2,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/rmsnorm/prompt.md",
    ),
    OpSpec(
        name="softmax_rows",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.5,
        speedup_floor=1.1,
        prompt_path="src/protean/tasks/softmax_rows/prompt.md",
    ),
    OpSpec(
        name="matmul_tile",
        dtype="float16",
        rtol=2e-2,
        atol=2e-2,
        p_target=1.2,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/matmul_tile/prompt.md",
    ),
    OpSpec(
        name="attention_softmax",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.5,
        speedup_floor=1.1,
        prompt_path="src/protean/tasks/attention_softmax/prompt.md",
    ),
    OpSpec(
        name="layernorm",
        dtype="float16",
        rtol=2e-2,
        atol=2e-2,
        p_target=1.2,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/layernorm/prompt.md",
    ),
    OpSpec(
        name="fused_mlp",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.5,
        speedup_floor=1.1,
        prompt_path="src/protean/tasks/fused_mlp/prompt.md",
    ),
    OpSpec(
        name="quantize_dequant",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.4,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/quantize_dequant/prompt.md",
    ),
    OpSpec(
        name="moe_routing",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.3,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/moe_routing/prompt.md",
    ),
    OpSpec(
        name="embedding_lookup",
        dtype="float16",
        rtol=1e-2,
        atol=1e-2,
        p_target=1.2,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/embedding_lookup/prompt.md",
    ),
    OpSpec(
        name="sum_reduction",
        dtype="float16",
        rtol=2e-2,
        atol=2e-2,
        p_target=1.3,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/sum_reduction/prompt.md",
    ),
    OpSpec(
        name="prefix_scan",
        dtype="float16",
        rtol=2e-2,
        atol=2e-2,
        p_target=1.2,
        speedup_floor=1.05,
        prompt_path="src/protean/tasks/prefix_scan/prompt.md",
    ),
)

OPS_BY_NAME = {op.name: op for op in OPS}


def get_op(name: str) -> OpSpec:
    try:
        return OPS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown op: {name}") from exc
