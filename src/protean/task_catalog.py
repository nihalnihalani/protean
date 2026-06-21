"""Task catalog for the verifier-first MVP.

The first submission target is deliberately narrow: one fused op with a real
reference, a known-good Triton implementation, and train/held-out shapes.
"""

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
)

OPS_BY_NAME = {op.name: op for op in OPS}


def get_op(name: str) -> OpSpec:
    try:
        return OPS_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown op: {name}") from exc
