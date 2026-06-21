"""The frozen registry of ~5 fixed fused ops Protean trains on.

See docs/IMPLEMENTATION_PLAN.md §4.1 / §4.6. Each OpSpec binds an eager reference,
dtype, allclose tolerances, and the speedup target/floor used by rewards.py.
"""
from __future__ import annotations  # NOTE: keep OUT of env.py (it breaks @env.template typed params) — fine here.
from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    op_name: str
    reference_module: str   # dotted path to hidden donotaccess/reference.py providing eager_fn(*tensors)
    dtype: str              # e.g. "float16"
    rtol: float
    atol: float
    p_target: float         # speedup at which the speedup-reward saturates (=1.0). plan: P_TARGET≈1.5
    speedup_floor: float    # below this, treat as no speedup. plan: 1.1
    prompt_path: str        # tasks/<op>/prompt.md


OPS = [
    OpSpec("elementwise_add_relu", "protean.tasks.elementwise_add_relu.donotaccess.reference",
           "float16", 1e-2, 1e-2, 1.5, 1.1, "tasks/elementwise_add_relu/prompt.md"),
]

OPS_BY_NAME = {o.op_name: o for o in OPS}
