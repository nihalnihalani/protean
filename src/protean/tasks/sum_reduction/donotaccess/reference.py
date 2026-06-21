"""Hidden reference adapter for Protean op tasks."""

from __future__ import annotations

import os

from protean.bench_core import eager_fn_for_op, make_inputs as _make_inputs


OP_NAME = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
eager_fn = eager_fn_for_op(OP_NAME)


def make_inputs(shape, seed: int, dtype="float16", device="cuda"):
    n = shape[-1] if isinstance(shape, tuple) else int(shape)
    dtype_name = getattr(dtype, "__name__", None) or str(dtype).replace("torch.", "")
    return _make_inputs(n, dtype_name, seed, OP_NAME)

