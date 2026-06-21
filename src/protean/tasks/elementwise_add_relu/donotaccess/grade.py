"""Hidden-grade compatible wrapper for the elementwise_add_relu MVP."""

from __future__ import annotations

from pathlib import Path

from protean.grader import grade_source


def grade_kernel(op, shape, dtype, kernel_src, split="held_out", step=0):
    return grade_source(kernel_src, op=op, split=split, shape=int(shape))


def grade(workdir, override=None, hidden_root=None):
    path = Path(workdir) / "solution.py"
    source = override if override is not None else path.read_text()
    return grade_source(source, op="elementwise_add_relu", split="held_out")
