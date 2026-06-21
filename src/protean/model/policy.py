"""Kernel-edit policy for the coding agent.

This file is deliberately small. The first policy is deterministic so the
optimizer loop is testable; the next policy should call a model and emit the
same CandidateEdit records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from protean.model.harness import harness_for_kernel_edit


@dataclass(frozen=True)
class CandidateEdit:
    name: str
    reason: str
    source: str
    harness: dict[str, int]
    policy: str = "local_deterministic"
    model_cost_usd: float = 0.0
    tokens: int = 0


def _replace_block_size(source: str, block_size: int) -> str:
    source = re.sub(r"triton\.cdiv\(n_elements,\s*\d+\)", f"triton.cdiv(n_elements, {block_size})", source)
    source = re.sub(r"block_size=\d+", f"block_size={block_size}", source)
    return source


def local_kernel_edits(current_best: str) -> Iterable[CandidateEdit]:
    """Edit the current best implementation instead of starting from scratch."""

    for block_size in (128, 256, 512, 1024, 2048):
        yield CandidateEdit(
            name=f"block_size_{block_size}",
            reason=f"Retune Triton block size to {block_size}.",
            source=_replace_block_size(current_best, block_size),
            harness=harness_for_kernel_edit(),
        )
