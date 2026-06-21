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
from protean.model.tiny_policy import ACTION_BLOCK_SIZES, TinyPolicyHead, action_index, state_features


@dataclass(frozen=True)
class CandidateEdit:
    name: str
    reason: str
    source: str
    harness: dict[str, int]
    policy: str = "local_deterministic"
    model_cost_usd: float = 0.0
    tokens: int = 0
    pricing_miss: bool = False


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


def learned_kernel_edits(current_best: str, best_state: dict | tuple[float, float, int], policy_path: str) -> Iterable[CandidateEdit]:
    """Order deterministic edits with a trained policy head."""

    policy = TinyPolicyHead.load(policy_path)
    edits = list(local_kernel_edits(current_best))
    by_action = {action_index(edit.name): edit for edit in edits}
    for action in policy.ranked_actions(state_features(best_state)):
        edit = by_action.get(action)
        if edit is not None:
            block_size = ACTION_BLOCK_SIZES[action]
            yield CandidateEdit(
                name=edit.name,
                reason=f"Learned policy head selected block size {block_size}.",
                source=edit.source,
                harness=edit.harness,
                policy="tiny_policy_head",
                model_cost_usd=0.0,
                tokens=0,
            )
