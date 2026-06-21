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


# Joint launch-config search space (KernelBand, arXiv:2511.18868).
# The deterministic sweep only touches block_size; the bandit path explores the
# full Cartesian product of these axes.
BLOCK_SIZES = (128, 256, 512, 1024, 2048)
NUM_WARPS = (4, 8, 16)
NUM_STAGES = (2, 3, 4)

# Matches a Triton kernel launch: ``_kernel[grid](arg0, arg1, ..., kw=...)``.
# Group 1 is the launch-argument body inside the parentheses.
_LAUNCH_CALL_RE = re.compile(r"(\w+\[[^\]]*\]\()(.*?)(\))", re.DOTALL)


def _replace_block_size(source: str, block_size: int) -> str:
    source = re.sub(r"triton\.cdiv\(n_elements,\s*\d+\)", f"triton.cdiv(n_elements, {block_size})", source)
    source = re.sub(r"block_size=\d+", f"block_size={block_size}", source)
    return source


def block_size_is_tunable(source: str) -> bool:
    """Whether ``_replace_block_size`` actually changes ``source``.

    The elementwise kernel pins a literal ``block_size=<int>`` that we can
    rewrite. The rmsnorm/softmax_rows seed kernels instead derive the block size
    from the data (``block_size = _next_power_of_2(n)``) because a single Triton
    block must span the full reduction; clamping it to an arbitrary literal would
    be *incorrect* (it would drop elements when the data dimension exceeds the
    literal). For those ops the block_size axis is intentionally inert, so the
    bandit should not pretend it is exploring it. Detect the literal-integer
    pattern directly rather than guessing from the op name.
    """

    return bool(
        re.search(r"block_size\s*=\s*\d+", source)
        or re.search(r"triton\.cdiv\(n_elements,\s*\d+\)", source)
    )


def _strip_launch_meta(body: str) -> str:
    """Remove any existing num_warps/num_stages kwargs from a launch body."""

    body = re.sub(r",\s*num_warps\s*=\s*\d+", "", body)
    body = re.sub(r",\s*num_stages\s*=\s*\d+", "", body)
    return body


def _replace_warps_and_stages(source: str, num_warps: int, num_stages: int) -> str:
    """Inject ``num_warps``/``num_stages`` launch kwargs into the kernel launch.

    Triton accepts these as launch-time meta parameters. We rewrite the first
    kernel launch found, stripping any pre-existing values so the edit is
    idempotent across rounds.
    """

    def _inject(match: re.Match) -> str:
        prefix, body, suffix = match.group(1), match.group(2), match.group(3)
        body = _strip_launch_meta(body)
        body = body.rstrip()
        return f"{prefix}{body}, num_warps={num_warps}, num_stages={num_stages}{suffix}"

    return _LAUNCH_CALL_RE.sub(_inject, source, count=1)


def make_config_edit(current_best: str, block_size: int, num_warps: int, num_stages: int) -> CandidateEdit:
    """Build one CandidateEdit for a point in the joint launch-config space."""

    source = _replace_block_size(current_best, block_size)
    source = _replace_warps_and_stages(source, num_warps, num_stages)
    return CandidateEdit(
        name=f"config_b{block_size}_w{num_warps}_s{num_stages}",
        reason=(
            f"Tune launch config: block_size={block_size}, "
            f"num_warps={num_warps}, num_stages={num_stages}."
        ),
        source=source,
        harness=harness_for_kernel_edit(),
        policy="bandit_config",
    )


def config_action_space(
    block_sizes: Iterable[int] = BLOCK_SIZES,
    num_warps: Iterable[int] = NUM_WARPS,
    num_stages: Iterable[int] = NUM_STAGES,
) -> list[tuple[int, int, int]]:
    """Enumerate the joint (block_size, num_warps, num_stages) arms."""

    arms: list[tuple[int, int, int]] = []
    for block_size in block_sizes:
        for warps in num_warps:
            for stages in num_stages:
                arms.append((block_size, warps, stages))
    return arms


def effective_action_space(
    source: str,
    block_sizes: Iterable[int] = BLOCK_SIZES,
    num_warps: Iterable[int] = NUM_WARPS,
    num_stages: Iterable[int] = NUM_STAGES,
) -> list[tuple[int, int, int]]:
    """Action space pruned to the axes that actually affect ``source``.

    When ``block_size`` is not a literal in the kernel launch (rmsnorm,
    softmax_rows), editing it is a no-op, so a 45-arm space would collapse to 9
    distinct kernels with 5x redundant arms. We instead pin the block size to a
    single representative value and enumerate only ``num_warps x num_stages``,
    so the bandit's logged arm count matches the number of *distinct* kernels it
    can actually produce. The elementwise kernel keeps the full Cartesian space.
    """

    if block_size_is_tunable(source):
        return config_action_space(block_sizes, num_warps, num_stages)
    pinned = next(iter(block_sizes))
    return [(pinned, warps, stages) for warps in num_warps for stages in num_stages]


def local_kernel_edits(current_best: str) -> Iterable[CandidateEdit]:
    """Edit the current best implementation instead of starting from scratch."""

    for block_size in BLOCK_SIZES:
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
