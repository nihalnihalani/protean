"""Zero-dependency LLM cost accounting for Fireworks calls.

Token counts already come back from call_fireworks_chat() in the API usage
field; this module turns those counts into a USD figure using per-million
prices stored in configs/fireworks_pricing.json.

Pricing numbers are VENDOR-STATED (fireworks.ai / docs.fireworks.ai, fetched
2026-06-20) and should be re-verified against an actual invoice before any
cost figure is presented as authoritative.

Two concerns are handled explicitly here so they are not silent:

1. Cached prompt tokens. Fireworks' OpenAI-compatible API can report
   ``usage.prompt_tokens_details.cached_tokens``; those tokens are billed at
   ``cached_input_usd_per_million`` (when present in the config) instead of the
   full ``input_usd_per_million`` rate. Without this, a long-context repeated
   prompt would be over-charged. The config field is therefore load-bearing,
   not decorative.

2. Pricing misses. If a model is absent from the pricing config the cost is
   0.0 (a safe, non-raising default that matches the historical behaviour),
   but the *fact* that pricing was missing is reported separately via
   ``compute_fireworks_cost_detailed`` so a 0.0 cost cannot be mistaken for a
   genuinely free run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PRICING_PATH = Path(__file__).parent / "configs" / "fireworks_pricing.json"


def load_pricing() -> dict[str, dict[str, float]]:
    """Load the Fireworks pricing config from disk (uncached, always fresh)."""

    return json.loads(_PRICING_PATH.read_text())


@lru_cache(maxsize=1)
def _cached_pricing() -> dict[str, dict[str, float]]:
    """Process-lifetime cached pricing table for hot optimizer loops."""

    return load_pricing()


@dataclass(frozen=True)
class CostBreakdown:
    """Result of pricing a single Fireworks call."""

    cost_usd: float
    pricing_miss: bool


def _compute(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    table: dict[str, dict[str, float]],
) -> CostBreakdown:
    entry = table.get(model)
    if entry is None:
        return CostBreakdown(cost_usd=0.0, pricing_miss=True)

    prompt = int(prompt_tokens)
    completion = int(completion_tokens)
    # cached_tokens are a subset of prompt_tokens; clamp so we never bill
    # negative uncached tokens if the API ever reports cached > prompt.
    cached = max(0, min(int(cached_tokens), prompt))
    uncached_prompt = prompt - cached

    input_rate = entry["input_usd_per_million"]
    output_rate = entry["output_usd_per_million"]
    cached_rate = entry.get("cached_input_usd_per_million", input_rate)

    cost = (
        uncached_prompt * input_rate / 1_000_000
        + cached * cached_rate / 1_000_000
        + completion * output_rate / 1_000_000
    )
    return CostBreakdown(cost_usd=round(cost, 8), pricing_miss=False)


def compute_fireworks_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """Return the USD cost for a single Fireworks call.

    Returns 0.0 if the model is not present in the pricing config (safe
    default, never raises), matching the previous behaviour where
    model_cost_usd defaulted to 0.0. Use ``compute_fireworks_cost_detailed``
    if you need to know whether a pricing miss occurred.

    ``cached_tokens`` (a subset of ``prompt_tokens``) are billed at the
    model's ``cached_input_usd_per_million`` rate when that field is present.
    """

    table = pricing if pricing is not None else _cached_pricing()
    return _compute(model, prompt_tokens, completion_tokens, cached_tokens, table).cost_usd


def compute_fireworks_cost_detailed(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    pricing: dict[str, dict[str, float]] | None = None,
) -> CostBreakdown:
    """Like ``compute_fireworks_cost`` but also reports a pricing miss.

    ``pricing_miss`` is True exactly when ``model`` is absent from the pricing
    table, so a $0.00 cost cannot be silently confused with an unpriced model.
    """

    table = pricing if pricing is not None else _cached_pricing()
    return _compute(model, prompt_tokens, completion_tokens, cached_tokens, table)
