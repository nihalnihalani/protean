from protean.model.pricing import (
    compute_fireworks_cost,
    compute_fireworks_cost_detailed,
    load_pricing,
)


def test_pricing_config_has_default_models():
    pricing = load_pricing()
    assert "accounts/fireworks/models/gpt-oss-120b" in pricing
    assert "accounts/fireworks/models/gpt-oss-20b" in pricing
    entry = pricing["accounts/fireworks/models/gpt-oss-120b"]
    assert entry["input_usd_per_million"] == 0.15
    assert entry["output_usd_per_million"] == 0.60


def test_compute_cost_120b():
    # 1M prompt tokens at $0.15 + 1M completion tokens at $0.60 = $0.75.
    cost = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-120b",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == 0.75


def test_compute_cost_20b_small_counts():
    cost = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-20b",
        prompt_tokens=1000,
        completion_tokens=2000,
    )
    expected = round(1000 * 0.07 / 1_000_000 + 2000 * 0.30 / 1_000_000, 8)
    assert cost == expected


def test_unknown_model_returns_zero():
    assert compute_fireworks_cost("accounts/fireworks/models/does-not-exist", 100, 100) == 0.0


def test_zero_tokens_is_zero_cost():
    assert compute_fireworks_cost("accounts/fireworks/models/gpt-oss-120b", 0, 0) == 0.0


def test_explicit_pricing_table_override():
    pricing = {"m": {"input_usd_per_million": 1.0, "output_usd_per_million": 2.0}}
    cost = compute_fireworks_cost("m", 1_000_000, 1_000_000, pricing=pricing)
    assert cost == 3.0


def test_config_cached_rate_field_is_used_for_cached_tokens():
    # 120b config carries a cached_input rate; cached prompt tokens must be
    # billed at that lower rate, not the full input rate.
    pricing = load_pricing()
    entry = pricing["accounts/fireworks/models/gpt-oss-120b"]
    assert "cached_input_usd_per_million" in entry
    # 1M prompt tokens, all cached, no completion: cost == cached rate.
    cost = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-120b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=1_000_000,
    )
    assert cost == entry["cached_input_usd_per_million"]


def test_cached_tokens_cheaper_than_uncached():
    full = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-120b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    cached = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-120b",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=1_000_000,
    )
    assert cached < full


def test_partial_cache_splits_prompt_tokens():
    pricing = {
        "m": {
            "input_usd_per_million": 1.0,
            "output_usd_per_million": 2.0,
            "cached_input_usd_per_million": 0.25,
        }
    }
    # 1M prompt, half cached, no completion:
    # 0.5M * 1.0 + 0.5M * 0.25 = 0.5 + 0.125 = 0.625
    cost = compute_fireworks_cost(
        "m", 1_000_000, 0, cached_tokens=500_000, pricing=pricing
    )
    assert cost == 0.625


def test_cached_without_config_rate_falls_back_to_input_rate():
    # 20b config has no cached rate; cached tokens bill at the full input rate.
    full = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-20b", 1_000_000, 0
    )
    cached = compute_fireworks_cost(
        "accounts/fireworks/models/gpt-oss-20b", 1_000_000, 0, cached_tokens=1_000_000
    )
    assert cached == full


def test_cached_tokens_clamped_to_prompt_tokens():
    pricing = {
        "m": {
            "input_usd_per_million": 1.0,
            "output_usd_per_million": 2.0,
            "cached_input_usd_per_million": 0.25,
        }
    }
    # cached > prompt should be clamped: all 100 prompt tokens cached, no negative.
    cost = compute_fireworks_cost("m", 100, 0, cached_tokens=999, pricing=pricing)
    assert cost == round(100 * 0.25 / 1_000_000, 8)


def test_detailed_reports_pricing_miss_on_unknown_model():
    result = compute_fireworks_cost_detailed(
        "accounts/fireworks/models/does-not-exist", 100, 100
    )
    assert result.pricing_miss is True
    assert result.cost_usd == 0.0


def test_detailed_no_miss_on_known_model():
    result = compute_fireworks_cost_detailed(
        "accounts/fireworks/models/gpt-oss-120b", 1_000_000, 1_000_000
    )
    assert result.pricing_miss is False
    assert result.cost_usd == 0.75
