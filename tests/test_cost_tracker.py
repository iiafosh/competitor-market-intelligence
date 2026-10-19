"""Tests for token accounting and USD cost estimation (no network, own pricing fixture)."""

from __future__ import annotations

import json
import logging

import pytest
from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage

import config
from cost_tracker import (
    ENVIRONMENT_OVERRIDE_SOURCE,
    ModelPricing,
    PricingError,
    UsageTracker,
    add_usage,
    calculate_api_cost,
    empty_usage,
    extract_usage,
    load_pricing_table,
    resolve_pricing,
    uncached_input_tokens,
)

NO_ENV: dict[str, str] = {}  # never let a developer's real OPENAI_PRICE_* variables leak into tests


def pricing_table() -> dict:
    return {
        "currency": "USD",
        "unit": "per_1m_tokens",
        "tier": "standard",
        "pricing_reference_date": "2026-10-17",
        "source_url": "https://example.test/pricing",
        "models": {
            "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "cache_write": None, "output": 1.60},
            "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "cache_write": None, "output": 8.00},
            "gpt-4o": {"input": 2.50, "cached_input": 1.25, "cache_write": None, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "cache_write": None, "output": 0.60},
            "gpt-6-sol": {"input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 10.00},
            "legacy-model": {"input": 5.00, "cached_input": None, "cache_write": None, "output": 15.00},
        },
    }


def usage(input_tokens=0, cached=0, cache_write=0, output=0, reasoning=0) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_tokens": cache_write,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": input_tokens + output,
    }


MINI = ModelPricing("gpt-4.1-mini", input_per_1m=0.40, output_per_1m=1.60, cached_input_per_1m=0.10)


# ----------------------------------------------------------------------------- cost formula
def test_known_numbers_with_cached_input():
    # 10,000 uncached x 0.40 + 2,000 cached x 0.10 + 6,000 output x 1.60 (per 1M tokens)
    cost = calculate_api_cost(usage(12_000, cached=2_000, output=6_000), MINI)
    assert cost["input_cost_usd"] == pytest.approx(0.004)
    assert cost["cached_input_cost_usd"] == pytest.approx(0.0002)
    assert cost["cache_write_cost_usd"] == 0
    assert cost["output_cost_usd"] == pytest.approx(0.0096)
    assert cost["total_cost_usd"] == pytest.approx(0.0138)


def test_cache_write_tokens_billed_at_cache_write_rate():
    pricing = resolve_pricing("gpt-6-sol", pricing_table(), NO_ENV)
    cost = calculate_api_cost(usage(10_000, cached=4_000, cache_write=3_000, output=1_000), pricing)
    assert cost["input_cost_usd"] == pytest.approx(3_000 * 2.00 / 1e6)  # uncached = 10k - 4k - 3k
    assert cost["cached_input_cost_usd"] == pytest.approx(4_000 * 0.20 / 1e6)
    assert cost["cache_write_cost_usd"] == pytest.approx(3_000 * 2.50 / 1e6)
    assert cost["output_cost_usd"] == pytest.approx(1_000 * 10.00 / 1e6)
    assert cost["total_cost_usd"] == pytest.approx(0.0243)


def test_missing_cached_and_cache_write_rates_fall_back_to_input_rate():
    pricing = resolve_pricing("legacy-model", pricing_table(), NO_ENV)
    cost = calculate_api_cost(usage(1_000_000, cached=200_000, cache_write=100_000, output=0), pricing)
    assert cost["total_cost_usd"] == pytest.approx(5.00)  # every input token at the 5.00 input rate


def test_reasoning_tokens_are_not_billed_twice():
    with_reasoning = calculate_api_cost(usage(1_000, output=2_000, reasoning=1_500), MINI)
    without = calculate_api_cost(usage(1_000, output=2_000), MINI)
    assert with_reasoning == without


def test_uncached_input_never_negative():
    assert uncached_input_tokens(usage(100, cached=80, cache_write=50)) == 0


# ----------------------------------------------------------------------------- pricing resolution
@pytest.mark.parametrize(
    ("model", "expected_key"),
    [
        ("gpt-4.1-mini", "gpt-4.1-mini"),
        ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
        ("gpt-4.1-2025-04-14", "gpt-4.1"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("GPT-4.1-MINI", "gpt-4.1-mini"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),  # OpenRouter-style vendor prefix
        ("openai/gpt-6-sol-20260922", "gpt-6-sol"),  # vendor prefix + compact snapshot date
    ],
)
def test_resolve_pricing_exact_and_snapshot_prefix(model, expected_key):
    pricing = resolve_pricing(model, pricing_table(), NO_ENV)
    assert pricing is not None and pricing.model == expected_key
    assert pricing.reference_date == "2026-10-17"
    assert "example.test/pricing" in pricing.source


@pytest.mark.parametrize("model", ["gpt-4o-mini-search-preview-2025-03-11", "gpt-4.1-minimal", "mystery-model"])
def test_resolve_pricing_rejects_unrelated_or_unknown_models(model):
    assert resolve_pricing(model, pricing_table(), NO_ENV) is None


def test_mini_snapshot_never_priced_as_parent_model():
    table = pricing_table()
    del table["models"]["gpt-4o-mini"]
    assert resolve_pricing("gpt-4o-mini-2024-07-18", table, NO_ENV) is None
    assert resolve_pricing("openai/gpt-4o-mini-2024-07-18", table, NO_ENV) is None


def test_gateway_reported_cost_and_provider_are_recorded():
    tracker = UsageTracker(pricing_table(), environ=NO_ENV)
    reported = CompletionUsage.model_validate(
        {"prompt_tokens": 1_000, "completion_tokens": 200, "total_tokens": 1_200, "cost": 0.00042}
    )
    tracker.record_call("P-1", 1, "openai/gpt-4.1-mini", reported, "success", provider="OpenAI")
    tracker.record_call("P-2", 1, "openai/gpt-4.1-mini", usage(1_000, output=200), "success")

    summary = tracker.summary("openai/gpt-4.1-mini")
    assert summary["pricing_model"] == "gpt-4.1-mini"
    assert summary["providers_reported_by_api"] == ["OpenAI"]
    assert summary["provider_reported_cost_usd"] == pytest.approx(0.00042)
    assert summary["provider_reported_cost_calls"] == 1
    assert summary["estimated_cost_usd"] == pytest.approx(2 * (1_000 * 0.40 + 200 * 1.60) / 1e6)
    records = tracker.calls_as_records()
    assert records[0]["provider"] == "OpenAI" and records[0]["provider_reported_cost_usd"] == pytest.approx(0.00042)
    assert records[1]["provider"] is None and records[1]["provider_reported_cost_usd"] is None


def test_environment_override_wins():
    env = {
        "OPENAI_PRICE_INPUT_PER_1M": "1.0",
        "OPENAI_PRICE_OUTPUT_PER_1M": "4.0",
        "OPENAI_PRICE_CACHED_INPUT_PER_1M": "0.25",
    }
    pricing = resolve_pricing("gpt-4.1-mini", pricing_table(), env)
    assert pricing.source == ENVIRONMENT_OVERRIDE_SOURCE
    assert (pricing.input_per_1m, pricing.cached_input_per_1m, pricing.cache_write_per_1m, pricing.output_per_1m) == (
        1.0,
        0.25,
        None,
        4.0,
    )
    assert resolve_pricing("model-missing-from-table", pricing_table(), env) is not None


def test_partial_environment_override_is_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="cost_tracker"):
        pricing = resolve_pricing("gpt-4.1-mini", pricing_table(), {"OPENAI_PRICE_INPUT_PER_1M": "9"})
    assert pricing.input_per_1m == 0.40
    assert "partial price override" in caplog.text


def test_invalid_environment_price_raises():
    with pytest.raises(PricingError, match="OPENAI_PRICE_OUTPUT_PER_1M"):
        resolve_pricing("gpt-4.1-mini", pricing_table(), {"OPENAI_PRICE_INPUT_PER_1M": "1", "OPENAI_PRICE_OUTPUT_PER_1M": "-2"})


# ----------------------------------------------------------------------------- pricing file
def test_load_pricing_table_valid_file(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(pricing_table()), encoding="utf-8")
    assert load_pricing_table(path)["models"]["gpt-4.1-mini"]["output"] == 1.60


def test_load_pricing_table_missing_file(tmp_path):
    with pytest.raises(PricingError, match="not found"):
        load_pricing_table(tmp_path / "absent.json")


def test_load_pricing_table_invalid_json(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PricingError, match="not valid JSON"):
        load_pricing_table(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["models"]["gpt-4.1"].update(output=None),
        lambda t: t["models"]["gpt-4.1"].update(input=-1),
        lambda t: t["models"]["gpt-4.1"].update(cached_input="cheap"),
        lambda t: t.update(models={}),
        lambda t: t.update(currency="EUR"),
        lambda t: t.update(unit="per_1k_tokens"),
    ],
)
def test_load_pricing_table_rejects_bad_content(tmp_path, mutate):
    table = pricing_table()
    mutate(table)
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(PricingError):
        load_pricing_table(path)


def test_shipped_pricing_file_is_valid_and_prices_default_model():
    table = load_pricing_table(config.PRICING_FILE)
    assert resolve_pricing(config.DEFAULT_OPENAI_MODEL, table, NO_ENV) is not None


# ----------------------------------------------------------------------------- usage extraction
def test_extract_usage_from_chat_completions_usage():
    sdk_usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1_200,
            "completion_tokens": 300,
            "total_tokens": 1_500,
            "prompt_tokens_details": {"cached_tokens": 1_024, "cache_write_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 128},
        }
    )
    assert extract_usage(sdk_usage) == {
        "input_tokens": 1_200,
        "cached_input_tokens": 1_024,
        "cache_write_tokens": 64,
        "output_tokens": 300,
        "reasoning_tokens": 128,
        "total_tokens": 1_500,
    }


def test_extract_usage_from_responses_usage():
    sdk_usage = ResponseUsage.model_validate(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 7},
        }
    )
    assert extract_usage(sdk_usage) == usage(100, cached=10, cache_write=5, output=20, reasoning=7)


def test_extract_usage_handles_missing_details_and_none():
    assert extract_usage({"prompt_tokens": 10, "completion_tokens": 5}) == usage(10, output=5)
    assert extract_usage({"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": None})["cached_input_tokens"] == 0
    assert extract_usage(None) == empty_usage()


def test_extract_usage_is_idempotent_and_add_usage_sums():
    normalised = usage(50, cached=10, cache_write=2, output=7, reasoning=3)
    assert extract_usage(normalised) == normalised
    assert add_usage(normalised, normalised)["cached_input_tokens"] == 20


# ----------------------------------------------------------------------------- tracker
def make_tracker(**kwargs) -> UsageTracker:
    return UsageTracker(pricing_table(), environ=kwargs.pop("environ", NO_ENV))


def test_tracker_counts_every_attempt_and_prices_api_reported_model():
    tracker = make_tracker()
    reported = "gpt-4.1-mini-2025-04-14"
    # Product A: invalid answer, then success. Product B: rate limited (no usage), then success.
    tracker.record_call("A", 1, reported, usage(1_000, cached=200, output=100), "invalid_response", "InvalidJSON", 0.8)
    tracker.record_call("A", 2, reported, usage(1_000, cached=200, output=150), "success", latency_seconds=0.9)
    tracker.record_call("B", 1, "gpt-4.1-mini", None, "api_error", "RateLimitError", 0.1)
    tracker.record_call("B", 2, reported, usage(800, output=120), "success")
    tracker.record_cache_hit("C")

    summary = tracker.summary("gpt-4.1-mini")
    assert summary["total_api_calls"] == 4
    assert summary["successful_api_calls"] == 2
    assert summary["failed_api_calls"] == 2
    assert summary["retried_calls"] == 2
    assert summary["api_calls_without_usage"] == 1
    assert summary["cache_hits"] == 1
    assert summary["input_tokens"] == 2_800
    assert summary["cached_input_tokens"] == 400
    assert summary["uncached_input_tokens"] == 2_400
    assert summary["output_tokens"] == 370
    assert summary["total_tokens"] == 3_170
    assert summary["models_reported_by_api"] == [reported]
    assert summary["pricing_model"] == "gpt-4.1-mini"
    expected = (2_400 * 0.40 + 400 * 0.10 + 370 * 1.60) / 1e6
    assert summary["estimated_cost_usd"] == pytest.approx(round(expected, 6))
    assert summary["cost_breakdown_usd"]["total_cost_usd"] == summary["estimated_cost_usd"]
    assert summary["pricing_rates_per_1m"] == {"input": 0.40, "cached_input": 0.10, "cache_write": None, "output": 1.60}
    assert summary["pricing_reference_date"] == "2026-10-17"
    assert "estimate" in summary["cost_note"] and "not an invoice" in summary["cost_note"]
    assert "list prices" in summary["cost_note"]
    json.dumps(summary)  # JSON-safe


def test_tracker_summary_contains_contract_keys():
    required = {
        "model", "pricing_model", "total_api_calls", "successful_api_calls", "failed_api_calls", "retried_calls",
        "cache_hits", "input_tokens", "cached_input_tokens", "cache_write_tokens", "uncached_input_tokens",
        "output_tokens", "reasoning_tokens", "total_tokens", "estimated_cost_usd", "cost_breakdown_usd",
        "pricing_rates_per_1m", "pricing_reference_date", "pricing_source", "cost_note",
    }  # fmt: skip
    assert required <= make_tracker().summary("gpt-4.1-mini").keys()


def test_tracker_unknown_pricing_gives_null_cost_with_explanation():
    tracker = make_tracker()
    tracker.record_call("A", 1, "mystery-model", usage(1_000, output=100), "success")
    summary = tracker.summary("mystery-model")
    assert summary["estimated_cost_usd"] is None
    assert summary["cost_breakdown_usd"] is None
    assert summary["input_tokens"] == 1_000
    assert "mystery-model" in summary["cost_note"]
    assert "OPENAI_PRICE_INPUT_PER_1M" in summary["cost_note"]


def test_tracker_falls_back_to_configured_pricing_for_unlisted_reported_model():
    tracker = make_tracker()
    tracker.record_call("A", 1, "gpt-4.1-mini-experimental", usage(1_000_000), "success")
    summary = tracker.summary("gpt-4.1-mini")
    assert summary["estimated_cost_usd"] == pytest.approx(0.40)
    assert "gpt-4.1-mini-experimental" in summary["cost_note"]


def test_tracker_with_only_cache_hits_costs_zero():
    tracker = make_tracker()
    tracker.record_cache_hit("A")
    summary = tracker.summary("gpt-4.1-mini")
    assert summary["total_api_calls"] == 0
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["pricing_model"] == "gpt-4.1-mini"
    assert "No billable API usage" in summary["cost_note"]


def test_tracker_environment_override_applies_to_all_calls():
    env = {"OPENAI_PRICE_INPUT_PER_1M": "1.0", "OPENAI_PRICE_OUTPUT_PER_1M": "2.0"}
    tracker = make_tracker(environ=env)
    tracker.record_call("A", 1, "gpt-4.1-mini-2025-04-14", usage(1_000_000, output=1_000_000), "success")
    summary = tracker.summary("gpt-4.1-mini")
    assert summary["estimated_cost_usd"] == pytest.approx(3.0)
    assert summary["pricing_source"] == ENVIRONMENT_OVERRIDE_SOURCE
    assert summary["pricing_model"] == "gpt-4.1-mini"


def test_calls_as_records_are_json_safe_and_priced():
    tracker = make_tracker()
    tracker.record_call("A", 1, "gpt-4.1-mini", usage(1_000_000), "success", latency_seconds=1.23456)
    tracker.record_call("A", 2, "gpt-4.1-mini", None, "api_error", "APITimeoutError")
    records = tracker.calls_as_records()
    json.dumps(records)
    assert records[0]["estimated_cost_usd"] == pytest.approx(0.40)
    assert records[0]["latency_seconds"] == 1.235
    assert records[1]["usage_reported"] is False and records[1]["input_tokens"] == 0


def test_record_call_validates_arguments():
    tracker = make_tracker()
    with pytest.raises(ValueError, match="outcome"):
        tracker.record_call("A", 1, "gpt-4.1-mini", None, "maybe")
    with pytest.raises(ValueError, match="attempt"):
        tracker.record_call("A", 0, "gpt-4.1-mini", None, "success")
