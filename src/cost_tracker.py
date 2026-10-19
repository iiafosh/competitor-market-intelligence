"""Token accounting and estimated USD cost of the OpenAI calls made by the pipeline.

Token semantics, verified against the type documentation of the installed OpenAI Python SDK (v3.34):

* ``prompt_tokens`` (Chat Completions) / ``input_tokens`` (Responses) is the whole prompt.
  ``cached_tokens`` ("cached tokens present in the prompt") and ``cache_write_tokens`` ("prompt tokens
  written to cache") are subsets of it, and OpenAI's Usage API defines uncached input as *excluding*
  cache-write tokens, so ``uncached = input - cached - cache_write``.
* ``reasoning_tokens`` are part of ``completion_tokens`` / ``output_tokens`` (the SDK notes they "are still
  counted in the total completion tokens for purposes of billing"), so they are reported but never added
  to the output cost a second time.

Costs are estimates from published list prices (``config/model_pricing.json`` or ``OPENAI_PRICE_*``
environment overrides), never an invoice.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

__all__ = [
    "CALL_OUTCOMES",
    "USAGE_FIELDS",
    "ApiCallRecord",
    "ModelPricing",
    "PricingError",
    "UsageTracker",
    "add_usage",
    "calculate_api_cost",
    "empty_usage",
    "extract_usage",
    "load_pricing_table",
    "match_pricing_key",
    "resolve_pricing",
    "uncached_input_tokens",
]

TOKENS_PER_PRICE_UNIT = 1_000_000
USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)
COST_FIELDS = ("input_cost_usd", "cached_input_cost_usd", "cache_write_cost_usd", "output_cost_usd", "total_cost_usd")
CALL_OUTCOMES = ("success", "invalid_response", "api_error")
COST_DECIMALS = 6

ENV_PRICE_INPUT = "OPENAI_PRICE_INPUT_PER_1M"
ENV_PRICE_CACHED_INPUT = "OPENAI_PRICE_CACHED_INPUT_PER_1M"
ENV_PRICE_CACHE_WRITE = "OPENAI_PRICE_CACHE_WRITE_PER_1M"
ENV_PRICE_OUTPUT = "OPENAI_PRICE_OUTPUT_PER_1M"
ENV_PRICING_REFERENCE_DATE = "OPENAI_PRICING_REFERENCE_DATE"  # optional label for overridden prices
ENVIRONMENT_OVERRIDE_SOURCE = "environment override"

# A model id may extend a pricing key only by a dated snapshot suffix ("-2025-04-14", "-0125",
# "-20250414"). This keeps "gpt-4o-mini-..." or "gpt-4o-mini-search-preview-..." from ever being
# priced as "gpt-4o" when the more specific entry is missing.
_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4})")

ESTIMATE_DISCLAIMER = (
    "Estimated cost = the token usage reported by the API for every call in this run (retried and failed "
    "attempts included) x OpenAI's published list prices per 1M tokens (see pricing_source and "
    "pricing_reference_date). Cached input is priced at the cached-input rate and cache writes at the "
    "cache-write rate (the regular input rate where no separate rate is published); reasoning tokens are "
    "already part of output tokens. This is an estimate, not an invoice: actual billing can differ because "
    "of price changes, service tier, regional-processing uplifts, discounts, credits or taxes."
)


class PricingError(ValueError):
    """Pricing data is missing or malformed."""


@dataclass(frozen=True)
class ModelPricing:
    """USD list prices per 1M tokens for one model; ``None`` means no separately published rate."""

    model: str
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    cache_write_per_1m: float | None = None
    source: str = "pricing table"
    reference_date: str | None = None

    def rates_per_1m(self) -> dict[str, float | None]:
        return {
            "input": self.input_per_1m,
            "cached_input": self.cached_input_per_1m,
            "cache_write": self.cache_write_per_1m,
            "output": self.output_per_1m,
        }


# ------------------------------------------------------------------------------- pricing table
def load_pricing_table(path: str | Path = config.PRICING_FILE) -> dict[str, Any]:
    """Load and validate the pricing table JSON; raise ``PricingError`` with a fix-it message."""
    pricing_path = Path(path)
    try:
        table = json.loads(pricing_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise PricingError(
            f"Pricing table not found at {pricing_path}. Restore config/model_pricing.json or set "
            f"{ENV_PRICE_INPUT} and {ENV_PRICE_OUTPUT} (USD per 1M tokens)."
        ) from exc
    except OSError as exc:
        raise PricingError(f"Cannot read pricing table {pricing_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PricingError(
            f"Pricing table {pricing_path} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    _validate_pricing_table(table, pricing_path)
    return table


def _validate_pricing_table(table: object, origin: object) -> None:
    if not isinstance(table, dict):
        raise PricingError(f"Pricing table {origin} must be a JSON object.")
    if table.get("currency", "USD") != "USD":
        raise PricingError(f"Pricing table {origin}: only USD prices are supported, got {table.get('currency')!r}.")
    if table.get("unit", "per_1m_tokens") != "per_1m_tokens":
        raise PricingError(f"Pricing table {origin}: unit must be 'per_1m_tokens', got {table.get('unit')!r}.")
    models = table.get("models")
    if not isinstance(models, dict) or not models:
        raise PricingError(f"Pricing table {origin} must contain a non-empty 'models' object.")
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise PricingError(f"Pricing table {origin}: entry for model {name!r} must be an object.")
        for rate in ("input", "output"):
            _rate_value(entry.get(rate), f"models.{name}.{rate}", required=True)
        for rate in ("cached_input", "cache_write"):
            _rate_value(entry.get(rate), f"models.{name}.{rate}", required=False)


def _rate_value(value: object, label: str, *, required: bool) -> float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
        raise PricingError(f"Invalid price {label}={value!r}: expected a non-negative number of USD per 1M tokens.")
    return float(value)


def match_pricing_key(model: str, model_keys: Iterable[str]) -> str | None:
    """Exact key, else the longest key that ``model`` extends with a dated snapshot suffix.

    A gateway vendor prefix is ignored, so OpenRouter's ``openai/gpt-6-luna-20260922`` is priced as
    ``gpt-6-luna`` (OpenRouter lists OpenAI models at OpenAI's own per-token prices).
    """
    wanted = model.strip().lower()
    if wanted.startswith("openai/"):
        wanted = wanted[len("openai/") :]
    keys = {str(key).lower(): str(key) for key in model_keys}
    if wanted in keys:
        return keys[wanted]
    matches = [key for key in keys if wanted.startswith(key) and _SNAPSHOT_SUFFIX.fullmatch(wanted[len(key) :])]
    return keys[max(matches, key=len)] if matches else None


def resolve_pricing(
    model: str, table: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None
) -> ModelPricing | None:
    """Pricing for ``model``: environment override, exact table key, or dated-snapshot prefix match."""
    override = _pricing_from_environment(model, os.environ if environ is None else environ)
    if override is not None:
        return override
    table = table or {}
    models = table.get("models") or {}
    key = match_pricing_key(model, models)
    if key is None:
        return None
    entry = models[key]
    return ModelPricing(
        model=key,
        input_per_1m=_rate_value(entry.get("input"), f"models.{key}.input", required=True),
        output_per_1m=_rate_value(entry.get("output"), f"models.{key}.output", required=True),
        cached_input_per_1m=_rate_value(entry.get("cached_input"), f"models.{key}.cached_input", required=False),
        cache_write_per_1m=_rate_value(entry.get("cache_write"), f"models.{key}.cache_write", required=False),
        source=_table_source_label(table),
        reference_date=table.get("pricing_reference_date"),
    )


def _table_source_label(table: Mapping[str, Any]) -> str:
    details = [f"{table['tier']} tier list prices" if table.get("tier") else "", str(table.get("source_url") or "")]
    details = [part for part in details if part]
    return f"config/model_pricing.json ({', '.join(details)})" if details else "config/model_pricing.json"


def _pricing_from_environment(model: str, environ: Mapping[str, str]) -> ModelPricing | None:
    raw_input = (environ.get(ENV_PRICE_INPUT) or "").strip()
    raw_output = (environ.get(ENV_PRICE_OUTPUT) or "").strip()
    if not raw_input and not raw_output:
        return None
    if not raw_input or not raw_output:
        logger.warning(
            "Ignoring the partial price override: set both %s and %s to override the pricing table.",
            ENV_PRICE_INPUT,
            ENV_PRICE_OUTPUT,
        )
        return None
    return ModelPricing(
        model=model,
        input_per_1m=_parse_env_rate(ENV_PRICE_INPUT, raw_input),
        output_per_1m=_parse_env_rate(ENV_PRICE_OUTPUT, raw_output),
        cached_input_per_1m=_optional_env_rate(environ, ENV_PRICE_CACHED_INPUT),
        cache_write_per_1m=_optional_env_rate(environ, ENV_PRICE_CACHE_WRITE),
        source=ENVIRONMENT_OVERRIDE_SOURCE,
        reference_date=(environ.get(ENV_PRICING_REFERENCE_DATE) or "").strip() or None,
    )


def _parse_env_rate(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        raise PricingError(f"{name} must be a non-negative number of USD per 1M tokens.")
    return value


def _optional_env_rate(environ: Mapping[str, str], name: str) -> float | None:
    raw = (environ.get(name) or "").strip()
    return _parse_env_rate(name, raw) if raw else None


# ------------------------------------------------------------------------------- usage maths
def empty_usage() -> dict[str, int]:
    return dict.fromkeys(USAGE_FIELDS, 0)


def add_usage(total: Mapping[str, int], extra: Mapping[str, int]) -> dict[str, int]:
    return {name: int(total.get(name, 0)) + int(extra.get(name, 0)) for name in USAGE_FIELDS}


def _field(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _first_field(source: Any, *names: str) -> Any:
    for name in names:
        value = _field(source, name)
        if value is not None:
            return value
    return None


def _count(value: Any) -> int:
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        return 0
    number = float(value)
    return max(int(number), 0) if math.isfinite(number) else 0


def extract_usage(sdk_usage: Any) -> dict[str, int]:
    """Normalise SDK usage (Chat Completions, Responses, a dict, or already-normalised) to ``USAGE_FIELDS``.

    Missing usage or missing detail objects count as 0.
    """
    if sdk_usage is None:
        return empty_usage()
    input_tokens = _count(_first_field(sdk_usage, "prompt_tokens", "input_tokens"))
    output_tokens = _count(_first_field(sdk_usage, "completion_tokens", "output_tokens"))
    input_details = _first_field(sdk_usage, "prompt_tokens_details", "input_tokens_details")
    output_details = _first_field(sdk_usage, "completion_tokens_details", "output_tokens_details")
    total = _field(sdk_usage, "total_tokens")

    def detail(details: Any, name: str, flat_name: str) -> int:
        # SDK objects nest the breakdown in *_details; a normalised dict keeps it flat.
        return _count(_field(details, name) or _field(sdk_usage, flat_name))

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": detail(input_details, "cached_tokens", "cached_input_tokens"),
        "cache_write_tokens": detail(input_details, "cache_write_tokens", "cache_write_tokens"),
        "output_tokens": output_tokens,
        "reasoning_tokens": detail(output_details, "reasoning_tokens", "reasoning_tokens"),
        "total_tokens": _count(total) if total is not None else input_tokens + output_tokens,
    }


def uncached_input_tokens(usage: Mapping[str, int]) -> int:
    """Input tokens billed at the regular rate: input minus cache reads and cache writes (floored at 0)."""
    return max(int(usage["input_tokens"]) - int(usage["cached_input_tokens"]) - int(usage["cache_write_tokens"]), 0)


def calculate_api_cost(usage: Any, pricing: ModelPricing) -> dict[str, float]:
    """Unrounded USD cost per billing category for one usage record (rounding happens at reporting)."""
    tokens = extract_usage(usage)
    cached_rate = pricing.cached_input_per_1m if pricing.cached_input_per_1m is not None else pricing.input_per_1m
    write_rate = pricing.cache_write_per_1m if pricing.cache_write_per_1m is not None else pricing.input_per_1m
    costs = {
        "input_cost_usd": uncached_input_tokens(tokens) * pricing.input_per_1m / TOKENS_PER_PRICE_UNIT,
        "cached_input_cost_usd": tokens["cached_input_tokens"] * cached_rate / TOKENS_PER_PRICE_UNIT,
        "cache_write_cost_usd": tokens["cache_write_tokens"] * write_rate / TOKENS_PER_PRICE_UNIT,
        "output_cost_usd": tokens["output_tokens"] * pricing.output_per_1m / TOKENS_PER_PRICE_UNIT,
    }
    costs["total_cost_usd"] = sum(costs.values())
    return costs


# ------------------------------------------------------------------------------- tracker
@dataclass
class ApiCallRecord:
    """One API attempt. ``model`` is the id the API reported (or the configured id when no response)."""

    product_id: str
    attempt: int
    model: str
    outcome: str
    error_type: str | None
    latency_seconds: float | None
    usage_reported: bool
    usage: dict[str, int]
    provider: str | None = None  # serving provider reported by a gateway (e.g. OpenRouter -> "OpenAI")
    provider_cost_usd: float | None = None  # actual charge reported by a gateway, when it reports one


def _reported_cost(usage: Any) -> float | None:
    """Gateway-reported charge for one call (OpenRouter returns ``usage.cost`` in USD credits)."""
    cost = _field(usage, "cost")
    if cost is None or isinstance(cost, bool) or not isinstance(cost, numbers.Real) or not math.isfinite(cost):
        return None
    return float(cost) if cost >= 0 else None


@dataclass
class _CostEstimate:
    total: float | None
    breakdown: dict[str, float] | None
    pricings: list[ModelPricing]
    unpriced_models: list[str]
    fallback_models: list[str]
    billable_calls: int


class UsageTracker:
    """Collects token usage for every API attempt and turns it into an execution cost summary."""

    def __init__(
        self, pricing_table: Mapping[str, Any] | None = None, *, environ: Mapping[str, str] | None = None
    ) -> None:
        self._pricing_table: Mapping[str, Any] = pricing_table if pricing_table is not None else {"models": {}}
        self._environ = environ
        self._calls: list[ApiCallRecord] = []
        self._cache_hits: list[str] = []

    def record_call(
        self,
        product_id: str,
        attempt: int,
        model: str | None,
        usage: Any,
        outcome: str,
        error_type: str | None = None,
        latency_seconds: float | None = None,
        *,
        provider: str | None = None,
    ) -> dict[str, int]:
        """Record one attempt (``usage`` may be None when no response arrived); return its normalised usage."""
        if outcome not in CALL_OUTCOMES:
            raise ValueError(f"outcome must be one of {CALL_OUTCOMES}, got {outcome!r}")
        if attempt < 1:
            raise ValueError(f"attempt numbers start at 1, got {attempt}")
        tokens = extract_usage(usage)
        self._calls.append(
            ApiCallRecord(
                product_id=str(product_id),
                attempt=int(attempt),
                model=str(model) if model else "unknown",
                outcome=outcome,
                error_type=error_type,
                latency_seconds=None if latency_seconds is None else round(float(latency_seconds), 3),
                usage_reported=usage is not None,
                usage=tokens,
                provider=provider or None,
                provider_cost_usd=_reported_cost(usage),
            )
        )
        return dict(tokens)

    def record_cache_hit(self, product_id: str) -> None:
        self._cache_hits.append(str(product_id))

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def total_usage(self) -> dict[str, int]:
        total = empty_usage()
        for call in self._calls:
            total = add_usage(total, call.usage)
        return total

    def calls_as_records(self) -> list[dict[str, Any]]:
        """JSON-safe per-attempt log (one dict per API call) with each call's estimated cost."""
        pricing_cache: dict[str, ModelPricing | None] = {}
        records = []
        for call in self._calls:
            pricing = self._pricing_for(call.model, pricing_cache)
            cost = calculate_api_cost(call.usage, pricing)["total_cost_usd"] if pricing else None
            records.append(
                {
                    "product_id": call.product_id,
                    "attempt": call.attempt,
                    "model": call.model,
                    "outcome": call.outcome,
                    "error_type": call.error_type,
                    "latency_seconds": call.latency_seconds,
                    "usage_reported": call.usage_reported,
                    **call.usage,
                    "estimated_cost_usd": None if cost is None else round(cost, COST_DECIMALS),
                    "provider": call.provider,
                    "provider_reported_cost_usd": None
                    if call.provider_cost_usd is None
                    else round(call.provider_cost_usd, 8),
                }
            )
        return records

    def summary(self, model: str) -> dict[str, Any]:
        """Execution metadata for ``model`` (the configured model id) - JSON-safe."""
        totals = self.total_usage()
        estimate = self._estimate_cost(model)
        successful = sum(1 for call in self._calls if call.outcome == "success")
        single_pricing = estimate.pricings[0] if len(estimate.pricings) == 1 else None
        reported = [call.provider_cost_usd for call in self._calls if call.provider_cost_usd is not None]
        return {
            "model": model,
            "models_reported_by_api": sorted({call.model for call in self._calls if call.usage_reported}),
            "providers_reported_by_api": sorted({call.provider for call in self._calls if call.provider}),
            "pricing_model": ", ".join(sorted({p.model for p in estimate.pricings})) or None,
            "total_api_calls": len(self._calls),
            "successful_api_calls": successful,
            "failed_api_calls": len(self._calls) - successful,
            "retried_calls": sum(1 for call in self._calls if call.attempt > 1),
            "api_calls_without_usage": sum(1 for call in self._calls if not call.usage_reported),
            "cache_hits": len(self._cache_hits),
            "input_tokens": totals["input_tokens"],
            "cached_input_tokens": totals["cached_input_tokens"],
            "cache_write_tokens": totals["cache_write_tokens"],
            "uncached_input_tokens": sum(uncached_input_tokens(call.usage) for call in self._calls),
            "output_tokens": totals["output_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "total_tokens": totals["total_tokens"],
            "estimated_cost_usd": None if estimate.total is None else round(estimate.total, COST_DECIMALS),
            "cost_breakdown_usd": None
            if estimate.breakdown is None
            else {name: round(value, COST_DECIMALS) for name, value in estimate.breakdown.items()},
            "pricing_currency": "USD",
            "pricing_rates_per_1m": single_pricing.rates_per_1m()
            if single_pricing
            else ({p.model: p.rates_per_1m() for p in estimate.pricings} or None),
            "pricing_reference_date": _joined({p.reference_date for p in estimate.pricings}),
            "pricing_source": _joined({p.source for p in estimate.pricings}),
            # Cross-check: what the API gateway itself reported charging (None when it reports nothing).
            "provider_reported_cost_usd": round(sum(reported), COST_DECIMALS) if reported else None,
            "provider_reported_cost_calls": len(reported),
            "cost_note": _cost_note(estimate),
        }

    def _pricing_for(self, model: str, resolved: dict[str, ModelPricing | None]) -> ModelPricing | None:
        if model not in resolved:
            try:
                pricing = resolve_pricing(model, self._pricing_table, self._environ)
            except PricingError as exc:
                logger.error("Cannot price model %s: %s", model, exc)
                pricing = None
            if pricing is not None and pricing.source == ENVIRONMENT_OVERRIDE_SOURCE:
                # One override prices every model id seen in the run; keep a single pricing identity.
                pricing = next(
                    (p for p in resolved.values() if p is not None and p.source == ENVIRONMENT_OVERRIDE_SOURCE), pricing
                )
            resolved[model] = pricing
        return resolved[model]

    def _estimate_cost(self, configured_model: str) -> _CostEstimate:
        resolved: dict[str, ModelPricing | None] = {}
        configured = self._pricing_for(configured_model, resolved)
        breakdown = dict.fromkeys(COST_FIELDS, 0.0)
        used: dict[ModelPricing, None] = {}
        unpriced: set[str] = set()
        fallback: set[str] = set()
        billable_calls = 0
        for call in self._calls:
            if not (call.usage["input_tokens"] or call.usage["output_tokens"]):
                continue
            billable_calls += 1
            pricing = self._pricing_for(call.model, resolved)
            if pricing is None and configured is not None:
                pricing = configured
                fallback.add(call.model)
            if pricing is None:
                unpriced.add(call.model)
                continue
            used[pricing] = None
            for name, value in calculate_api_cost(call.usage, pricing).items():
                breakdown[name] += value
        if not billable_calls and configured is not None:
            used[configured] = None
        if unpriced:
            return _CostEstimate(None, None, list(used), sorted(unpriced), sorted(fallback), billable_calls)
        return _CostEstimate(breakdown["total_cost_usd"], breakdown, list(used), [], sorted(fallback), billable_calls)


def _joined(values: set[str | None]) -> str | None:
    present = sorted(value for value in values if value)
    return "; ".join(present) if present else None


def _cost_note(estimate: _CostEstimate) -> str:
    if estimate.unpriced_models:
        return (
            f"No price is configured for model(s) {', '.join(estimate.unpriced_models)}, so estimated_cost_usd is "
            "null; the token counts above are exactly as reported by the API. Add the model to "
            f"config/model_pricing.json or set {ENV_PRICE_INPUT} and {ENV_PRICE_OUTPUT} (USD per 1M tokens) "
            "to enable the estimate, which is computed from published list prices and is not an invoice."
        )
    notes = [ESTIMATE_DISCLAIMER]
    if estimate.fallback_models:
        notes.append(
            f"Calls reported as {', '.join(estimate.fallback_models)} have no price entry and were priced at the "
            "configured model's rates."
        )
    if not estimate.billable_calls:
        notes.append("No billable API usage was recorded in this run (cached results only, or no call was made).")
    return " ".join(notes)
