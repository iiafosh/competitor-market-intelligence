"""Tests for Phase 4 exports: CSV encoding/columns, JSON structure and cross-file consistency."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from exporter import (
    REQUIRED_CSV_COLUMNS,
    build_enriched_dataframe,
    build_full_analysis,
    build_market_report,
    export_enriched_csv,
    export_full_analysis,
    to_json_safe,
    verify_exports,
)


def _product(product_id: str, name: str, price: float | None, currency: str | None, source: str) -> dict:
    return {
        "product_id": product_id,
        "product_name": name,
        "price": price,
        "currency": currency,
        "rating": 4,
        "availability": "in_stock",
        "description": "Sample description",
        "source": source,
        "product_url": f"https://example.com/{product_id}",
        "scraped_at": "2026-10-17T12:00:00Z",
        "review_text": None,
        "category": "Mystery",
        "review_count": 0,
        "review_source": None,
        "stock_quantity": 5,
        "original_price": None,
        "rating_unrounded": 4.0,
        "price_raw": f"£{price}",
        "rating_raw": "Four",
        "availability_raw": "In stock (5 available)",
        "data_gaps": "customer_reviews",
    }


ANALYSIS = {
    "product_name": "Café Noir — £ edition",
    "sentiment_summary": "Customer sentiment cannot be reliably determined: no review excerpts were provided.",
    "pros": ["Priced 12% below the category median"],
    "cons": [],
    "pricing_verdict": "51.77 GBP vs a 45.00 GBP median.",
    "strategic_recommendation": "=Monitor weekly and match within 5%.",
}


@pytest.fixture()
def dataset():
    products = [
        _product("P-1", "Café Noir — £ edition", 51.77, "GBP", "books_toscrape"),
        _product("P-2", "Failed Product", 10.0, "USD", "webscraper_io"),
        _product("P-3", "Never Analysed", None, None, "web_scraping_dev"),
    ]
    outcomes = [
        {"product_id": "P-1", "source": "books_toscrape", "product_url": "https://example.com/P-1", "product_name": products[0]["product_name"], "status": "success", "analysis": ANALYSIS, "error": None, "attempts": 1, "from_cache": False, "model": "gpt-6-luna", "usage": {"input_tokens": 900, "output_tokens": 150}},
        {"product_id": "P-2", "source": "webscraper_io", "product_url": "https://example.com/P-2", "product_name": "Failed Product", "status": "failed", "analysis": None, "error": {"type": "AnalysisValidationError", "message": "missing field"}, "attempts": 2, "from_cache": False, "model": "gpt-6-luna", "usage": {}},
    ]
    contexts = {
        "P-1": {"comparison_basis": "same category and currency", "peer_count": 4, "price_stats": {"median": 45.0}, "price_position": {"difference_vs_median_pct": 15.0}},
        "P-2": {"comparison_basis": "none", "peer_count": 0, "price_stats": None, "price_position": None},
    }
    return products, outcomes, contexts


def test_enriched_frame_has_required_columns_and_no_fabricated_analysis(dataset):
    products, outcomes, contexts = dataset
    frame = build_enriched_dataframe(products, outcomes, contexts)

    assert list(frame.columns[: len(REQUIRED_CSV_COLUMNS)]) == list(REQUIRED_CSV_COLUMNS)
    statuses = dict(zip(frame["product_id"], frame["analysis_status"]))
    assert statuses == {"P-1": "success", "P-2": "failed", "P-3": "not_analyzed"}
    failed = frame.set_index("product_id").loc["P-2"]
    assert pd.isna(failed["sentiment_summary"]) and pd.isna(failed["pros"]) and pd.isna(failed["pricing_verdict"])
    assert json.loads(frame.set_index("product_id").loc["P-1", "pros"]) == ANALYSIS["pros"]
    assert frame.set_index("product_id").loc["P-1", "cons"] == "[]"
    assert str(frame["rating"].dtype) == "Int64"
    assert pd.isna(frame.set_index("product_id").loc["P-3", "price"])  # missing price stays missing, never 0


def test_csv_is_utf8_sig_and_neutralises_formula_cells(dataset, tmp_path):
    products, outcomes, contexts = dataset
    path = export_enriched_csv(build_enriched_dataframe(products, outcomes, contexts), tmp_path / "enriched.csv")

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    table = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert "Café Noir — £ edition" in table["product_name"].tolist()
    assert table.loc[table["product_id"] == "P-1", "strategic_recommendation"].item().startswith("'=")


def test_full_analysis_contains_only_successes_and_records_failures(dataset):
    products, outcomes, contexts = dataset
    payload = build_full_analysis(
        products=products,
        outcomes=outcomes,
        contexts=contexts,
        execution_summary={"model": "gpt-6-luna", "estimated_cost_usd": float("nan")},
        data_sources=[],
        data_quality={},
        methodology={},
    )
    assert [item["product_id"] for item in payload["products"]] == ["P-1"]
    assert tuple(payload["products"][0]["analysis"]) == config.ANALYSIS_FIELDS
    assert payload["errors"][0]["product_id"] == "P-2"
    assert payload["execution_summary"]["estimated_cost_usd"] is None  # NaN never reaches the JSON
    json.dumps(payload, allow_nan=False)


def test_round_trip_verification_passes_and_detects_mismatch(dataset, tmp_path):
    products, outcomes, contexts = dataset
    frame = build_enriched_dataframe(products, outcomes, contexts)
    payload = build_full_analysis(products=products, outcomes=outcomes, contexts=contexts, execution_summary={}, data_sources=[], data_quality={}, methodology={})
    csv_path = export_enriched_csv(frame, tmp_path / "enriched.csv")
    json_path = export_full_analysis(payload, tmp_path / "full.json")

    ok = verify_exports(csv_path, json_path, ["P-1", "P-2", "P-3"])
    assert ok["consistent"], ok["problems"]

    payload["products"] = []
    export_full_analysis(payload, json_path)
    broken = verify_exports(csv_path, json_path, ["P-1", "P-2", "P-3"])
    assert not broken["consistent"]


def test_to_json_safe_converts_pandas_and_numpy_values():
    data = {"ts": pd.Timestamp("2026-10-17 12:00:00", tz="UTC"), "na": pd.NA, "n": pd.array([3], dtype="Int64")[0], "f": float("nan")}
    assert to_json_safe(data) == {"ts": "2026-10-17T12:00:00Z", "na": None, "n": 3, "f": None}


def test_market_report_renders(dataset):
    products, outcomes, contexts = dataset
    frame = build_enriched_dataframe(products, outcomes, contexts)
    payload = {"generated_at": "2026-10-17T12:00:00Z", "execution_summary": {"model": "gpt-6-luna", "estimated_cost_usd": 0.0012, "total_tokens": 1050, "total_api_calls": 3, "failed_analyses": 1}, "data_quality": {}}
    markdown = build_market_report(frame, payload, {"books_toscrape": "Competitor A"})
    assert "# Competitor Market Intelligence Report" in markdown
    assert "Competitor A" in markdown
