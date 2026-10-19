"""Phase 4 — Output delivery: enriched CSV, full-analysis JSON and a client market report.

Every export is written atomically and then re-opened and cross-checked, so a delivery is
only reported as complete when both files parse and describe the same products.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

PROJECT_NAME = "Automated Competitor Market Intelligence"
CLIENT_NAME = "MarketScope Analytics (fictional client from the capstone brief)"
SCENARIO_DISCLAIMER = (
    "Simulated competitive-intelligence scenario. The three 'competitors' are public web-scraping "
    "sandboxes that explicitly permit automated collection (books.toscrape.com, webscraper.io test "
    "shop, web-scraping.dev). Their prices, ratings and reviews are demo data published by those "
    "sites (Books to Scrape states its prices and ratings were randomly assigned); review excerpts "
    "are the sandbox's sample reviews, not real customer feedback. The analysis demonstrates the "
    "method and must not be read as statements about real products or markets."
)

STATUS_SUCCESS, STATUS_FAILED, STATUS_NOT_ANALYZED = "success", "failed", "not_analyzed"

# Assignment section 9.1, in the order listed there.
REQUIRED_CSV_COLUMNS = (
    "product_name",
    "price",
    "currency",
    "rating",
    "availability",
    "description",
    "source",
    "product_url",
    "scraped_at",
    "sentiment_summary",
    "pros",
    "cons",
    "pricing_verdict",
    "strategic_recommendation",
    "analysis_status",
)
EXTRA_CSV_COLUMNS = (
    "product_id",
    "category",
    "review_text",
    "review_count",
    "review_source",
    "stock_quantity",
    "original_price",
    "rating_unrounded",
    "data_gaps",
    "benchmark_basis",
    "benchmark_peer_count",
    "benchmark_median_price",
    "price_vs_benchmark_median_pct",
    "analysis_error",
    "analysis_attempts",
    "analysis_from_cache",
    "analysis_model",
    "price_raw",
    "rating_raw",
    "availability_raw",
)
ENRICHED_CSV_COLUMNS = REQUIRED_CSV_COLUMNS + EXTRA_CSV_COLUMNS
_INTEGER_COLUMNS = ("rating", "review_count", "stock_quantity", "benchmark_peer_count", "analysis_attempts")
_FLOAT_COLUMNS = ("price", "original_price", "rating_unrounded", "benchmark_median_price", "price_vs_benchmark_median_pct")

# Scraped text is untrusted. A cell starting with one of these characters can be executed as a
# formula when the CSV is opened in Excel (CSV injection), so such cells get a leading apostrophe.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
_FREE_TEXT_COLUMNS = (
    "product_name",
    "description",
    "category",
    "review_text",
    "review_source",
    "sentiment_summary",
    "pricing_verdict",
    "strategic_recommendation",
    "analysis_error",
    "price_raw",
    "rating_raw",
    "availability_raw",
)


# --------------------------------------------------------------------------- helpers
def utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-compatible native Python types."""
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        stamp = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _atomic_write_text(path: Path, text: str, encoding: str) -> None:
    """Write via a temporary file + rename so a crash never leaves a half-written deliverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _neutralize_formula(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    value = to_json_safe(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), digits)
    return None


def _outcomes_by_product(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(outcome["product_id"]): outcome for outcome in outcomes}


# ------------------------------------------------------------------ enriched dataset
def build_enriched_dataframe(
    products: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Join cleaned products with their AI analysis and price benchmark (one row per product).

    Products whose analysis failed (or was never attempted) keep empty analysis columns —
    nothing is invented to fill the gap.
    """
    outcome_lookup = _outcomes_by_product(outcomes)
    rows: list[dict[str, Any]] = []
    for product in products:
        product_id = str(product["product_id"])
        outcome = outcome_lookup.get(product_id)
        context = contexts.get(product_id) or {}
        analysis = outcome.get("analysis") if outcome else None
        succeeded = bool(outcome) and outcome.get("status") == STATUS_SUCCESS and isinstance(analysis, Mapping)
        if outcome is None:
            status = STATUS_NOT_ANALYZED
        else:
            status = STATUS_SUCCESS if succeeded else STATUS_FAILED
        error = (outcome.get("error") if outcome else None) or {}
        stats = context.get("price_stats") or {}
        position = context.get("price_position") or {}

        row = {column: product.get(column) for column in ENRICHED_CSV_COLUMNS if column in product}
        row.update(
            {
                "sentiment_summary": analysis["sentiment_summary"] if succeeded else None,
                "pros": json.dumps(analysis["pros"], ensure_ascii=False) if succeeded else None,
                "cons": json.dumps(analysis["cons"], ensure_ascii=False) if succeeded else None,
                "pricing_verdict": analysis["pricing_verdict"] if succeeded else None,
                "strategic_recommendation": analysis["strategic_recommendation"] if succeeded else None,
                "analysis_status": status,
                "benchmark_basis": context.get("comparison_basis"),
                "benchmark_peer_count": context.get("peer_count"),
                "benchmark_median_price": stats.get("median"),
                "price_vs_benchmark_median_pct": position.get("difference_vs_median_pct"),
                "analysis_error": f"{error.get('type')}: {error.get('message')}" if error else None,
                "analysis_attempts": (outcome or {}).get("attempts") if outcome else None,
                "analysis_from_cache": bool((outcome or {}).get("from_cache")) if outcome else None,
                "analysis_model": (outcome or {}).get("model") if outcome else None,
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows).reindex(columns=list(ENRICHED_CSV_COLUMNS))
    for column in _INTEGER_COLUMNS:
        frame[column] = pd.array([to_json_safe(v) for v in frame[column]], dtype="Int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def export_enriched_csv(frame: pd.DataFrame, path: Path = config.ENRICHED_CSV_FILE) -> Path:
    """Write the enriched dataset as Excel-compatible CSV (UTF-8 with BOM, RFC 4180 line endings)."""
    safe = frame.copy()
    for column in _FREE_TEXT_COLUMNS:
        if column in safe.columns:
            safe[column] = safe[column].map(_neutralize_formula)
    csv_text = safe.to_csv(index=False, lineterminator="\r\n")
    _atomic_write_text(path, csv_text, encoding="utf-8-sig")
    logger.info("Wrote %s (%d rows, %d columns)", path, len(safe), len(safe.columns))
    return path


# ------------------------------------------------------------------ full JSON report
def build_full_analysis(
    *,
    products: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    execution_summary: Mapping[str, Any],
    data_sources: Sequence[Mapping[str, Any]],
    data_quality: Mapping[str, Any],
    methodology: Mapping[str, Any],
    extra_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble ``full_analysis.json``: validated analyses, run metadata and every failure."""
    product_lookup = {str(p["product_id"]): p for p in products}
    analyzed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = [dict(error) for error in extra_errors]

    for outcome in outcomes:
        product_id = str(outcome["product_id"])
        product = product_lookup.get(product_id, {})
        if outcome.get("status") == STATUS_SUCCESS and isinstance(outcome.get("analysis"), Mapping):
            analyzed.append(
                {
                    "product_id": product_id,
                    "source": outcome.get("source") or product.get("source"),
                    "product_url": outcome.get("product_url") or product.get("product_url"),
                    "category": product.get("category"),
                    "price": product.get("price"),
                    "currency": product.get("currency"),
                    "rating": product.get("rating"),
                    "availability": product.get("availability"),
                    "review_count": product.get("review_count"),
                    "scraped_at": product.get("scraped_at"),
                    "data_gaps": product.get("data_gaps"),
                    "competitor_benchmark": contexts.get(product_id),
                    "analysis": dict(outcome["analysis"]),
                    "analysis_meta": {
                        "model": outcome.get("model"),
                        "attempts": outcome.get("attempts"),
                        "from_cache": bool(outcome.get("from_cache")),
                        "usage": outcome.get("usage"),
                    },
                }
            )
        else:
            error = outcome.get("error") or {}
            errors.append(
                {
                    "stage": "ai_analysis",
                    "product_id": product_id,
                    "source": outcome.get("source") or product.get("source"),
                    "product_url": outcome.get("product_url") or product.get("product_url"),
                    "product_name": outcome.get("product_name") or product.get("product_name"),
                    "error_type": error.get("type", "unknown"),
                    "message": error.get("message", "Analysis did not complete."),
                    "attempts": outcome.get("attempts"),
                }
            )

    payload = {
        "project": PROJECT_NAME,
        "client": CLIENT_NAME,
        "generated_at": utc_now_iso(),
        "disclaimer": SCENARIO_DISCLAIMER,
        "data_sources": list(data_sources),
        "methodology": dict(methodology),
        "products": analyzed,
        "execution_summary": dict(execution_summary),
        "data_quality": dict(data_quality),
        "errors": errors,
    }
    return to_json_safe(payload)


def export_full_analysis(payload: Mapping[str, Any], path: Path = config.FULL_ANALYSIS_FILE) -> Path:
    """Write the JSON report (UTF-8, human-readable; NaN is rejected so the file is strict JSON)."""
    text = json.dumps(payload, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
    _atomic_write_text(path, text, encoding="utf-8")
    logger.info("Wrote %s (%d analyzed products, %d error records)", path, len(payload.get("products", [])), len(payload.get("errors", [])))
    return path


# ----------------------------------------------------------------------- verification
def verify_exports(csv_path: Path, json_path: Path, expected_product_ids: Sequence[str]) -> dict[str, Any]:
    """Re-open both deliverables and check they are valid and describe the same products."""
    problems: list[str] = []

    raw_bytes = csv_path.read_bytes()
    if not raw_bytes.startswith(b"\xef\xbb\xbf"):
        problems.append("CSV is missing the UTF-8 BOM (utf-8-sig) Excel relies on.")
    table = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = [c for c in REQUIRED_CSV_COLUMNS if c not in table.columns]
    if missing_columns:
        problems.append(f"CSV is missing required columns: {missing_columns}")
    csv_ids = table["product_id"].tolist() if "product_id" in table.columns else []
    if sorted(csv_ids) != sorted(str(pid) for pid in expected_product_ids):
        problems.append("CSV product_id set differs from the cleaned dataset.")
    if len(set(csv_ids)) != len(csv_ids):
        problems.append("CSV contains duplicate product_id values.")

    success_rows = table[table.get("analysis_status", pd.Series(dtype=str)) == STATUS_SUCCESS]
    for column in ("pros", "cons"):
        for cell in success_rows.get(column, []):
            try:
                parsed = json.loads(cell)
            except json.JSONDecodeError:
                problems.append(f"CSV column {column} holds invalid JSON: {cell[:60]!r}")
                break
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                problems.append(f"CSV column {column} is not a JSON array of strings: {cell[:60]!r}")
                break

    with json_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    for key in ("products", "execution_summary", "errors"):
        if key not in report:
            problems.append(f"JSON report is missing '{key}'.")
    json_ids = [item.get("product_id") for item in report.get("products", [])]
    if sorted(json_ids) != sorted(success_rows.get("product_id", pd.Series(dtype=str)).tolist()):
        problems.append("JSON products do not match the CSV rows marked analysis_status=success.")
    for item in report.get("products", []):
        analysis = item.get("analysis", {})
        if tuple(analysis.keys()) != config.ANALYSIS_FIELDS:
            problems.append(f"Analysis for {item.get('product_id')} does not have exactly the six required fields.")
            break

    result = {
        "csv_rows": int(len(table)),
        "csv_success_rows": int(len(success_rows)),
        "json_products": len(json_ids),
        "json_errors": len(report.get("errors", [])),
        "consistent": not problems,
        "problems": problems,
    }
    if problems:
        logger.error("Export verification found %d problem(s): %s", len(problems), problems)
    else:
        logger.info("Export verification passed: %d CSV rows, %d analyzed products in JSON.", result["csv_rows"], result["json_products"])
    return result


# ----------------------------------------------------------------------- market report
def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _money(value: Any, currency: Any) -> str:
    amount = _round_or_none(value)
    return "n/a" if amount is None else f"{amount:,.2f} {currency or ''}".strip()


def _shorten(text: Any, limit: int = 60) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_market_report(frame: pd.DataFrame, payload: Mapping[str, Any], competitor_names: Mapping[str, str]) -> str:
    """Render a client-facing Markdown summary from the enriched dataset (no extra API calls)."""
    summary = payload.get("execution_summary", {})
    analyzed = frame[frame["analysis_status"] == STATUS_SUCCESS]
    cost = summary.get("estimated_cost_usd")
    cost_text = f"${cost:.4f} (estimate from list prices, not an invoice)" if isinstance(cost, (int, float)) else "not available"

    lines = [
        "# Competitor Market Intelligence Report",
        "",
        f"*Generated {payload.get('generated_at')} for {CLIENT_NAME}.*",
        "",
        f"> {SCENARIO_DISCLAIMER}",
        "",
        "## 1. Executive summary",
        "",
        f"- **{len(frame)} products** collected from **{frame['source'].nunique()} competitor catalogues**; "
        f"**{len(analyzed)}** analysed by `{summary.get('model')}` "
        f"({summary.get('failed_analyses', 0)} failed).",
        f"- AI cost for this run: {cost_text}; {summary.get('total_tokens', 0):,} tokens over "
        f"{summary.get('total_api_calls', 0)} API calls.",
        f"- {int((frame['availability'] == 'in_stock').sum())} products are listed in stock, "
        f"{int((frame['availability'] == 'unknown').sum())} have no stock information.",
        f"- {int(frame['review_text'].notna().sum())} products carry review excerpts; for the rest, "
        "sentiment is reported as not determinable instead of being guessed.",
        "",
        "## 2. Competitor overview",
        "",
    ]

    def by_competitor(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """Groups ordered by the client-facing competitor label (Competitor A, B, C...)."""
        return sorted(data.groupby("source"), key=lambda item: competitor_names.get(item[0], item[0]))

    overview_rows = []
    for source, group in by_competitor(frame):
        priced = group.dropna(subset=["price"])
        currencies = ", ".join(sorted(priced["currency"].dropna().unique())) or "n/a"
        median = priced["price"].median() if len(priced) else None
        low, high = (priced["price"].min(), priced["price"].max()) if len(priced) else (None, None)
        avg_rating = group["rating"].astype("Float64").mean()
        overview_rows.append(
            [
                competitor_names.get(source, source),
                len(group),
                currencies,
                _money(median, "") if median is not None else "n/a",
                f"{_money(low, '')} – {_money(high, '')}" if low is not None else "n/a",
                f"{avg_rating:.2f}" if pd.notna(avg_rating) else "n/a",
                int((group["availability"] == "in_stock").sum()),
                int(group["review_text"].notna().sum()),
            ]
        )
    lines += [
        _markdown_table(
            ["Competitor", "Products", "Currency", "Median price", "Price range", "Avg rating", "In stock", "With reviews"],
            overview_rows,
        ),
        "",
        "## 3. Pricing observations (same category and currency only)",
        "",
    ]

    benchmarked = frame.dropna(subset=["price_vs_benchmark_median_pct"])
    for title, ascending in (("Priced furthest above their benchmark median", False), ("Priced furthest below their benchmark median", True)):
        top = benchmarked.sort_values("price_vs_benchmark_median_pct", ascending=ascending).head(5)
        rows = [
            [
                _shorten(row.product_name),
                competitor_names.get(row.source, row.source),
                _money(row.price, row.currency),
                _money(row.benchmark_median_price, row.currency),
                f"{row.price_vs_benchmark_median_pct:+.1f}%",
            ]
            for row in top.itertuples()
        ]
        lines += [f"**{title}**", "", _markdown_table(["Product", "Competitor", "Price", "Benchmark median", "Difference"], rows), ""]

    lines += ["## 4. Customer sentiment (products with review evidence)", ""]
    reviewed = analyzed[analyzed["review_text"].notna()]
    if reviewed.empty:
        lines += ["No product in this run carried review excerpts.", ""]
    else:
        rows = [[_shorten(row.product_name, 40), row.rating if pd.notna(row.rating) else "n/a", row.sentiment_summary] for row in reviewed.itertuples()]
        lines += [_markdown_table(["Product", "Rating", "AI sentiment summary"], rows), ""]

    lines += ["## 5. Recommended actions (largest price gaps per competitor)", ""]
    for source, group in by_competitor(analyzed):
        ranked = group.assign(gap=group["price_vs_benchmark_median_pct"].abs()).sort_values("gap", ascending=False, na_position="last").head(3)
        lines.append(f"**{competitor_names.get(source, source)}**")
        lines.append("")
        for row in ranked.itertuples():
            lines.append(f"- *{_shorten(row.product_name, 50)}* — {row.strategic_recommendation}")
        lines.append("")

    quality = payload.get("data_quality", {})
    lines += [
        "## 6. Data quality and limitations",
        "",
        f"- Raw records scraped: {quality.get('raw_records', 'n/a')}; after cleaning and de-duplication: {quality.get('final_records', len(frame))}.",
        f"- Missing ratings: {int(frame['rating'].isna().sum())}; missing descriptions: {int(frame['description'].isna().sum())}; "
        f"unknown availability: {int((frame['availability'] == 'unknown').sum())}.",
        "- Benchmarks compare products only within the same currency (no FX conversion) and category; "
        "the sandbox catalogues do not overlap, so peers usually come from the same competitor.",
        "- Ratings and prices are taken as published by the demo stores; AI text is generated from that evidence only and "
        "should be validated before real commercial decisions.",
        "",
    ]
    return "\n".join(lines)


def export_market_report(markdown: str, path: Path = config.MARKET_REPORT_FILE) -> Path:
    _atomic_write_text(path, markdown, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
