"""MarketScope competitor-intelligence pipeline — command-line entry point.

Run from the project root:

    python src/main.py                  # full run: scrape -> clean -> AI analysis -> export
    python src/main.py --skip-scrape    # reuse outputs/raw_products.json instead of re-scraping
    python src/main.py --dry-run        # everything except paid API calls (no API key needed)

See ``python src/main.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import config
from ai_analyzer import (
    ANALYSIS_SCHEMA,
    PROMPT_VERSION,
    AnalysisCache,
    ConfigurationError,
    FatalAPIError,
    ProductAnalyzer,
    load_openai_settings,
    preview_prompts,
)
from competitor_context import build_all_contexts
from cost_tracker import PricingError, UsageTracker, load_pricing_table
from data_cleaner import clean_product_data, dataframe_to_records, save_cleaned_csv
from exporter import (
    build_enriched_dataframe,
    build_full_analysis,
    build_market_report,
    export_enriched_csv,
    export_full_analysis,
    export_market_report,
    to_json_safe,
    utc_now_iso,
    verify_exports,
)
from http_client import PoliteHttpClient
from scraper import load_raw_products, save_raw_products, scrape_products

logger = logging.getLogger("pipeline")

EXIT_OK, EXIT_UNEXPECTED, EXIT_CONFIG, EXIT_FATAL_API, EXIT_NO_DATA, EXIT_EXPORT = 0, 1, 2, 3, 4, 5


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python src/main.py",
        description="Scrape competitor catalogues, clean the data with pandas, analyse every product "
        "with the OpenAI API (strict JSON schema) and export CSV/JSON deliverables.",
    )
    source_ids = [source.source_id for source in config.SOURCES]
    parser.add_argument("--sources", default=",".join(source_ids), help=f"comma-separated source ids (default: all of {source_ids})")
    parser.add_argument("--max-pages", type=int, default=None, help="override the per-listing page limit from config.py")
    parser.add_argument("--max-products-per-source", type=int, default=None, help="cap the products collected per source")
    parser.add_argument("--delay", type=float, default=config.REQUEST_DELAY_SECONDS, help="minimum seconds between requests to one host (robots.txt Crawl-delay still applies)")
    parser.add_argument("--skip-scrape", action="store_true", help="reuse the raw scrape in --raw-input instead of scraping again")
    parser.add_argument("--raw-input", type=Path, default=config.RAW_PRODUCTS_FILE, help="raw products JSON used with --skip-scrape")
    parser.add_argument("--dry-run", action="store_true", help="stop before the OpenAI calls and write prompt previews (no API key needed)")
    parser.add_argument("--limit", type=int, default=None, help="analyse only the first N products (smoke test; the deliverable analyses all)")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached analyses and call the API for every product")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="where deliverables are written (default: outputs/)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    unknown = sorted(set(args.sources.split(",")) - set(source_ids))
    if unknown:
        parser.error(f"unknown source id(s): {unknown}; choose from {source_ids}")
    for name in ("max_pages", "max_products_per_source", "limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def configure_logging(level: str, log_file: Path) -> None:
    """Console + file logging. The log file is part of the run evidence (delays, pages, retries)."""
    for stream in (sys.stdout, sys.stderr):
        try:  # never crash on a console that cannot encode "£" or "★"
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("urllib3", "httpx", "httpx2", "httpcore", "httpcore2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------------- stages
def stage_scrape(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase 1 — collect raw product records (or reload a previous raw scrape)."""
    if args.skip_scrape:
        logger.info("Phase 1 skipped: loading raw records from %s", args.raw_input)
        return load_raw_products(args.raw_input)

    wanted = set(args.sources.split(","))
    sources = [source for source in config.SOURCES if source.source_id in wanted]
    logger.info("Phase 1: scraping %d source(s): %s", len(sources), ", ".join(s.display_name for s in sources))
    client = PoliteHttpClient(default_delay=args.delay)
    result = scrape_products(sources=sources, client=client, max_pages=args.max_pages, max_products=args.max_products_per_source)
    save_raw_products(result, args.output_dir / config.RAW_PRODUCTS_FILE.name)
    report = result.as_dict()
    report.pop("records", None)
    return list(result.records), report


def build_data_sources(scrape_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Source documentation for the JSON report: where data came from and on what basis."""
    reports = {item.get("source_id"): item for item in scrape_report.get("source_reports", [])}
    documented = []
    for source in config.SOURCES:
        if source.source_id not in reports:
            continue
        report = reports[source.source_id]
        documented.append(
            {
                "source_id": source.source_id,
                "competitor_alias": source.competitor_alias,
                "display_name": source.display_name,
                "base_url": source.base_url,
                "robots_txt": source.base_url.rstrip("/") + "/robots.txt",
                "listing_start_urls": [start.url for start in source.listing_starts],
                "access_basis": source.access_notes,
                "products_collected": report.get("products_collected"),
                "product_pages_fetched": report.get("product_pages_fetched"),
                "listings": report.get("listings"),
            }
        )
    return documented


def build_methodology(cleaning_report: dict[str, Any], settings: Any | None) -> dict[str, Any]:
    return {
        "pipeline": "Requests + BeautifulSoup scraping -> pandas cleaning/validation -> deterministic same-currency price benchmark -> OpenAI Chat Completions with strict JSON Schema -> CSV/JSON export",
        "cleaning_policies": cleaning_report.get("policies", {}),
        "benchmark_policy": (
            "Peers must share the product's currency (no FX conversion) and category; when fewer than 3 such "
            "peers exist the benchmark falls back to the same source and currency, and with no peers the AI is "
            "told comparison data is insufficient."
        ),
        "prompt_version": PROMPT_VERSION,
        "model_parameters": None
        if settings is None
        else {
            "model": settings.model,
            "api_endpoint": settings.endpoint_label,
            "max_output_tokens": settings.max_output_tokens,
            "reasoning_effort": settings.reasoning_effort,
            "temperature": settings.temperature,
        },
        "analysis_json_schema": ANALYSIS_SCHEMA,
    }


def build_execution_summary(
    tracker_summary: dict[str, Any],
    *,
    raw_count: int,
    clean_count: int,
    outcomes: Sequence[dict[str, Any]],
    scrape_report: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    analyzed = sum(1 for outcome in outcomes if outcome.get("status") == "success")
    failed = sum(1 for outcome in outcomes if outcome.get("status") != "success")
    # Key order mirrors the assignment's example, followed by the detailed breakdown.
    summary: dict[str, Any] = {
        "model": tracker_summary.get("model"),
        "total_products_scraped": raw_count,
        "total_products_after_cleaning": clean_count,
        "total_products_analyzed": analyzed,
        "failed_analyses": failed,
        "not_analyzed": clean_count - len(outcomes),
        "total_api_calls": tracker_summary.get("total_api_calls"),
        "input_tokens": tracker_summary.get("input_tokens"),
        "output_tokens": tracker_summary.get("output_tokens"),
        "total_tokens": tracker_summary.get("total_tokens"),
        "estimated_cost_usd": tracker_summary.get("estimated_cost_usd"),
        "pricing_reference_date": tracker_summary.get("pricing_reference_date"),
    }
    summary.update({key: value for key, value in tracker_summary.items() if key not in summary})
    summary["scraping"] = {"http": scrape_report.get("http_stats"), "started_at": scrape_report.get("started_at"), "finished_at": scrape_report.get("finished_at")}
    summary["pipeline_runtime_seconds"] = round(time.monotonic() - started, 1)
    return summary


def collect_stage_errors(scrape_report: dict[str, Any], rejected_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scraping and cleaning problems that belong in the report's `errors` section."""
    errors = [dict(error, stage=error.get("stage", "scraping")) for error in scrape_report.get("errors", [])]
    for rejected in rejected_records:
        record = rejected.get("record", {})
        errors.append(
            {
                "stage": "cleaning",
                "source": record.get("source"),
                "product_url": record.get("product_url"),
                "product_name": record.get("product_name"),
                "error_type": "rejected_record",
                "message": rejected.get("reason"),
            }
        )
    return errors


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 — scraping
    raw_records, scrape_report = stage_scrape(args)
    if not raw_records:
        logger.error("No product records were collected; nothing to clean or analyse. Check the log for network/robots errors.")
        return EXIT_NO_DATA
    logger.info("Phase 1 complete: %d raw product records.", len(raw_records))

    # Phase 2 — cleaning, validation and price benchmarks
    cleaning = clean_product_data(raw_records)
    frame = cleaning.dataframe
    save_cleaned_csv(frame, output_dir / config.CLEANED_PRODUCTS_FILE.name)
    products = dataframe_to_records(frame)
    contexts = build_all_contexts(frame)
    logger.info(
        "Phase 2 complete: %d raw -> %d clean records (%d rejected, %d duplicates removed).",
        len(raw_records),
        len(products),
        len(cleaning.rejected_records),
        cleaning.report.get("duplicates_removed_same_url", 0) + cleaning.report.get("duplicates_removed_same_content", 0),
    )
    if not products:
        logger.error("Every record was rejected during cleaning; see %s.", output_dir / config.CLEANED_PRODUCTS_FILE.name)
        return EXIT_NO_DATA

    to_analyze = products[: args.limit] if args.limit else products
    if args.limit and args.limit < len(products):
        logger.warning("--limit %d: only %d of %d products will be analysed (smoke test).", args.limit, len(to_analyze), len(products))

    if args.dry_run:
        preview_path = output_dir / "prompt_preview.json"
        previews = preview_prompts(to_analyze, contexts, limit=min(3, len(to_analyze)))
        preview_path.write_text(json.dumps(to_json_safe(previews), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Dry run: no API calls made. Prompt previews written to %s.", preview_path)
        return EXIT_OK

    # Phase 3 — AI analysis
    try:
        settings = load_openai_settings()
        pricing_table = load_pricing_table(config.PRICING_FILE)
    except (ConfigurationError, PricingError) as error:
        logger.error("%s", error)
        return EXIT_CONFIG
    tracker = UsageTracker(pricing_table)
    cache = None if args.no_cache else AnalysisCache(config.ANALYSIS_CACHE_FILE)
    analyzer = ProductAnalyzer(settings, tracker, cache=cache)
    logger.info("Phase 3: analysing %d products with %s via %s.", len(to_analyze), settings.model, settings.endpoint_label)
    try:
        outcomes = [outcome.as_dict() for outcome in analyzer.analyze_all(to_analyze, contexts)]
    except FatalAPIError as error:
        logger.error("AI analysis stopped: %s", error)
        logger.error("Existing deliverables were left untouched. Successful analyses are cached, so a re-run will not pay for them again.")
        return EXIT_FATAL_API

    # Phase 4 — delivery
    summary = build_execution_summary(
        tracker.summary(settings.model),
        raw_count=len(raw_records),
        clean_count=len(products),
        outcomes=outcomes,
        scrape_report=scrape_report,
        started=started,
    )
    enriched = build_enriched_dataframe(products, outcomes, contexts)
    payload = build_full_analysis(
        products=products,
        outcomes=outcomes,
        contexts=contexts,
        execution_summary=summary,
        data_sources=build_data_sources(scrape_report),
        data_quality=cleaning.report,
        methodology=build_methodology(cleaning.report, settings),
        extra_errors=collect_stage_errors(scrape_report, cleaning.rejected_records),
    )
    payload["api_calls"] = to_json_safe(tracker.calls_as_records())
    csv_path = export_enriched_csv(enriched, output_dir / config.ENRICHED_CSV_FILE.name)
    json_path = export_full_analysis(payload, output_dir / config.FULL_ANALYSIS_FILE.name)
    competitor_names = {s.source_id: f"{s.competitor_alias} ({s.display_name})" for s in config.SOURCES}
    export_market_report(build_market_report(enriched, payload, competitor_names), output_dir / config.MARKET_REPORT_FILE.name)
    verification = verify_exports(csv_path, json_path, [p["product_id"] for p in products])

    print_run_summary(summary, csv_path, json_path, verification)
    return EXIT_OK if verification["consistent"] else EXIT_EXPORT


def print_run_summary(summary: dict[str, Any], csv_path: Path, json_path: Path, verification: dict[str, Any]) -> None:
    cost = summary.get("estimated_cost_usd")
    lines = [
        "",
        "=" * 64,
        "MarketScope pipeline finished",
        f"  Products scraped / cleaned : {summary['total_products_scraped']} / {summary['total_products_after_cleaning']}",
        f"  Analysed / failed          : {summary['total_products_analyzed']} / {summary['failed_analyses']}",
        f"  Model                      : {summary['model']}",
        f"  API calls / total tokens   : {summary['total_api_calls']} / {summary['total_tokens']:,}",
        f"  Estimated cost (USD)       : {'n/a' if cost is None else f'${cost:.6f}'} (estimate, not an invoice)",
        f"  Enriched CSV               : {csv_path}",
        f"  Full analysis JSON         : {json_path}",
        f"  Export verification        : {'passed' if verification['consistent'] else 'FAILED - see log'}",
        "=" * 64,
    ]
    logger.info("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.output_dir / config.LOG_FILE.name)
    logger.info("Run started %s (outputs -> %s)", utc_now_iso(), args.output_dir)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return EXIT_UNEXPECTED
    except Exception:  # last-resort guard: log the traceback instead of dying silently
        logger.exception("Unexpected error - the pipeline stopped.")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
