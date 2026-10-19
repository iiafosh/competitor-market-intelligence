"""Central configuration for the MarketScope competitor-intelligence pipeline.

Every tunable value lives here (or in an environment variable) so the pipeline can be
pointed at new sources, delays or models without touching business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_FILE = PROJECT_ROOT / ".env"
PRICING_FILE = CONFIG_DIR / "model_pricing.json"

RAW_PRODUCTS_FILE = OUTPUT_DIR / "raw_products.json"
CLEANED_PRODUCTS_FILE = OUTPUT_DIR / "cleaned_products.csv"
ENRICHED_CSV_FILE = OUTPUT_DIR / "enriched_products.csv"
FULL_ANALYSIS_FILE = OUTPUT_DIR / "full_analysis.json"
MARKET_REPORT_FILE = OUTPUT_DIR / "market_report.md"
LOG_FILE = OUTPUT_DIR / "pipeline.log"
ANALYSIS_CACHE_FILE = CACHE_DIR / "analysis_cache.json"

# Secrets live only in the local, git-ignored .env file (see .env.example).
# Real environment variables take precedence over values in .env.
load_dotenv(ENV_FILE, override=False)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        return default


# ------------------------------------------------------------------- HTTP / scraping
# Truthful identification: who we are and why we are crawling.
USER_AGENT = "MarketScopeCapstone/1.0 (Educational Project; competitor-intelligence capstone; honours robots.txt)"
ROBOTS_USER_AGENT_TOKEN = "MarketScopeCapstone"  # product token matched against robots.txt groups
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT = (10.0, 20.0)  # (connect, read) seconds
REQUEST_DELAY_SECONDS = _env_float("SCRAPER_DELAY_SECONDS", 1.5)  # minimum gap between requests to one host
MAX_RETRIES = _env_int("SCRAPER_MAX_RETRIES", 3)  # retries after the first attempt (transient errors only)
BACKOFF_BASE_SECONDS = 2.0  # exponential backoff: base * 2 ** (attempt - 1) + jitter
BACKOFF_MAX_SECONDS = 30.0
MAX_RETRY_AFTER_SECONDS = 60.0  # upper bound when honouring a server's Retry-After header


@dataclass(frozen=True)
class ListingStart:
    """A listing (catalogue) URL to paginate from, plus the category label it represents."""

    url: str
    category: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    """Static description of one competitor catalogue."""

    source_id: str  # stable identifier written to the `source` column
    display_name: str  # human-readable store name
    competitor_alias: str  # how the store is referred to in client deliverables
    base_url: str
    listing_starts: tuple[ListingStart, ...]
    max_pages_per_listing: int  # hard stop for pagination, per listing start URL
    max_products: int | None  # optional cap on products collected from this source
    fetch_product_pages: bool  # visit product detail pages for description/stock/reviews
    min_delay_seconds: float  # politeness floor for this host (robots.txt Crawl-delay may raise it)
    access_notes: str  # documented basis for automated access


SOURCES: tuple[SourceConfig, ...] = (
    SourceConfig(
        source_id="books_toscrape",
        display_name="Books to Scrape",
        competitor_alias="Competitor A",
        base_url="https://books.toscrape.com/",
        listing_starts=(
            ListingStart(
                "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
                category="Mystery",
            ),
        ),
        max_pages_per_listing=5,
        max_products=None,
        fetch_product_pages=True,
        min_delay_seconds=1.5,
        access_notes=(
            "Public sandbox built for scraping practice ('This is a demo website for web scraping "
            "purposes'); no robots.txt (HTTP 404 = no restrictions under RFC 9309). Prices and "
            "ratings are randomly assigned by the site."
        ),
    ),
    SourceConfig(
        source_id="webscraper_io",
        display_name="WebScraper.io Test Shop",
        competitor_alias="Competitor B",
        base_url="https://webscraper.io/",
        listing_starts=(
            ListingStart(
                "https://webscraper.io/test-sites/e-commerce/static/computers/laptops",
                category="Laptops",
            ),
        ),
        max_pages_per_listing=3,
        max_products=None,
        fetch_product_pages=False,
        min_delay_seconds=1.5,
        access_notes=(
            "Public e-commerce test site provided for scraping practice. robots.txt only disallows "
            "/test-sites/product/, /test-sites/pagination/, /test-sites/scroll/ and "
            "/test-sites/load-more/; the static e-commerce catalogue used here is allowed."
        ),
    ),
    SourceConfig(
        source_id="web_scraping_dev",
        display_name="web-scraping.dev Store",
        competitor_alias="Competitor C",
        base_url="https://web-scraping.dev/",
        listing_starts=(
            ListingStart("https://web-scraping.dev/products?category=apparel", category="Apparel"),
            ListingStart("https://web-scraping.dev/products?category=consumables", category="Consumables"),
            ListingStart("https://web-scraping.dev/products?category=household", category="Household"),
        ),
        max_pages_per_listing=5,
        max_products=None,
        fetch_product_pages=True,
        min_delay_seconds=2.0,
        access_notes=(
            "Scrapfly's public scraping sandbox. robots.txt disallows only /robots-disallowed for "
            "all agents and sets Crawl-delay: 2, which the scraper honours. Reviews are the sandbox's "
            "sample reviews embedded in the product page (first page only; the CSRF-protected "
            "review API is deliberately not used)."
        ),
    ),
)

# ------------------------------------------------------------------- data standards
AVAILABILITY_VALUES = ("in_stock", "out_of_stock", "unknown")
RATING_MIN, RATING_MAX = 1, 5
REVIEW_SEPARATOR = " || "  # separates individual excerpts inside `review_text`

# Required cleaned-dataset columns, in export order (assignment section 7.6).
REQUIRED_CLEAN_COLUMNS = (
    "product_name",
    "price",
    "currency",
    "rating",
    "availability",
    "description",
    "source",
    "product_url",
    "scraped_at",
    "review_text",
)

# The six fields every AI analysis must contain (assignment section 8.5).
ANALYSIS_FIELDS = (
    "product_name",
    "sentiment_summary",
    "pros",
    "cons",
    "pricing_verdict",
    "strategic_recommendation",
)

# ------------------------------------------------------------------------- OpenAI
# gpt-6-luna: current small flagship model; supports Chat Completions + strict Structured
# Outputs, and is OpenAI's named replacement for the deprecated gpt-4.1-mini / gpt-4o-mini.
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 1500)
OPENAI_TIMEOUT_SECONDS = _env_float("OPENAI_TIMEOUT_SECONDS", 60.0)
OPENAI_MAX_ATTEMPTS = _env_int("OPENAI_MAX_ATTEMPTS", 4)  # per product, transient errors included

# Prompt budget controls (cost efficiency: never send raw HTML or unbounded text).
PROMPT_MAX_DESCRIPTION_CHARS = 700
PROMPT_MAX_REVIEWS = 5
PROMPT_MAX_REVIEW_CHARS = 300
PROMPT_MAX_PEERS = 5
