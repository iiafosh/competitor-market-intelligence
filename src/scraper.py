"""Phase 1 - scrape competitor catalogues into raw product records.

One scraper class per source, registered under its ``SourceConfig.source_id``, turns listing and
product pages into records that follow the raw-record contract (:data:`RECORD_FIELDS`, every key
always present). The base class owns everything that is not site specific: pagination and its stop
conditions, product-page enrichment, error isolation (one bad card, page or product never stops a
source; one bad source never stops the run) and reporting. All network access goes through
:class:`http_client.PoliteHttpClient` (robots.txt, per-host delays, retries, caching).
"""

from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Protocol, TypeVar
from collections.abc import Iterable, Mapping
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup
from bs4.element import Tag

import config
from http_client import FetchError, PoliteHttpClient, RobotsDisallowedError

logger = logging.getLogger(__name__)

# Raw-record schema (see CONTRACT.md, stage 1). Every record carries every key.
RECORD_FIELDS: tuple[str, ...] = (
    "product_name",
    "price_raw",
    "currency_raw",
    "original_price_raw",
    "listing_price_raw",
    "rating_raw",
    "availability_raw",
    "description",
    "source",
    "product_url",
    "scraped_at",
    "review_text",
    "review_count",
    "review_source",
    "category",
    "source_product_id",
    "listing_page_url",
    "extraction_warnings",
)
# Fields whose gaps are counted per source in the scrape report.
REPORTED_FIELDS: tuple[str, ...] = (
    "product_name",
    "price_raw",
    "currency_raw",
    "rating_raw",
    "availability_raw",
    "description",
    "review_text",
    "review_count",
)
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Why pagination of a listing stopped (ListingReport.stop_reason).
STOP_LAST_PAGE = "last_page"  # no further page: natural end of pagination
STOP_PAGE_LIMIT = "page_limit_reached"  # configured maximum pages per listing reached
STOP_REPEATED_URL = "repeated_page_url"  # next link points to a page already visited
STOP_EMPTY_LISTING = "empty_listing"  # page without product cards
STOP_PRODUCT_CAP = "product_cap_reached"  # maximum products for the source collected
STOP_FETCH_FAILED = "page_fetch_failed"  # listing page could not be downloaded
STOP_ROBOTS = "robots_disallowed"  # listing page forbidden by robots.txt
STOP_PARSE_FAILED = "page_parse_failed"  # listing or pagination markup could not be parsed

REVIEW_SOURCE_JSON_LD = "web-scraping.dev product page (schema.org JSON-LD sample reviews, first page only)"
REVIEW_SOURCE_EMBEDDED = "web-scraping.dev product page (embedded reviews-data JSON sample reviews, first page only)"

_PRODUCT_ID_RE = re.compile(r"/product/(\d+)")
_PRICE_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WSD_PAGING_RE = re.compile(r"page\s+(\d+)\s+of\s+total\s+\d+\s+results?\s+in\s+(\d+)\s+pages?", re.IGNORECASE)


class PageStructureError(ValueError):
    """A page lacks the structure its parser relies on."""


class PageFetcher(Protocol):
    """What the scrapers need from an HTTP client (PoliteHttpClient or a test double)."""

    def get_soup(self, url: str, *, min_delay: float | None = None) -> BeautifulSoup: ...

    def stats_as_dict(self) -> dict[str, Any]: ...


class ReviewExcerpt(NamedTuple):
    """One review as published by the source."""

    text: str
    rating: str | None = None
    best_rating: str | None = None


# --------------------------------------------------------------------------- reports
@dataclass
class ListingReport:
    """Pagination outcome for one ListingStart."""

    start_url: str
    category: str | None
    pages_visited: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    products_found: int = 0


@dataclass
class SourceReport:
    """Per-source summary written to the scrape report."""

    source_id: str
    display_name: str
    competitor_alias: str
    base_url: str
    listings: list[ListingReport] = field(default_factory=list)
    product_pages_fetched: int = 0
    products_collected: int = 0
    records_with_warnings: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def for_source(cls, source: config.SourceConfig) -> SourceReport:
        return cls(source.source_id, source.display_name, source.competitor_alias, source.base_url)

    def summarise(self, records: list[dict[str, Any]]) -> None:
        """Refresh the counters derived from the collected records."""
        self.products_collected = len(records)
        self.records_with_warnings = sum(1 for record in records if record["extraction_warnings"])
        self.missing_fields = {name: sum(1 for record in records if record[name] is None) for name in REPORTED_FIELDS}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScrapeResult:
    """Everything the scraping stage produced: records plus an auditable report."""

    records: list[dict[str, Any]]
    source_reports: list[SourceReport]
    errors: list[dict[str, Any]]  # {stage, source, url, error_type, message}
    started_at: str
    finished_at: str
    http_stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe representation (records included)."""
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_records": len(self.records),
            "records_per_source": {report.source_id: report.products_collected for report in self.source_reports},
            "source_reports": [report.as_dict() for report in self.source_reports],
            "errors": [dict(error) for error in self.errors],
            "http_stats": self.http_stats,
            "records": [
                dict(record, extraction_warnings=list(record["extraction_warnings"])) for record in self.records
            ],
        }


# --------------------------------------------------------------------------- record helpers
def utc_timestamp() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def new_record(source_id: str, product_url: str, listing_page_url: str, scraped_at: str) -> dict[str, Any]:
    """A raw record with every contract key present (None until extracted)."""
    record: dict[str, Any] = dict.fromkeys(RECORD_FIELDS)
    record.update(
        source=source_id,
        product_url=product_url,
        listing_page_url=listing_page_url,
        scraped_at=scraped_at,
        extraction_warnings=[],
    )
    return record


def format_review_excerpts(excerpts: Iterable[ReviewExcerpt]) -> str | None:
    """Join excerpts with ``config.REVIEW_SEPARATOR``, each prefixed ``[rating/best]`` when rated."""
    parts: list[str] = []
    for excerpt in excerpts:
        body = _clean_text(excerpt.text)
        if body is None:
            continue
        while "||" in body:  # keep the separator unambiguous
            body = body.replace("||", "|")
        prefix = f"[{excerpt.rating}/{excerpt.best_rating or 5}] " if excerpt.rating else ""
        parts.append(prefix + body)
    return config.REVIEW_SEPARATOR.join(parts) if parts else None


def error_entry(stage: str, source_id: str, url: str | None, error: BaseException) -> dict[str, Any]:
    """Uniform, JSON-safe error description for the scrape report."""
    return {"stage": stage, "source": source_id, "url": url, "error_type": type(error).__name__, "message": str(error)}


def _apply_updates(record: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Merge parser output into a record: non-None values win, warnings accumulate."""
    for key, value in updates.items():
        if key not in RECORD_FIELDS:
            raise KeyError(f"parser produced unknown record field {key!r}")
        if key == "extraction_warnings":
            record["extraction_warnings"].extend(value)
        elif value is not None:
            record[key] = value


# --------------------------------------------------------------------------- HTML helpers
def _clean_text(value: Any) -> str | None:
    """Collapse whitespace; None for missing, blank or non-scalar values."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = " ".join(str(value).split())
    return text or None


def _element_text(element: Any) -> str | None:
    return _clean_text(element.get_text(" ")) if isinstance(element, Tag) else None


def _attribute(element: Any, name: str) -> str | None:
    if not isinstance(element, Tag):
        return None
    value = element.get(name)
    if isinstance(value, list):  # multi-valued attributes such as class/rel
        value = " ".join(value)
    return _clean_text(value)


def _absolute_url(base_url: str, href: str | None) -> str | None:
    """Resolve ``href`` against the page URL and drop any fragment."""
    if not href:
        return None
    return urldefrag(urljoin(base_url, href)).url


def _product_id_from_url(url: str | None) -> str | None:
    match = _PRODUCT_ID_RE.search(url or "")
    return match.group(1) if match else None


def _int_or_raw(value: Any) -> int | str | None:
    """Whole numbers become int (``"1,024"`` -> 1024); anything else stays as the original text."""
    text = _clean_text(value)
    if text is None:
        return None
    digits = text.replace(",", "")
    return int(digits) if digits.isascii() and digits.isdigit() else text


def _rating_token(value: Any) -> str | None:
    """``5`` / ``"5"`` / ``"5.0"`` -> ``"5"``; ``"4.5"`` stays; None when missing."""
    text = _clean_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return str(int(number)) if number.is_integer() else text


def _price_number(price: str | None) -> float | None:
    match = _PRICE_NUMBER_RE.search(price or "")
    return float(match.group(0).replace(",", "")) if match else None


def _star_rating_token(element: Any) -> str | None:
    """``<p class="star-rating Three">`` -> ``"Three"``."""
    if not isinstance(element, Tag):
        return None
    tokens = [name for name in element.get("class") or [] if name != "star-rating"]
    return tokens[0] if tokens else None


def _microdata_rating(scope: Tag) -> str | None:
    """webscraper.io rating: ``data-rating`` attribute, else the number of rendered stars."""
    rating = _attribute(scope.select_one("[data-rating]"), "data-rating")
    if rating:
        return rating
    ratings = scope.select_one("div.ratings")
    stars = len(ratings.select(".ws-icon-star")) if isinstance(ratings, Tag) else 0
    return str(stars) if stars else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_ld_product(soup: BeautifulSoup) -> Mapping[str, Any] | None:
    """The first schema.org Product object in the page's JSON-LD blocks."""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text() or "null")
        except ValueError:
            logger.debug("Skipping malformed JSON-LD block")
            continue
        for candidate in _flatten_json_ld(data):
            types = candidate.get("@type")
            for type_name in types if isinstance(types, list) else [types]:
                if isinstance(type_name, str) and type_name.rsplit("/", 1)[-1] == "Product":
                    return candidate
    return None


def _flatten_json_ld(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return [item for element in data for item in _flatten_json_ld(element)]
    if isinstance(data, Mapping):
        graph = data.get("@graph")
        return _flatten_json_ld(graph) if graph is not None else [data]
    return []


def _first_offer(offers: Any) -> Mapping[str, Any]:
    if isinstance(offers, list):
        return next((offer for offer in offers if isinstance(offer, Mapping)), {})
    return _mapping(offers)


def _aggregate_rating_token(aggregate: Mapping[str, Any]) -> str | None:
    """``ratingValue`` as published; ``"value/best"`` when the scale is not out of 5."""
    value = _clean_text(aggregate.get("ratingValue"))
    if value is None:
        return None
    best = _clean_text(aggregate.get("bestRating"))
    try:
        other_scale = best is not None and float(best) != 5.0
    except ValueError:
        other_scale = False
    return f"{value}/{best}" if other_scale else value


def _json_ld_reviews(product: Mapping[str, Any]) -> list[ReviewExcerpt]:
    reviews = product.get("review")
    excerpts: list[ReviewExcerpt] = []
    for review in reviews if isinstance(reviews, list) else [reviews]:
        if not isinstance(review, Mapping):
            continue
        text = _clean_text(review.get("reviewBody") or review.get("description"))
        if text is None:
            continue
        rating = _mapping(review.get("reviewRating"))
        excerpts.append(
            ReviewExcerpt(text, _rating_token(rating.get("ratingValue")), _rating_token(rating.get("bestRating")))
        )
    return excerpts


def _embedded_reviews(soup: BeautifulSoup) -> list[ReviewExcerpt]:
    """Reviews from ``<script id="reviews-data" type="application/json">`` (first page only)."""
    script = soup.select_one("script#reviews-data")
    if not isinstance(script, Tag):
        return []
    try:
        data = json.loads(script.string or script.get_text() or "[]")
    except ValueError:
        logger.debug("Skipping malformed reviews-data block")
        return []
    excerpts: list[ReviewExcerpt] = []
    for review in data if isinstance(data, list) else []:
        text = _clean_text(review.get("text")) if isinstance(review, Mapping) else None
        if text is not None:
            excerpts.append(ReviewExcerpt(text, _rating_token(review.get("rating"))))
    return excerpts


# --------------------------------------------------------------------------- base scraper
class BaseScraper(ABC):
    """Pagination, product-page enrichment, error isolation and reporting shared by every source.

    Subclasses provide the site-specific parsing hooks: :meth:`parse_card`, :meth:`find_next_page`
    and, when the source has useful product pages, :meth:`parse_product_page`.
    """

    source_id: ClassVar[str]
    card_selector: ClassVar[str]
    expected_listing_fields: ClassVar[tuple[str, ...]] = ("product_name", "price_raw")
    expected_product_page_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        source: config.SourceConfig,
        client: PageFetcher,
        *,
        max_pages: int | None = None,
        max_products: int | None = None,
    ) -> None:
        if source.source_id != self.source_id:
            raise ValueError(f"{type(self).__name__} handles {self.source_id!r}, not {source.source_id!r}")
        self.source = source
        self.client = client
        self.max_pages = max_pages if max_pages is not None else source.max_pages_per_listing
        self.max_products = max_products if max_products is not None else source.max_products
        self.records: list[dict[str, Any]] = []
        self.report = SourceReport.for_source(source)

    # ----------------------------------------------------------------- site-specific hooks
    def find_cards(self, soup: BeautifulSoup) -> list[Tag]:
        """Product cards on a listing page."""
        return [card for card in soup.select(self.card_selector) if isinstance(card, Tag)]

    @abstractmethod
    def parse_card(self, card: Tag, page_url: str) -> dict[str, Any]:
        """Record fields visible on one listing card (must include ``product_url``)."""

    @abstractmethod
    def find_next_page(self, soup: BeautifulSoup, page_url: str) -> str | None:
        """Absolute URL of the next listing page, or None on the last page."""

    def parse_product_page(self, soup: BeautifulSoup, record: Mapping[str, Any]) -> dict[str, Any]:
        """Record updates from a product page; ``record`` holds the listing data. Default: none."""
        return {}

    # ----------------------------------------------------------------- orchestration
    def scrape(self) -> list[dict[str, Any]]:
        """Scrape every listing of the source. Records are also kept in ``self.records``."""
        sid = self.source.source_id
        logger.info(
            "[%s] Scraping %s: %d listing(s), at most %d page(s) each", sid, self.source.display_name,
            len(self.source.listing_starts), self.max_pages,
        )
        for listing in self.source.listing_starts:
            listing_report = ListingReport(start_url=listing.url, category=listing.category)
            self.report.listings.append(listing_report)
            if self._product_cap_reached():
                listing_report.stop_reason = STOP_PRODUCT_CAP
                continue
            self._scrape_listing(listing, listing_report)
        self.report.summarise(self.records)
        logger.info(
            "[%s] Finished: %d record(s), %d product page(s) fetched, %d error(s)", sid, len(self.records),
            self.report.product_pages_fetched, len(self.report.errors),
        )
        return self.records

    def _scrape_listing(self, listing: config.ListingStart, report: ListingReport) -> None:
        sid = self.source.source_id
        visited: set[str] = set()
        page_url = urldefrag(listing.url).url
        while True:
            soup = self._fetch_listing_page(page_url, report)
            if soup is None:
                return
            visited.add(page_url)
            report.pages_visited.append(page_url)
            fetched_at = utc_timestamp()
            try:
                cards = self.find_cards(soup)
            except Exception as exc:  # defensive: unexpected markup must not abort the source
                self._record_error("listing_page_parse", page_url, exc)
                report.stop_reason = STOP_PARSE_FAILED
                return
            logger.info(
                "[%s] Listing page %d: %s (%d product cards)", sid, len(report.pages_visited), page_url, len(cards)
            )
            if not cards:
                logger.warning(
                    "[%s] No product cards on %s (empty listing or changed markup); stopping this listing",
                    sid, page_url,
                )
                report.stop_reason = STOP_EMPTY_LISTING
                return
            for card in cards:
                if self._product_cap_reached():
                    report.stop_reason = STOP_PRODUCT_CAP
                    logger.info("[%s] Product cap of %s reached", sid, self.max_products)
                    return
                record = self._record_from_card(card, page_url, listing, fetched_at)
                if record is not None:
                    self.records.append(record)
                    report.products_found += 1
            try:
                next_url = self.find_next_page(soup, page_url)
            except Exception as exc:  # defensive: broken pagination markup ends this listing only
                self._record_error("pagination", page_url, exc)
                report.stop_reason = STOP_PARSE_FAILED
                return
            stop_reason = self._stop_reason(next_url, visited, len(report.pages_visited))
            if stop_reason is not None or next_url is None:
                report.stop_reason = stop_reason or STOP_LAST_PAGE
                logger.info(
                    "[%s] Listing %s done after %d page(s): %s",
                    sid, listing.url, len(report.pages_visited), report.stop_reason,
                )
                return
            logger.info("[%s] Next page: %s", sid, next_url)
            page_url = next_url

    def _stop_reason(self, next_url: str | None, visited: set[str], pages_done: int) -> str | None:
        if next_url is None:
            return STOP_LAST_PAGE
        if self._product_cap_reached():
            return STOP_PRODUCT_CAP
        if next_url in visited:
            return STOP_REPEATED_URL
        if pages_done >= self.max_pages:
            return STOP_PAGE_LIMIT
        return None

    def _fetch_listing_page(self, page_url: str, report: ListingReport) -> BeautifulSoup | None:
        try:
            return self.client.get_soup(page_url, min_delay=self.source.min_delay_seconds)
        except FetchError as exc:
            self._record_error("listing_page", page_url, exc)
            report.stop_reason = STOP_ROBOTS if isinstance(exc, RobotsDisallowedError) else STOP_FETCH_FAILED
            return None

    def _record_from_card(
        self, card: Tag, page_url: str, listing: config.ListingStart, fetched_at: str
    ) -> dict[str, Any] | None:
        try:
            item = self.parse_card(card, page_url)
        except Exception as exc:  # defensive: one malformed card must not cost the rest of the page
            self._record_error("listing_card", page_url, exc)
            return None
        product_url = item.get("product_url")
        if not product_url:
            missing_link = PageStructureError("product card without a product link skipped")
            self._record_error("listing_card", page_url, missing_link)
            return None
        record = new_record(self.source.source_id, product_url, page_url, fetched_at)
        record["category"] = listing.category
        _apply_updates(record, item)
        if self.source.fetch_product_pages:
            self._enrich_from_product_page(record)
        if record["review_text"] is None:
            record["review_source"] = None
        self._flag_missing_fields(record)
        return record

    def _enrich_from_product_page(self, record: dict[str, Any]) -> None:
        """Merge product-page data into ``record``; on failure keep the listing data and warn."""
        url = record["product_url"]
        logger.info("[%s] Product %d: %s", self.source.source_id, len(self.records) + 1, url)
        try:
            soup = self.client.get_soup(url, min_delay=self.source.min_delay_seconds)
        except FetchError as exc:
            record["extraction_warnings"].append(f"product page fetch failed: {exc.summary}")
            self._record_error("product_page", url, exc)
            return
        record["scraped_at"] = utc_timestamp()
        self.report.product_pages_fetched += 1
        try:
            updates = self.parse_product_page(soup, record)
        except Exception as exc:  # defensive: unexpected product markup keeps the listing data
            record["extraction_warnings"].append(f"product page parse failed: {type(exc).__name__}: {exc}")
            self._record_error("product_page_parse", url, exc)
            return
        _apply_updates(record, updates)
        self._flag_price_discrepancy(record)

    @staticmethod
    def _flag_price_discrepancy(record: dict[str, Any]) -> None:
        listing_price, page_price = record["listing_price_raw"], record["price_raw"]
        listing_value, page_value = _price_number(listing_price), _price_number(page_price)
        if listing_value is not None and page_value is not None and abs(listing_value - page_value) > 0.005:
            record["extraction_warnings"].append(
                f"listing card price {listing_price!r} differs from product page price {page_price!r}; "
                "product page price kept"
            )

    def _flag_missing_fields(self, record: dict[str, Any]) -> None:
        expected = self.expected_listing_fields
        if self.source.fetch_product_pages:
            expected += self.expected_product_page_fields
        record["extraction_warnings"].extend(
            f"missing expected field: {name}" for name in expected if record[name] is None
        )

    def _product_cap_reached(self) -> bool:
        return self.max_products is not None and len(self.records) >= self.max_products

    def _record_error(self, stage: str, url: str | None, error: BaseException) -> None:
        self.report.errors.append(error_entry(stage, self.source.source_id, url, error))
        detail = str(error) if isinstance(error, FetchError) else f"{type(error).__name__}: {error} ({url})"
        logger.warning("[%s] %s failed - %s", self.source.source_id, stage.replace("_", " "), detail)


ScraperT = TypeVar("ScraperT", bound=BaseScraper)
SCRAPER_REGISTRY: dict[str, type[BaseScraper]] = {}


def register_scraper(cls: type[ScraperT]) -> type[ScraperT]:
    """Class decorator that makes a scraper available to :func:`scrape_products`."""
    if cls.source_id in SCRAPER_REGISTRY:
        raise ValueError(f"A scraper for {cls.source_id!r} is already registered")
    SCRAPER_REGISTRY[cls.source_id] = cls
    return cls


# --------------------------------------------------------------------------- concrete scrapers
@register_scraper
class BooksToScrapeScraper(BaseScraper):
    """books.toscrape.com: listing cards + product pages (UPC, stock count, description)."""

    source_id = "books_toscrape"
    card_selector = "article.product_pod"
    expected_listing_fields = ("product_name", "price_raw", "rating_raw", "availability_raw")
    expected_product_page_fields = ("description", "review_count", "source_product_id")

    def parse_card(self, card: Tag, page_url: str) -> dict[str, Any]:
        link = card.select_one("h3 a[href]")
        price = _element_text(card.select_one("p.price_color"))
        return {
            "product_name": _attribute(link, "title") or _element_text(link),  # link text is truncated
            "product_url": _absolute_url(page_url, _attribute(link, "href")),
            "price_raw": price,
            "listing_price_raw": price,
            "rating_raw": _star_rating_token(card.select_one("p.star-rating")),
            "availability_raw": _element_text(card.select_one("p.availability")),
        }

    def find_next_page(self, soup: BeautifulSoup, page_url: str) -> str | None:
        return _absolute_url(page_url, _attribute(soup.select_one("li.next a[href]"), "href"))

    def parse_product_page(self, soup: BeautifulSoup, record: Mapping[str, Any]) -> dict[str, Any]:
        main = soup.select_one("div.product_main")
        if not isinstance(main, Tag):
            raise PageStructureError("div.product_main not found on product page")
        information = self._product_information(soup)
        crumbs = soup.select("ul.breadcrumb li a")  # Home > Books > <category>
        return {
            "product_name": _element_text(main.select_one("h1")),
            "price_raw": _element_text(main.select_one("p.price_color")),
            "availability_raw": _element_text(main.select_one("p.availability")) or information.get("Availability"),
            "rating_raw": _star_rating_token(main.select_one("p.star-rating")),
            "description": self._description(soup),
            "review_count": _int_or_raw(information.get("Number of reviews")),
            "source_product_id": information.get("UPC"),
            "category": _element_text(crumbs[2]) if len(crumbs) >= 3 else None,
        }

    @staticmethod
    def _description(soup: BeautifulSoup) -> str | None:
        """The ``<p>`` directly after the "Product Description" header; some books have none."""
        header = soup.select_one("div#product_description")
        paragraph = header.find_next_sibling() if isinstance(header, Tag) else None
        return _element_text(paragraph) if isinstance(paragraph, Tag) and paragraph.name == "p" else None

    @staticmethod
    def _product_information(soup: BeautifulSoup) -> dict[str, str]:
        """The "Product Information" table as ``{header: value}``."""
        information: dict[str, str] = {}
        for row in soup.select("table.table-striped tr"):
            key, value = _element_text(row.find("th")), _element_text(row.find("td"))
            if key and value is not None:
                information[key] = value
        return information


@register_scraper
class WebScraperIoScraper(BaseScraper):
    """webscraper.io static test shop: schema.org microdata on every listing card."""

    source_id = "webscraper_io"
    card_selector = "div.card.thumbnail"
    expected_listing_fields = ("product_name", "price_raw", "currency_raw", "rating_raw", "description", "review_count")

    def parse_card(self, card: Tag, page_url: str) -> dict[str, Any]:
        link = card.select_one("a.title[href]")
        product_url = _absolute_url(page_url, _attribute(link, "href"))
        return {
            **self._microdata_fields(card),
            "product_name": _attribute(link, "title") or _element_text(link),
            "product_url": product_url,
            "listing_price_raw": _element_text(card.select_one("[itemprop=price]")),
            "source_product_id": _product_id_from_url(product_url),
        }

    def find_next_page(self, soup: BeautifulSoup, page_url: str) -> str | None:
        return _absolute_url(page_url, _attribute(soup.select_one("ul.pagination a[rel~=next][href]"), "href"))

    def parse_product_page(self, soup: BeautifulSoup, record: Mapping[str, Any]) -> dict[str, Any]:
        product = soup.select_one('[itemtype$="schema.org/Product"]')
        if not isinstance(product, Tag):
            raise PageStructureError("schema.org Product microdata not found on product page")
        return {**self._microdata_fields(product), "product_name": _element_text(product.select_one("[itemprop=name]"))}

    @staticmethod
    def _microdata_fields(scope: Tag) -> dict[str, Any]:
        return {
            "price_raw": _element_text(scope.select_one("[itemprop=price]")),
            "currency_raw": _attribute(scope.select_one("meta[itemprop=priceCurrency]"), "content"),
            "description": _element_text(scope.select_one("[itemprop=description]")),
            "review_count": _int_or_raw(_element_text(scope.select_one("[itemprop=reviewCount]"))),
            "rating_raw": _microdata_rating(scope),
        }


@register_scraper
class WebScrapingDevScraper(BaseScraper):
    """web-scraping.dev: listing cards + product pages (JSON-LD offers, rating and sample reviews).

    The paid price is the product-page price (it carries the currency symbol and can differ from the
    listing card); the listing price is kept in ``listing_price_raw`` for traceability. Reviews come
    from the first page embedded in the product page; the CSRF-protected review API is never called.
    """

    source_id = "web_scraping_dev"
    card_selector = "div.products div.row.product"
    expected_listing_fields = ("product_name", "listing_price_raw")
    expected_product_page_fields = (
        "price_raw", "currency_raw", "rating_raw", "availability_raw", "description", "review_count", "review_text",
    )

    def parse_card(self, card: Tag, page_url: str) -> dict[str, Any]:
        link = card.select_one("h3 a[href]")
        product_url = _absolute_url(page_url, _attribute(link, "href"))
        price = _element_text(card.select_one("div.price"))
        return {
            "product_name": _element_text(link),
            "product_url": product_url,
            "price_raw": price,  # replaced by the product-page price when available
            "listing_price_raw": price,
            "description": _element_text(card.select_one("div.short-description")),
            "source_product_id": _product_id_from_url(product_url),
        }

    def find_next_page(self, soup: BeautifulSoup, page_url: str) -> str | None:
        # "page 3 of total 12 results in 3 pages": the ">" link keeps pointing past the last page.
        match = _WSD_PAGING_RE.search(_element_text(soup.select_one("div.paging-meta")) or "")
        if match and int(match.group(1)) >= int(match.group(2)):
            return None
        links = soup.select("div.paging a[href]")
        if not links or _element_text(links[-1]) not in {">", "›", "»"}:
            return None
        return _absolute_url(page_url, _attribute(links[-1], "href"))

    def parse_product_page(self, soup: BeautifulSoup, record: Mapping[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        product = _json_ld_product(soup)
        if product is None:
            warnings.append("schema.org JSON-LD Product data not found on product page")
            product = {}
        offer = _first_offer(product.get("offers"))
        aggregate = _mapping(product.get("aggregateRating"))

        price = _element_text(soup.select_one("span.product-price"))
        if price is None:
            price = _clean_text(offer.get("price") or offer.get("lowPrice"))
            if price is not None:
                warnings.append("displayed price not found; JSON-LD offer price used")

        excerpts, review_source = _json_ld_reviews(product), REVIEW_SOURCE_JSON_LD
        if not excerpts:
            excerpts, review_source = _embedded_reviews(soup), REVIEW_SOURCE_EMBEDDED
        review_text = format_review_excerpts(excerpts)

        return {
            "product_name": _clean_text(product.get("name")) or _element_text(soup.select_one("h3.product-title")),
            "price_raw": price,
            "original_price_raw": _element_text(soup.select_one("span.product-price-full")),
            "currency_raw": _clean_text(offer.get("priceCurrency")),
            "availability_raw": _clean_text(offer.get("availability")),
            "rating_raw": _aggregate_rating_token(aggregate),
            "review_count": _int_or_raw(aggregate.get("reviewCount")),
            "description": (
                _element_text(soup.select_one("p.product-description"))
                or record.get("description")
                or _clean_text(product.get("description"))
            ),
            "review_text": review_text,
            "review_source": review_source if review_text else None,
            "extraction_warnings": warnings,
        }


# --------------------------------------------------------------------------- public API
def scrape_products(
    sources: Iterable[config.SourceConfig] = config.SOURCES,
    client: PageFetcher | None = None,
    max_pages: int | None = None,
    max_products: int | None = None,
) -> ScrapeResult:
    """Scrape every configured source and return records plus an auditable report.

    ``max_pages`` (pages per listing) and ``max_products`` (records per source) override the
    per-source configuration when given. A failing source is logged and recorded in ``errors``;
    the remaining sources still run.
    """
    for name, limit in (("max_pages", max_pages), ("max_products", max_products)):
        if limit is not None and limit < 1:
            raise ValueError(f"{name} must be a positive integer, got {limit!r}")
    own_client = client is None
    fetcher: PageFetcher = PoliteHttpClient() if client is None else client
    started_at = utc_timestamp()
    records: list[dict[str, Any]] = []
    reports: list[SourceReport] = []
    try:
        for source in sources:
            report, source_records = _scrape_source(source, fetcher, max_pages, max_products)
            reports.append(report)
            records.extend(source_records)
        http_stats = fetcher.stats_as_dict()
    finally:
        if own_client and isinstance(fetcher, PoliteHttpClient):
            fetcher.close()
    errors = [error for report in reports for error in report.errors]
    logger.info(
        "Scraping finished: %d record(s) from %d source(s), %d error(s)", len(records), len(reports), len(errors)
    )
    return ScrapeResult(
        records=records,
        source_reports=reports,
        errors=errors,
        started_at=started_at,
        finished_at=utc_timestamp(),
        http_stats=http_stats,
    )


def _scrape_source(
    source: config.SourceConfig, fetcher: PageFetcher, max_pages: int | None, max_products: int | None
) -> tuple[SourceReport, list[dict[str, Any]]]:
    scraper_class = SCRAPER_REGISTRY.get(source.source_id)
    if scraper_class is None:
        report = SourceReport.for_source(source)
        error = LookupError(f"no scraper registered for source_id {source.source_id!r}")
        report.errors.append(error_entry("source", source.source_id, source.base_url, error))
        logger.error("[%s] %s", source.source_id, error)
        return report, []
    scraper = scraper_class(source, fetcher, max_pages=max_pages, max_products=max_products)
    try:
        scraper.scrape()
    except Exception as exc:  # last-resort isolation: a bug in one source must not lose the others
        logger.exception(
            "[%s] Source failed unexpectedly; keeping %d record(s) collected so far",
            source.source_id, len(scraper.records),
        )
        scraper.report.errors.append(error_entry("source", source.source_id, source.base_url, exc))
        scraper.report.summarise(scraper.records)
    return scraper.report, scraper.records


def save_raw_products(result: ScrapeResult, path: Path | str = config.RAW_PRODUCTS_FILE) -> Path:
    """Write ``{"generated_at", "scrape_report", "records"}`` as UTF-8 JSON (atomic replace)."""
    target = Path(path)
    report = result.as_dict()
    records = report.pop("records")
    payload = {"generated_at": utc_timestamp(), "scrape_report": report, "records": records}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    logger.info("Saved %d raw product record(s) to %s", len(records), target)
    return target


def load_raw_products(path: Path | str = config.RAW_PRODUCTS_FILE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a file written by :func:`save_raw_products`; returns ``(records, scrape_report)``.

    Raises FileNotFoundError when the file is missing and ValueError when it is corrupt.
    Records missing contract keys (older files) get them filled with None.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Raw products file not found: {source}. Run the scraping stage first.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Raw products file {source} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Raw products file {source} must be a JSON object with a 'records' list")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload["records"]):
        if not isinstance(item, dict):
            raise ValueError(f"Raw products file {source}: record #{index} is not a JSON object")
        record: dict[str, Any] = dict.fromkeys(RECORD_FIELDS)
        record.update(item)
        warnings = record["extraction_warnings"]
        if not isinstance(warnings, list):
            record["extraction_warnings"] = [] if warnings is None else [str(warnings)]
        records.append(record)
    report = payload.get("scrape_report")
    return records, report if isinstance(report, dict) else {}
