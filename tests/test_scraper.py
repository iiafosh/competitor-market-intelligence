"""Offline tests for the scraping stage: parsers, pagination, error isolation and persistence."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import pytest
import requests
from bs4 import BeautifulSoup

import config
import scraper
from http_client import PermanentHTTPError, PoliteHttpClient, RetryExhaustedError
from scraper import (
    RECORD_FIELDS,
    REVIEW_SOURCE_EMBEDDED,
    REVIEW_SOURCE_JSON_LD,
    STOP_EMPTY_LISTING,
    STOP_FETCH_FAILED,
    STOP_LAST_PAGE,
    STOP_PAGE_LIMIT,
    STOP_PRODUCT_CAP,
    STOP_REPEATED_URL,
    BooksToScrapeScraper,
    ReviewExcerpt,
    WebScraperIoScraper,
    WebScrapingDevScraper,
    format_review_excerpts,
    load_raw_products,
    save_raw_products,
    scrape_products,
)

FIXTURES = Path(__file__).parent / "fixtures"
BOOKS_LISTING = "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"
BOOKS_PAGE_2 = "https://books.toscrape.com/catalogue/category/books/mystery_3/page-2.html"
LIGHT_IN_THE_ATTIC = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
TIPPING_THE_VELVET = "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html"
SOUMISSION = "https://books.toscrape.com/catalogue/soumission_998/index.html"
WSIO_LISTING = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"
WSD_LISTING = "https://web-scraping.dev/products?category=consumables"
WSD_PRODUCT_1 = "https://web-scraping.dev/product/1"
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# --------------------------------------------------------------------------- helpers
def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def soup_of(name: str) -> BeautifulSoup:
    return BeautifulSoup(fixture(name), "html.parser")


def source_for(source_id: str, **overrides: Any) -> config.SourceConfig:
    base = next(source for source in config.SOURCES if source.source_id == source_id)
    return dataclasses.replace(base, **overrides)


class FakeFetcher:
    """Serves fixture bytes per URL; unknown URLs raise HTTP 404 like the real client."""

    def __init__(self, pages: Mapping[str, bytes | Exception]) -> None:
        self.pages = dict(pages)
        self.requested: list[str] = []
        self.min_delays: list[float | None] = []

    def get_soup(self, url: str, *, min_delay: float | None = None) -> BeautifulSoup:
        self.requested.append(url)
        self.min_delays.append(min_delay)
        page = self.pages.get(url)
        if page is None:
            raise PermanentHTTPError(url, "HTTP 404 Not Found", status_code=404)
        if isinstance(page, Exception):
            raise page
        return BeautifulSoup(page, "html.parser")

    def stats_as_dict(self) -> dict[str, Any]:
        return {"requests_sent": len(self.requested)}


def books_page(titles: list[str], next_href: str | None) -> bytes:
    """Minimal books.toscrape.com category page."""
    cards = "".join(
        f'<article class="product_pod"><p class="star-rating Two"></p>'
        f'<h3><a href="../../../{title.lower()}_1/index.html" title="{title}">{title}</a></h3>'
        f'<p class="price_color">£10.00</p><p class="instock availability">In stock</p></article>'
        for title in titles
    )
    pager = f'<ul class="pager"><li class="next"><a href="{next_href}">next</a></li></ul>' if next_href else ""
    return f'<html><head><meta charset="utf-8"></head><body><ol>{cards}</ol>{pager}</body></html>'.encode()


def run_books(
    pages: Mapping[str, bytes | Exception],
    *,
    max_pages: int | None = None,
    max_products: int | None = None,
    fetch_product_pages: bool = False,
) -> BooksToScrapeScraper:
    source = source_for("books_toscrape", fetch_product_pages=fetch_product_pages)
    books = BooksToScrapeScraper(source, FakeFetcher(pages), max_pages=max_pages, max_products=max_products)
    books.scrape()
    return books


def two_book_pages() -> dict[str, bytes]:
    return {BOOKS_LISTING: fixture("books_listing_page1.html"), BOOKS_PAGE_2: fixture("books_listing_page2.html")}


# --------------------------------------------------------------------------- books.toscrape.com parsers
def test_books_listing_card_fields() -> None:
    books = BooksToScrapeScraper(source_for("books_toscrape"), FakeFetcher({}))
    soup = soup_of("books_listing_page1.html")

    cards = books.find_cards(soup)
    item = books.parse_card(cards[0], BOOKS_LISTING)

    assert len(cards) == 3
    assert item == {
        "product_name": "A Light in the Attic",  # title attribute, not the truncated link text
        "product_url": LIGHT_IN_THE_ATTIC,
        "price_raw": "£51.77",
        "listing_price_raw": "£51.77",
        "rating_raw": "Three",
        "availability_raw": "In stock",
    }
    assert books.find_next_page(soup, BOOKS_LISTING) == BOOKS_PAGE_2
    assert books.find_next_page(soup_of("books_listing_page2.html"), BOOKS_PAGE_2) is None


def test_books_product_page_fields() -> None:
    books = BooksToScrapeScraper(source_for("books_toscrape"), FakeFetcher({}))

    updates = books.parse_product_page(soup_of("books_product.html"), {})

    assert updates["product_name"] == "A Light in the Attic"
    assert updates["price_raw"] == "£51.77"
    assert updates["availability_raw"] == "In stock (22 available)"
    assert updates["rating_raw"] == "Three"
    assert updates["description"].startswith("It's hard to imagine a world without A Light in the Attic.")
    assert updates["review_count"] == 0
    assert updates["source_product_id"] == "a897fe39b1053632"
    assert updates["category"] == "Poetry"


def test_books_product_without_description_or_breadcrumb() -> None:
    books = BooksToScrapeScraper(source_for("books_toscrape"), FakeFetcher({}))
    soup = soup_of("books_product.html")
    header = soup.select_one("div#product_description")
    header.find_next_sibling("p").decompose()
    header.decompose()
    soup.select_one("ul.breadcrumb").decompose()

    updates = books.parse_product_page(soup, {})

    assert updates["description"] is None
    assert updates["category"] is None  # the listing category is kept by the merge
    assert updates["price_raw"] == "£51.77"


def test_books_card_with_missing_elements_does_not_crash() -> None:
    books = BooksToScrapeScraper(source_for("books_toscrape"), FakeFetcher({}))
    markup = '<article class="product_pod"><h3><a href="x_1/index.html">X</a></h3></article>'
    card = BeautifulSoup(markup, "html.parser").article

    item = books.parse_card(card, BOOKS_LISTING)

    assert item["product_name"] == "X"
    assert item["product_url"] == "https://books.toscrape.com/catalogue/category/books/mystery_3/x_1/index.html"
    assert item["price_raw"] is None and item["rating_raw"] is None and item["availability_raw"] is None


# --------------------------------------------------------------------------- webscraper.io parsers
def test_webscraper_listing_card_fields() -> None:
    shop = WebScraperIoScraper(source_for("webscraper_io"), FakeFetcher({}))
    soup = soup_of("webscraper_listing.html")

    cards = shop.find_cards(soup)
    item = shop.parse_card(cards[0], WSIO_LISTING)

    assert len(cards) == 2
    assert item == {
        "product_name": "Packard 255 G2",
        "product_url": "https://webscraper.io/test-sites/e-commerce/static/product/31",
        "price_raw": "$416.99",
        "listing_price_raw": "$416.99",
        "currency_raw": "USD",
        "description": '15.6", AMD E2-3800 1.3GHz, 4GB, 500GB, Windows 8.1',
        "review_count": 2,
        "rating_raw": "2",
        "source_product_id": "31",
    }
    assert shop.parse_card(cards[1], WSIO_LISTING)["rating_raw"] == "3"
    assert shop.find_next_page(soup, WSIO_LISTING) == f"{WSIO_LISTING}?page=2"


def test_webscraper_last_page_has_no_next_link() -> None:
    shop = WebScraperIoScraper(source_for("webscraper_io"), FakeFetcher({}))
    soup = soup_of("webscraper_listing.html")
    soup.select_one("a[rel~=next]").decompose()

    assert shop.find_next_page(soup, WSIO_LISTING) is None


def test_webscraper_product_page_microdata() -> None:
    shop = WebScraperIoScraper(source_for("webscraper_io"), FakeFetcher({}))

    updates = shop.parse_product_page(soup_of("webscraper_product.html"), {})

    assert updates["product_name"] == "Packard 255 G2"  # not the sidebar's itemprop=name entries
    assert updates["price_raw"] == "$416.99"
    assert updates["currency_raw"] == "USD"
    assert updates["review_count"] == 2
    assert updates["rating_raw"] == "2"  # counted stars: product pages have no data-rating


def test_webscraper_card_with_missing_elements_does_not_crash() -> None:
    shop = WebScraperIoScraper(source_for("webscraper_io"), FakeFetcher({}))
    card = BeautifulSoup('<div class="card thumbnail"><a class="title" href="/p/9">Laptop</a></div>', "html.parser").div

    item = shop.parse_card(card, WSIO_LISTING)

    assert item["product_name"] == "Laptop"
    assert item["price_raw"] is None and item["currency_raw"] is None and item["rating_raw"] is None
    assert item["review_count"] is None and item["source_product_id"] is None


# --------------------------------------------------------------------------- web-scraping.dev parsers
def test_wsd_listing_card_fields_and_next_page() -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))
    soup = soup_of("wsd_listing.html")

    cards = store.find_cards(soup)
    item = store.parse_card(cards[0], WSD_LISTING)

    assert len(cards) == 2
    assert item["product_name"] == "Box of Chocolate Candy"
    assert item["product_url"] == WSD_PRODUCT_1
    assert item["listing_price_raw"] == item["price_raw"] == "24.99"
    assert item["description"].startswith("Indulge your sweet tooth")
    assert item["source_product_id"] == "1"
    assert store.find_next_page(soup, WSD_LISTING) == f"{WSD_LISTING}&page=2"


@pytest.mark.parametrize(
    ("paging", "expected"),
    [
        # last page: the ">" link still points one page further (to an empty page)
        ('<div class="paging-meta">page 3 of total 12 results in 3 pages</div>'
         '<div class="paging"><a href="/products?page=2">&lt;</a><a href="/products?page=4">&gt;</a></div>', None),
        # beyond the end the ">" link has no href
        ('<div class="paging"><a href="/products?page=4">&lt;</a><a>&gt;</a></div>', None),
        ('<div class="paging"><a href="/products?page=1">&lt;</a><a href="/products?page=2">&gt;</a></div>',
         "https://web-scraping.dev/products?page=2"),
    ],
)
def test_wsd_next_page_detection(paging: str, expected: str | None) -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))
    soup = BeautifulSoup(f"<html><body>{paging}</body></html>", "html.parser")

    assert store.find_next_page(soup, "https://web-scraping.dev/products?page=1") == expected


def test_wsd_product_page_fields() -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))

    updates = store.parse_product_page(soup_of("wsd_product.html"), {"description": "listing text"})

    assert updates["product_name"] == "Box of Chocolate Candy"
    assert updates["price_raw"] == "$9.99"
    assert updates["original_price_raw"] == "$12.99"
    assert updates["currency_raw"] == "USD"
    assert updates["availability_raw"] == "https://schema.org/InStock"
    assert updates["rating_raw"] == "4.7"
    assert updates["review_count"] == 10
    assert updates["description"].startswith("Indulge your sweet tooth with our Box of Chocolate Candy. Each box")
    assert updates["review_text"] == (
        "[5/5] Absolutely delicious! The orange flavor is my favorite."
        " || [4/5] I bought these as a gift, and they were well received. Will definitely purchase again."
    )
    assert updates["review_source"] == REVIEW_SOURCE_JSON_LD
    assert updates["extraction_warnings"] == []


def test_wsd_product_page_fallbacks_without_json_ld() -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))
    soup = soup_of("wsd_product.html")
    soup.select_one('script[type="application/ld+json"]').decompose()
    soup.select_one("p.product-description").decompose()

    updates = store.parse_product_page(soup, {"description": "Short listing description."})

    assert updates["product_name"] == "Box of Chocolate Candy"  # from the page heading
    assert updates["description"] == "Short listing description."
    assert updates["currency_raw"] is None and updates["rating_raw"] is None
    assert updates["review_text"].startswith("[5/5] Absolutely delicious!")  # embedded reviews-data JSON
    assert updates["review_source"] == REVIEW_SOURCE_EMBEDDED
    assert any("JSON-LD" in warning for warning in updates["extraction_warnings"])


def test_wsd_description_falls_back_to_json_ld_last() -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))
    soup = soup_of("wsd_product.html")
    soup.select_one("p.product-description").decompose()

    updates = store.parse_product_page(soup, {"description": None})

    assert updates["description"].endswith("(JSON-LD description).")


def test_wsd_empty_product_page_does_not_crash() -> None:
    store = WebScrapingDevScraper(source_for("web_scraping_dev"), FakeFetcher({}))

    updates = store.parse_product_page(BeautifulSoup("<html><body></body></html>", "html.parser"), {})

    for key in ("price_raw", "currency_raw", "rating_raw", "review_text", "description"):
        assert updates[key] is None, key


def test_format_review_excerpts() -> None:
    excerpts = [
        ReviewExcerpt("Great || value", "5"),
        ReviewExcerpt("  Meh\n ok ", None),
        ReviewExcerpt("  ", "3"),
        ReviewExcerpt("Fine", "8", "10"),
    ]

    assert format_review_excerpts(excerpts) == "[5/5] Great | value || Meh ok || [8/10] Fine"
    assert format_review_excerpts([]) is None


# --------------------------------------------------------------------------- pagination
def test_pagination_stops_at_natural_end() -> None:
    books = run_books(two_book_pages())

    listing = books.report.listings[0]
    assert listing.pages_visited == [BOOKS_LISTING, BOOKS_PAGE_2]
    assert listing.stop_reason == STOP_LAST_PAGE
    names = [record["product_name"] for record in books.records]
    assert names == ["A Light in the Attic", "Tipping the Velvet", "Soumission", "Sharp Objects"]
    assert books.records[3]["listing_page_url"] == BOOKS_PAGE_2


def test_pagination_respects_page_limit() -> None:
    books = run_books(two_book_pages(), max_pages=1)

    assert books.client.requested == [BOOKS_LISTING]
    assert books.report.listings[0].stop_reason == STOP_PAGE_LIMIT
    assert len(books.records) == 3


def test_pagination_stops_on_repeated_url() -> None:
    pages = {BOOKS_LISTING: books_page(["Alpha"], "page-2.html"), BOOKS_PAGE_2: books_page(["Beta"], "index.html")}

    books = run_books(pages)

    assert books.client.requested == [BOOKS_LISTING, BOOKS_PAGE_2]
    assert books.report.listings[0].stop_reason == STOP_REPEATED_URL
    assert len(books.records) == 2


def test_pagination_stops_on_empty_listing() -> None:
    books = run_books({BOOKS_LISTING: books_page([], "page-2.html")})

    assert books.records == []
    assert books.report.listings[0].stop_reason == STOP_EMPTY_LISTING
    assert books.client.requested == [BOOKS_LISTING]


def test_failed_page_stops_listing_but_keeps_collected_records() -> None:
    books = run_books({BOOKS_LISTING: fixture("books_listing_page1.html")})  # page 2 -> HTTP 404

    assert books.report.listings[0].stop_reason == STOP_FETCH_FAILED
    assert len(books.records) == 3
    assert books.report.errors[0]["stage"] == "listing_page"
    assert books.report.errors[0]["error_type"] == "PermanentHTTPError"


def test_product_cap_stops_the_source() -> None:
    books = run_books({BOOKS_LISTING: fixture("books_listing_page1.html")}, max_products=2)

    assert len(books.records) == 2
    assert books.report.listings[0].stop_reason == STOP_PRODUCT_CAP


def test_card_without_link_is_skipped_and_reported() -> None:
    page = books_page(["Alpha"], None).replace(b"<ol>", b'<ol><article class="product_pod"><h3>No link</h3></article>')

    books = run_books({BOOKS_LISTING: page})

    assert [record["product_name"] for record in books.records] == ["Alpha"]
    assert books.report.errors[0]["stage"] == "listing_card"


# --------------------------------------------------------------------------- product-page enrichment
def test_product_page_failures_keep_listing_data_with_warning() -> None:
    pages = {
        BOOKS_LISTING: fixture("books_listing_page1.html"),
        BOOKS_PAGE_2: books_page([], None),
        LIGHT_IN_THE_ATTIC: fixture("books_product.html"),
        SOUMISSION: RetryExhaustedError(SOUMISSION, "ReadTimeout", attempts=4),
        # TIPPING_THE_VELVET is missing -> HTTP 404
    }
    books = run_books(pages, fetch_product_pages=True)
    enriched, missing, timed_out = books.records

    assert enriched["availability_raw"] == "In stock (22 available)"
    assert enriched["source_product_id"] == "a897fe39b1053632"
    assert enriched["category"] == "Poetry"
    assert enriched["extraction_warnings"] == []
    assert missing["price_raw"] == "£53.74" and missing["rating_raw"] == "One"
    assert missing["category"] == "Mystery"
    assert "product page fetch failed: HTTP 404 Not Found" in missing["extraction_warnings"]
    assert "missing expected field: description" in missing["extraction_warnings"]
    assert "product page fetch failed: ReadTimeout (after 4 attempts)" in timed_out["extraction_warnings"]
    assert [error["stage"] for error in books.report.errors].count("product_page") == 2
    assert books.report.product_pages_fetched == 1


def test_unexpected_product_page_structure_keeps_listing_data() -> None:
    pages = {
        BOOKS_LISTING: books_page(["Alpha"], None),
        "https://books.toscrape.com/catalogue/alpha_1/index.html": b"<html><body>Maintenance</body></html>",
    }

    books = run_books(pages, fetch_product_pages=True)

    record = books.records[0]
    assert record["price_raw"] == "£10.00"
    assert record["extraction_warnings"][0].startswith("product page parse failed: PageStructureError")
    assert books.report.errors[0]["stage"] == "product_page_parse"


def test_wsd_product_page_price_wins_and_discrepancy_is_recorded() -> None:
    pages = {WSD_LISTING: fixture("wsd_listing.html"), WSD_PRODUCT_1: fixture("wsd_product.html")}
    source = source_for("web_scraping_dev", listing_starts=(config.ListingStart(WSD_LISTING, "Consumables"),))
    fetcher = FakeFetcher(pages)

    store = WebScrapingDevScraper(source, fetcher, max_pages=1)
    store.scrape()
    record = store.records[0]

    assert record["price_raw"] == "$9.99"
    assert record["listing_price_raw"] == "24.99"
    assert record["category"] == "Consumables"
    assert any("differs from product page price" in warning for warning in record["extraction_warnings"])
    assert set(fetcher.min_delays) == {2.0}  # the source's politeness floor is passed on every request
    assert "api/reviews" not in " ".join(fetcher.requested)


# --------------------------------------------------------------------------- run-level behaviour
def test_records_follow_the_contract() -> None:
    pages = {
        BOOKS_LISTING: fixture("books_listing_page1.html"),
        BOOKS_PAGE_2: fixture("books_listing_page2.html"),
        LIGHT_IN_THE_ATTIC: fixture("books_product.html"),
        WSIO_LISTING: fixture("webscraper_listing.html"),
        WSD_LISTING: fixture("wsd_listing.html"),
        WSD_PRODUCT_1: fixture("wsd_product.html"),
    }
    sources = [
        source_for("books_toscrape"),
        source_for("webscraper_io"),
        source_for("web_scraping_dev", listing_starts=(config.ListingStart(WSD_LISTING, "Consumables"),)),
    ]

    result = scrape_products(sources, client=FakeFetcher(pages), max_pages=2)

    assert {record["source"] for record in result.records} == {"books_toscrape", "webscraper_io", "web_scraping_dev"}
    for record in result.records:
        assert tuple(record) == RECORD_FIELDS
        assert TIMESTAMP_RE.match(record["scraped_at"])
        assert record["product_url"].startswith("https://")
        assert isinstance(record["extraction_warnings"], list)
        assert (record["review_source"] is None) == (record["review_text"] is None)
    json.dumps(result.as_dict())  # JSON-serialisable end to end
    assert result.as_dict()["records_per_source"] == {"books_toscrape": 4, "webscraper_io": 2, "web_scraping_dev": 2}


def test_one_failing_source_does_not_stop_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingScraper(WebScraperIoScraper):
        def scrape(self) -> list[dict[str, Any]]:
            raise RuntimeError("unexpected bug")

    monkeypatch.setitem(scraper.SCRAPER_REGISTRY, "webscraper_io", ExplodingScraper)
    unknown = dataclasses.replace(source_for("webscraper_io"), source_id="unknown_shop")
    sources = [source_for("books_toscrape", fetch_product_pages=False), source_for("webscraper_io"), unknown]

    result = scrape_products(sources, client=FakeFetcher(two_book_pages()))

    assert len(result.records) == 4
    assert [(error["source"], error["stage"], error["error_type"]) for error in result.errors] == [
        ("webscraper_io", "source", "RuntimeError"),
        ("unknown_shop", "source", "LookupError"),
    ]
    assert [report.source_id for report in result.source_reports] == ["books_toscrape", "webscraper_io", "unknown_shop"]


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        scrape_products([], client=FakeFetcher({}), max_pages=0)


def test_end_to_end_with_real_client_decodes_pound_sign() -> None:
    """Real PoliteHttpClient + fake session: robots 404, header without charset, polite delays."""

    class Session:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str, headers: Any = None, timeout: Any = None) -> requests.Response:
            self.urls.append(url)
            response = requests.Response()
            response.url = url
            if url == BOOKS_LISTING:
                response.status_code, response._content = 200, fixture("books_listing_page1.html")
                response.headers["Content-Type"] = "text/html"  # books.toscrape.com sends no charset
            else:
                response.status_code, response._content, response.reason = 404, b"not found", "Not Found"
            return response

    sleeps: list[float] = []
    session = Session()
    client = PoliteHttpClient(session, sleep=sleeps.append, clock=lambda: 0.0)

    result = scrape_products([source_for("books_toscrape", fetch_product_pages=False)], client=client, max_pages=1)

    assert [record["price_raw"] for record in result.records] == ["£51.77", "£53.74", "£50.10"]
    assert session.urls == ["https://books.toscrape.com/robots.txt", BOOKS_LISTING]
    assert sleeps == [1.5]  # politeness gap between robots.txt and the listing page
    assert result.http_stats["robots_txt"]["https://books.toscrape.com"]["status"] == "unavailable"


# --------------------------------------------------------------------------- persistence
def test_save_and_load_round_trip(tmp_path: Path) -> None:
    sources = [source_for("books_toscrape", fetch_product_pages=False)]
    result = scrape_products(sources, client=FakeFetcher(two_book_pages()))
    target = tmp_path / "nested" / "raw_products.json"

    save_raw_products(result, target)
    records, report = load_raw_products(target)

    raw = target.read_text(encoding="utf-8")
    assert "£51.77" in raw  # ensure_ascii=False keeps non-ASCII characters readable
    payload = json.loads(raw)
    assert set(payload) == {"generated_at", "scrape_report", "records"}
    assert "records" not in payload["scrape_report"]
    assert records == result.records
    assert report["total_records"] == 4
    assert report["source_reports"][0]["listings"][0]["stop_reason"] == STOP_LAST_PAGE


def test_load_raw_products_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_products(tmp_path / "missing.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_products(corrupt)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"records": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_products(wrong_shape)


def test_load_fills_missing_contract_keys(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    old_record = {"product_name": "X", "source": "s", "product_url": "https://x.test/1"}
    legacy.write_text(json.dumps({"records": [old_record]}), encoding="utf-8")

    records, report = load_raw_products(legacy)

    assert set(RECORD_FIELDS) <= set(records[0])
    assert records[0]["extraction_warnings"] == []
    assert report == {}
