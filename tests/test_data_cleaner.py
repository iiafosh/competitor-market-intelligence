"""Tests for the Pandas cleaning and validation layer (assignment sections 7 and 13.2)."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import pandas as pd
import pytest

import config
import data_cleaner
from data_cleaner import (
    canonicalize_url,
    clean_description,
    clean_product_data,
    clean_review_text,
    clean_text,
    dataframe_to_records,
    is_null_like,
    load_raw_records,
    make_product_id,
    normalize_availability,
    normalize_currency_code,
    normalize_rating,
    parse_count,
    parse_price,
    parse_rating,
    parse_timestamp,
    repair_mojibake,
    save_cleaned_csv,
    validate_clean_dataset,
)

# Contract Stage 2, written out independently of the implementation.
EXPECTED_SCHEMA = {
    "product_id": "string",
    "product_name": "string",
    "price": "float64",
    "currency": "string",
    "rating": "Int64",
    "availability": "string",
    "description": "string",
    "source": "string",
    "product_url": "string",
    "scraped_at": "datetime64[ns, UTC]",
    "review_text": "string",
    "category": "string",
    "review_count": "Int64",
    "review_source": "string",
    "stock_quantity": "Int64",
    "original_price": "float64",
    "rating_unrounded": "float64",
    "price_raw": "string",
    "rating_raw": "string",
    "availability_raw": "string",
    "data_gaps": "string",
}
RAW_KEYS = """
    product_name price_raw currency_raw original_price_raw listing_price_raw rating_raw availability_raw
    description source product_url scraped_at review_text review_count review_source category
    source_product_id listing_page_url extraction_warnings
""".split()

BOOK_URL = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
LAPTOP_URL = "https://webscraper.io/test-sites/e-commerce/static/product/31"
CANDY_URL = "https://web-scraping.dev/product/1"
CANDY_REVIEWS = (
    "[5/5] Absolutely delicious! The orange flavor is my favorite. || [4/5] I bought these as a gift, "
    "and they were well received. Will definitely purchase again."
)


def make_record(**overrides: object) -> dict:
    """A Stage 1 raw record: every contract key present, None unless overridden."""
    record: dict = {key: None for key in RAW_KEYS}
    record["extraction_warnings"] = []
    record.update(overrides)
    return record


def book_record(**overrides: object) -> dict:
    return make_record(
        **{
            "product_name": "A Light in the Attic",
            "price_raw": "£51.77",
            "rating_raw": "Three",
            "availability_raw": "In stock (22 available)",
            "description": (
                "It's hard to imagine a world without A Light in the Attic. This now-classic collection of "
                "poetry and drawings from Shel Silverstein celebrates its 20th anniversary. ...more"
            ),
            "source": "books_toscrape",
            "product_url": BOOK_URL,
            "scraped_at": "2026-10-17T12:00:00Z",
            "review_count": "0",
            "category": "Mystery",
            "listing_price_raw": "£51.77",
            "source_product_id": "a897fe39b1053632",
            "listing_page_url": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
            **overrides,
        }
    )


def laptop_record(**overrides: object) -> dict:
    return make_record(
        **{
            "product_name": "Packard 255 G2",
            "price_raw": "$416.99",
            "currency_raw": "USD",
            "rating_raw": "2",
            "description": '15.6", AMD E2-3800 1.3GHz, 4GB, 500GB, Windows 8.1',
            "source": "webscraper_io",
            "product_url": LAPTOP_URL,
            "scraped_at": "2026-10-17T12:01:00Z",
            "review_count": "2",
            "category": "Laptops",
            "listing_price_raw": "$416.99",
            "source_product_id": "31",
            "listing_page_url": "https://webscraper.io/test-sites/e-commerce/static/computers/laptops",
            **overrides,
        }
    )


def candy_record(**overrides: object) -> dict:
    return make_record(
        **{
            "product_name": "Box of Chocolate Candy",
            "price_raw": "$9.99",
            "currency_raw": "USD",
            "original_price_raw": "$12.99",
            "listing_price_raw": "24.99",
            "rating_raw": "4.7",
            "availability_raw": "https://schema.org/InStock",
            "description": "Indulge your sweet tooth with our Box of Chocolate Candy.",
            "source": "web_scraping_dev",
            "product_url": CANDY_URL,
            "scraped_at": "2026-10-17T12:02:00Z",
            "review_text": CANDY_REVIEWS,
            "review_count": 10,
            "review_source": "web-scraping.dev product page (schema.org JSON-LD sample reviews, first page only)",
            "category": "Consumables",
            "source_product_id": "1",
            "listing_page_url": "https://web-scraping.dev/products?category=consumables",
            **overrides,
        }
    )


def clean(records: list) -> data_cleaner.CleaningResult:
    return clean_product_data(records)


def row_for(df: pd.DataFrame, url: str) -> pd.Series:
    matches = df.loc[df["product_url"] == url]
    assert len(matches) == 1, f"expected exactly one row for {url}"
    return matches.iloc[0]


# ------------------------------------------------------------------ assignment tables
@pytest.mark.parametrize(
    ("raw", "expected_price", "expected_currency"),
    [
        ("$1,299.99", 1299.99, "USD"),
        ("€89.50", 89.5, "EUR"),
        ("£120", 120.0, "GBP"),
        ("USD 249.00", 249.0, "USD"),
    ],
)
def test_parse_price_assignment_table(raw: str, expected_price: float, expected_currency: str) -> None:
    price, currency = parse_price(raw)
    assert isinstance(price, float)
    assert price == expected_price
    assert currency == expected_currency


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4.5 out of 5", 5), ("Rating: 4 stars", 4), ("★★★★★", 5), ("3/5", 3)],
)
def test_rating_assignment_table(raw: str, expected: int) -> None:
    assert normalize_rating(parse_rating(raw)) == expected
    assert normalize_rating(raw) == expected


# ------------------------------------------------------------------------ price parsing
@pytest.mark.parametrize(
    ("raw", "expected_price", "expected_currency"),
    [
        ("1.299,99 €", 1299.99, "EUR"),
        ("1 299,99 €", 1299.99, "EUR"),
        ("1 299,99 EUR", 1299.99, "EUR"),  # non-breaking space
        ("1 299,99 €", 1299.99, "EUR"),  # narrow non-breaking space (French locale)
        ("CHF 1'299.00", 1299.0, "CHF"),
        ("12,50 €", 12.5, "EUR"),  # decimal comma
        ("$1,299", 1299.0, "USD"),  # thousands comma
        ("1.299 KWD", 1.299, "KWD"),  # three-decimal currency
        ("KWD 12.500", 12.5, "KWD"),
        ("12.500 EUR", 12500.0, "EUR"),  # "." + exactly three digits = thousands otherwise
        ("249.00 EUR", 249.0, "EUR"),  # code after the amount
        ("EGP 1,299.50", 1299.5, "EGP"),
        ("E£1,250", 1250.0, "EGP"),
        ("CA$19.99", 19.99, "CAD"),
        ("C$19.99", 19.99, "CAD"),
        ("A$5", 5.0, "AUD"),
        ("AU$5", 5.0, "AUD"),
        ("US$5", 5.0, "USD"),
        ("¥1200", 1200.0, "JPY"),
        ("₹1,29,999.00", 129999.0, "INR"),  # Indian lakh grouping
        ("SAR 99", 99.0, "SAR"),
        ("0.999", 0.999, None),  # a leading zero group is always decimal
        ("24.99", 24.99, None),  # no currency evidence at all
        ("2 for $10", 10.0, "USD"),  # amount with a currency wins over a bare quantity
        ("١٢٣٫٥٠ ج.م", 123.5, "EGP"),  # Arabic-Indic digits and Egyptian pound abbreviation
    ],
)
def test_parse_price_international_formats(raw: str, expected_price: float, expected_currency: str | None) -> None:
    assert parse_price(raw) == (expected_price, expected_currency)


@pytest.mark.parametrize(("raw", "expected"), [("Â£51.77", (51.77, "GBP")), ("â‚¬12,50", (12.5, "EUR"))])
def test_parse_price_repairs_mojibake(raw: str, expected: tuple) -> None:
    assert parse_price(raw) == expected


def test_parse_price_range_returns_lowest_bound() -> None:
    assert parse_price("$34.00 – $45.00") == (34.0, "USD")
    assert parse_price("34.00 - 45.00 USD") == (34.0, "USD")
    assert parse_price("$45 to $34") == (34.0, "USD")


@pytest.mark.parametrize("raw", ["-$5.00", "$-5", "USD -12.00", -3, -0.5])
def test_parse_price_negative_is_missing(raw: object) -> None:
    assert parse_price(raw)[0] is None


@pytest.mark.parametrize("raw", ["free", "FREE", "Free!"])
def test_parse_price_free_is_zero(raw: str) -> None:
    assert parse_price(raw) == (0.0, None)


@pytest.mark.parametrize(
    "raw",
    [None, float("nan"), pd.NA, "", "   ", "N/A", "n/a", "-", "Call for price", "Free shipping", "abc", "20% off"]
    + ["1,299,99"],  # the same separator cannot both group thousands and mark decimals
)
def test_parse_price_unparseable_is_none_never_zero(raw: object) -> None:
    price, _ = parse_price(raw)
    assert price is None


def test_currency_hint_wins_over_symbol_when_valid() -> None:
    assert parse_price("$12.99", "CAD") == (12.99, "CAD")
    assert parse_price("£5", "usd") == (5.0, "USD")
    assert parse_price("£5", "dollars") == (5.0, "GBP")  # invalid hint ignored
    assert parse_price(9.99, "USD") == (9.99, "USD")  # numeric JSON-LD price
    assert parse_price(None, "USD") == (None, "USD")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("usd", "USD"),
        (" EUR ", "EUR"),
        ("£", "GBP"),
        ("$", "USD"),
        ("RMB", "CNY"),
        ("XXX", None),  # ISO-4217 "no currency"
        ("Dollar", None),
        (None, None),
    ],
)
def test_normalize_currency_code(value: object, expected: str | None) -> None:
    assert normalize_currency_code(value) == expected


# ----------------------------------------------------------------------- rating parsing
@pytest.mark.parametrize(
    ("raw", "unrounded", "rounded"),
    [
        ("Three", 3.0, 3),
        ("star-rating Three", 3.0, 3),
        ("five", 5.0, 5),
        ("★★★★☆", 4.0, 4),
        ("8/10", 4.0, 4),
        ("4,5/5", 4.5, 5),
        ("92%", 4.6, 5),
        ("4.7", 4.7, 5),
        ("2", 2.0, 2),
        ("3.49", 3.49, 3),
        ("2.5", 2.5, 3),
        ("Rated 4.50 out of 5", 4.5, 5),
        ("4½ stars", 4.5, 5),
        (4.5, 4.5, 5),
    ],
)
def test_parse_and_normalize_rating_formats(raw: object, unrounded: float, rounded: int) -> None:
    assert parse_rating(raw) == pytest.approx(unrounded)
    assert normalize_rating(parse_rating(raw)) == rounded


@pytest.mark.parametrize(
    "raw", [None, "", "n/a", "not rated", "6", "11/10", "120%", "☆☆☆☆☆", "0", "Zero", True]
)
def test_invalid_or_missing_ratings_stay_missing(raw: object) -> None:
    assert normalize_rating(parse_rating(raw)) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4.5, 5),
        (3.49, 3),
        (3.5, 4),
        (2.5, 3),  # half-up, not banker's rounding
        (0.5, 1),
        (1, 1),
        (5, 5),
        (0, None),  # 0 stars = not rated
        (5.01, None),
        (-1, None),
        (float("nan"), None),
        (None, None),
    ],
)
def test_normalize_rating_round_half_up(value: object, expected: int | None) -> None:
    assert normalize_rating(value) == expected


# ------------------------------------------------------------------------- availability
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("In stock (22 available)", ("in_stock", 22)),
        ("https://schema.org/InStock", ("in_stock", None)),
        ("http://schema.org/OutOfStock", ("out_of_stock", None)),
        ("SoldOut", ("out_of_stock", None)),
        ("https://schema.org/Discontinued", ("out_of_stock", None)),
        ("https://schema.org/BackOrder", ("out_of_stock", None)),
        ("https://schema.org/PreOrder", ("unknown", None)),
        ("PreSale", ("unknown", None)),
        ("InStoreOnly", ("unknown", None)),
        ("LimitedAvailability", ("in_stock", None)),
        ("OnlineOnly", ("in_stock", None)),
        ("Out of stock", ("out_of_stock", None)),
        ("Not in stock", ("out_of_stock", None)),
        ("Currently unavailable", ("out_of_stock", None)),
        ("Sold out", ("out_of_stock", None)),
        ("Only 3 left in stock", ("in_stock", 3)),
        ("In stock (0 available)", ("out_of_stock", 0)),
        ("Available", ("in_stock", None)),
        ("Pre-order now", ("unknown", None)),
        ("garbage", ("unknown", None)),
        (None, ("unknown", None)),
        ("", ("unknown", None)),
    ],
)
def test_normalize_availability(raw: object, expected: tuple) -> None:
    status, quantity = normalize_availability(raw)
    assert (status, quantity) == expected
    assert status in config.AVAILABILITY_VALUES


# ------------------------------------------------------------------- text / URL helpers
@pytest.mark.parametrize(
    "value", [None, float("nan"), pd.NA, pd.NaT, "", "  ", "N/A", "none", "NULL", "nan", "-", "--"]
)
def test_is_null_like_detects_null_markers(value: object) -> None:
    assert is_null_like(value)


@pytest.mark.parametrize("value", ["0", 0, 0.0, "Nana", "A", [], False])
def test_is_null_like_keeps_real_values(value: object) -> None:
    assert not is_null_like(value)


def test_clean_text_normalises_whitespace_entities_and_mojibake() -> None:
    assert clean_text("  A Light \n in   the​  Attic ") == "A Light in the Attic"
    assert clean_text("Tom &amp; Jerry &#39;s") == "Tom & Jerry 's"
    assert clean_text("Itâ€™s great") == "It’s great"
    assert clean_text("n/a") is None
    assert clean_description("Great read. ...more") == "Great read."
    assert clean_description("Only marker ...more") == "Only marker"
    assert clean_text("Great read. ...more") == "Great read. ...more"  # only descriptions lose the marker
    assert repair_mojibake("Â£51.77") == "£51.77"
    assert repair_mojibake("pâte") == "pâte"  # genuine accents are untouched


def test_clean_review_text_keeps_separator_and_drops_empty_excerpts() -> None:
    raw = "[5/5]  Great   taste ||  || [4/5]"
    assert clean_review_text(raw) == "[5/5] Great taste"
    assert clean_review_text(CANDY_REVIEWS) == CANDY_REVIEWS
    assert clean_review_text("  ") is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (BOOK_URL, BOOK_URL),
        (
            "HTTPS://Books.ToScrape.com/catalogue/x/index.html#reviews",
            "https://books.toscrape.com/catalogue/x/index.html",
        ),
        (f"{CANDY_URL}?utm_source=newsletter&variant=red&gclid=1", f"{CANDY_URL}?variant=red"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:8080/a?b=1", "http://example.com:8080/a?b=1"),
        ("https://user:secret@example.com/p", "https://example.com/p"),
        ("/catalogue/relative/index.html", None),
        ("ftp://example.com/a", None),
        ("javascript:alert(1)", None),
        ("not a url", None),
        (None, None),
        (42, None),
    ],
)
def test_canonicalize_url(url: object, expected: str | None) -> None:
    assert canonicalize_url(url) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("14 reviews", 14),
        (10, 10),
        ("1,204", 1204),
        ("0", 0),
        ("4.5 (123 reviews)", 123),
        ("No reviews", 0),
        ("-3", None),
        (2.5, None),
        (None, None),
    ],
)
def test_parse_count(raw: object, expected: int | None) -> None:
    assert parse_count(raw) == expected


def test_parse_timestamp_normalises_to_utc() -> None:
    expected = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    assert parse_timestamp("2026-10-17T12:00:00Z") == expected
    assert parse_timestamp("2026-10-17T14:00:00+02:00") == expected
    assert parse_timestamp("2026-10-17 12:00:00") == expected  # naive values are assumed UTC
    assert parse_timestamp("now") is None  # never invent a collection time
    assert parse_timestamp(1_700_000_000) is None


def test_make_product_id_is_stable_sha1_prefix() -> None:
    expected = "P-" + hashlib.sha1(f"books_toscrape|{BOOK_URL}".encode()).hexdigest()[:10]
    assert make_product_id("books_toscrape", BOOK_URL) == expected


# -------------------------------------------------------------------- cleaning pipeline
def test_clean_dataframe_has_contract_columns_and_dtypes() -> None:
    df = clean([book_record(), laptop_record(), candy_record()]).dataframe
    assert list(df.columns) == list(EXPECTED_SCHEMA)
    assert {column: str(dtype) for column, dtype in df.dtypes.items()} == EXPECTED_SCHEMA
    assert tuple(df.columns[: len(config.REQUIRED_CLEAN_COLUMNS) + 1][1:]) == config.REQUIRED_CLEAN_COLUMNS
    assert list(df.index) == [0, 1, 2]


def test_books_record_is_cleaned_per_contract() -> None:
    row = row_for(clean([book_record()]).dataframe, BOOK_URL)
    assert row["product_id"] == make_product_id("books_toscrape", BOOK_URL)
    assert row["price"] == 51.77
    assert row["currency"] == "GBP"  # from the £ symbol: books.toscrape has no structured currency
    assert row["rating"] == 3 and row["rating_unrounded"] == 3.0
    assert row["availability"] == "in_stock" and row["stock_quantity"] == 22
    assert not row["description"].endswith("more") and row["description"].endswith("20th anniversary.")
    assert row["review_count"] == 0
    assert row["product_url"] == BOOK_URL  # URL preserved exactly
    assert row["scraped_at"] == pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    assert row["price_raw"] == "£51.77" and row["rating_raw"] == "Three"
    assert row["data_gaps"] == "customer_reviews"


def test_webscraper_record_without_availability() -> None:
    row = row_for(clean([laptop_record()]).dataframe, LAPTOP_URL)
    assert (row["price"], row["currency"], row["rating"]) == (416.99, "USD", 2)
    assert row["availability"] == "unknown" and pd.isna(row["availability_raw"])
    assert row["description"] == '15.6", AMD E2-3800 1.3GHz, 4GB, 500GB, Windows 8.1'
    assert row["review_count"] == 2
    assert row["data_gaps"] == "availability; customer_reviews"


def test_web_scraping_dev_record_is_complete() -> None:
    row = row_for(clean([candy_record()]).dataframe, CANDY_URL)
    assert (row["price"], row["original_price"], row["currency"]) == (9.99, 12.99, "USD")
    assert row["rating"] == 5 and row["rating_unrounded"] == pytest.approx(4.7)
    assert row["availability"] == "in_stock" and pd.isna(row["stock_quantity"])
    assert row["review_text"] == CANDY_REVIEWS and row["review_count"] == 10
    assert row["data_gaps"] == ""


def test_missing_price_is_nan_not_zero() -> None:
    result = clean(
        [
            candy_record(price_raw=None, original_price_raw=None),
            laptop_record(price_raw="Call for price"),
        ]
    )
    df = result.dataframe
    assert df["price"].isna().all()
    assert not (df["price"] == 0).any()
    assert all("price" in gaps.split("; ") for gaps in df["data_gaps"])
    assert result.report["price_parse_failures"] == [
        {"source": "webscraper_io", "product_url": LAPTOP_URL, "price_raw": "Call for price"}
    ]
    assert result.report["missing_values"]["price"] == 2


def test_missing_rating_stays_missing() -> None:
    result = clean([candy_record(rating_raw=None), laptop_record(rating_raw="0")])
    df = result.dataframe
    assert df["rating"].isna().all()
    assert df["rating_unrounded"].isna().all()
    assert df["rating"].dtype == "Int64"
    assert [item["rating_raw"] for item in result.report["rating_parse_failures"]] == ["0"]


def test_pagination_duplicate_removed_keeping_most_complete_record() -> None:
    sparse_first = book_record(description=None, availability_raw=None, scraped_at="2026-10-17T11:00:00Z")
    complete_later = book_record(
        scraped_at="2026-10-17T12:00:00Z",
        listing_page_url="https://books.toscrape.com/catalogue/category/books/mystery_3/page-2.html",
    )
    result = clean([sparse_first, complete_later])
    df = result.dataframe
    assert len(df) == 1
    assert not pd.isna(df.loc[0, "description"]) and df.loc[0, "availability"] == "in_stock"
    assert result.report["duplicates_removed_same_url"] == 1
    assert result.report["duplicate_details"][0]["rule"] == data_cleaner.RULE_SAME_URL


def test_equally_complete_duplicates_keep_earliest_scrape() -> None:
    later = book_record(scraped_at="2026-10-17T13:00:00Z")
    earlier = book_record(scraped_at="2026-10-17T12:00:00Z")
    df = clean([later, earlier]).dataframe
    assert len(df) == 1
    assert df.loc[0, "scraped_at"] == pd.Timestamp("2026-10-17 12:00:00", tz="UTC")


def test_tracking_parameters_and_fragments_do_not_hide_duplicates() -> None:
    result = clean([candy_record(), candy_record(product_url=f"{CANDY_URL}?utm_source=mail#reviews")])
    assert len(result.dataframe) == 1
    assert result.dataframe.loc[0, "product_url"] == CANDY_URL
    assert result.report["duplicates_removed_same_url"] == 1


def test_identical_content_under_different_urls_is_removed() -> None:
    result = clean([candy_record(), candy_record(product_url=f"{CANDY_URL}?variant=gift-box")])
    assert len(result.dataframe) == 1
    assert result.report["duplicates_removed_same_content"] == 1
    assert result.report["duplicate_details"][0]["rule"] == data_cleaner.RULE_SAME_CONTENT


def test_same_title_in_two_sources_is_kept() -> None:
    other_store = candy_record(
        product_name="A Light in the Attic",
        price_raw="$9.99",
        description=None,
        product_url="https://web-scraping.dev/product/99",
        category="Books",
    )
    df = clean([book_record(), other_store]).dataframe
    assert len(df) == 2
    assert set(df["source"]) == {"books_toscrape", "web_scraping_dev"}


def test_same_title_same_source_different_price_is_kept() -> None:
    first = laptop_record(
        product_name="Lenovo V110-15IAP",
        price_raw="$321.94",
        description='Lenovo V110-15IAP, 15.6" HD, Celeron N3350 1.1GHz, 4GB, 128GB SSD, Windows 10 Home',
        product_url="https://webscraper.io/test-sites/e-commerce/static/product/40",
    )
    second = laptop_record(
        product_name="Lenovo V110-15IAP",
        price_raw="$356.49",
        description='Lenovo V110-15IAP, 15.6" HD, Core i3-6006U 2.0GHz, 4GB, 128GB SSD, Windows 10 Home',
        product_url="https://webscraper.io/test-sites/e-commerce/static/product/41",
    )
    df = clean([first, second]).dataframe
    assert len(df) == 2
    assert sorted(df["price"].tolist()) == [321.94, 356.49]


def test_records_missing_identifiers_are_rejected_with_reasons() -> None:
    records = [
        book_record(),
        book_record(product_name="  ", product_url="https://books.toscrape.com/catalogue/x/index.html"),
        laptop_record(product_url="/test-sites/e-commerce/static/product/99"),
        candy_record(source=None, product_url="https://web-scraping.dev/product/7"),
        candy_record(product_url=None),
        "not a record",
    ]
    result = clean(records)
    assert len(result.dataframe) == 1
    reasons = [item["reason"] for item in result.rejected_records]
    assert "malformed record: expected a JSON object" in reasons
    assert "missing product_name" in reasons
    assert any(reason.startswith("invalid product_url") for reason in reasons)
    assert "missing source" in reasons
    assert "missing product_url" in reasons
    assert result.report["rejected_missing_identifiers"] == 4
    assert result.report["rejected_malformed_records"] == 1
    assert result.report["rejection_reasons"]["invalid product_url"] == 1
    unnamed = next(item for item in result.rejected_records if item["reason"] == "missing product_name")
    assert unnamed["record"]["product_url"] == "https://books.toscrape.com/catalogue/x/index.html"  # original kept


def test_report_contents_and_record_accounting() -> None:
    records = [
        book_record(),
        book_record(scraped_at="2026-10-17T12:30:00Z"),  # pagination duplicate
        laptop_record(),
        candy_record(),
        candy_record(product_url=f"{CANDY_URL}?variant=gift-box"),  # same content, different URL
        make_record(product_name="Orphan", source="webscraper_io"),  # no URL -> rejected
    ]
    report = clean(records).report
    expected_keys = set(
        """
        raw_records rejected_missing_identifiers duplicates_removed_same_url duplicates_removed_same_content
        final_records records_per_source missing_values price_parse_failures rating_parse_failures
        currency_counts availability_counts validation_issues policies
        """.split()
    )
    assert expected_keys <= set(report)
    outcome_keys = (
        "rejected_malformed_records",
        "rejected_missing_identifiers",
        "duplicates_removed_same_url",
        "duplicates_removed_same_content",
        "removed_by_validation",
        "final_records",
    )
    assert sum(report[key] for key in outcome_keys) == report["raw_records"] == 6
    assert report["final_records"] == 3
    assert report["records_per_source"] == {"books_toscrape": 1, "web_scraping_dev": 1, "webscraper_io": 1}
    assert report["currency_counts"] == {"GBP": 1, "USD": 2}
    assert report["availability_counts"] == {"in_stock": 2, "out_of_stock": 0, "unknown": 1}
    assert set(report["missing_values"]) == set(EXPECTED_SCHEMA)
    assert report["missing_values"]["review_text"] == 2
    assert report["validation_issues"] == []
    for key in ("deduplication", "missing_values", "rounding"):
        assert isinstance(report["policies"][key], str) and report["policies"][key]


def test_scraped_at_is_utc_and_missing_timestamps_are_reported() -> None:
    result = clean(
        [
            book_record(scraped_at="2026-10-17T14:00:00+02:00"),
            laptop_record(scraped_at="2026-10-17 12:00:00"),
            candy_record(scraped_at=None),
        ]
    )
    df = result.dataframe
    assert str(df["scraped_at"].dtype) == "datetime64[ns, UTC]"
    assert df.loc[0, "scraped_at"] == pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    assert df.loc[1, "scraped_at"] == pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    assert pd.isna(df.loc[2, "scraped_at"])
    checks = [(issue["check"], issue["action"]) for issue in result.report["validation_issues"]]
    assert checks == [("scraped_at_present", "none")]


def test_cleaned_records_are_json_serialisable_with_native_types() -> None:
    result = clean([book_record(), laptop_record(), candy_record(rating_raw=None, price_raw=None)])
    records = dataframe_to_records(result.dataframe)
    json.dumps(records, allow_nan=False)
    json.dumps(result.report, allow_nan=False)
    json.dumps(result.rejected_records, allow_nan=False)
    book, laptop, candy = records
    assert isinstance(book["rating"], int) and not isinstance(book["rating"], bool)
    assert isinstance(book["price"], float) and isinstance(book["review_count"], int)
    assert book["scraped_at"] == "2026-10-17T12:00:00Z"
    assert laptop["stock_quantity"] is None and laptop["review_text"] is None
    assert candy["price"] is None and candy["rating"] is None and candy["rating_unrounded"] is None
    assert list(book) == list(EXPECTED_SCHEMA)


def test_input_records_are_not_mutated() -> None:
    records = [book_record(), laptop_record(), candy_record()]
    snapshot = copy.deepcopy(records)
    clean(records)
    assert records == snapshot


def test_accepts_full_raw_file_payload() -> None:
    payload = {"generated_at": "2026-10-17T12:00:00Z", "scrape_report": {}, "records": [book_record(), laptop_record()]}
    result = clean_product_data(payload)
    assert result.report["raw_records"] == 2 and len(result.dataframe) == 2


@pytest.mark.parametrize("records", [[], None, {"records": []}])
def test_empty_input_returns_empty_frame_with_schema(records: list | dict | None) -> None:
    result = clean_product_data(records)
    assert result.dataframe.empty
    assert {column: str(dtype) for column, dtype in result.dataframe.dtypes.items()} == EXPECTED_SCHEMA
    assert result.report["raw_records"] == 0 and result.report["final_records"] == 0
    assert result.rejected_records == []


# --------------------------------------------------------------------------- validation
def test_validate_clean_dataset_passes_clean_frame() -> None:
    df = clean([book_record(), laptop_record(), candy_record()]).dataframe
    assert validate_clean_dataset(df) == []


def test_validate_clean_dataset_reports_hard_and_soft_failures() -> None:
    df = clean([book_record(), laptop_record(), candy_record()]).dataframe
    broken = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate (source, product_url)
    broken.loc[0, "price"] = -5.0
    broken.loc[1, "rating"] = 7
    broken.loc[2, "availability"] = "maybe"
    broken.loc[1, "currency"] = "Dollars"
    broken.loc[2, "product_name"] = " "
    issues = validate_clean_dataset(broken)
    found = {(issue["check"], issue["row"], issue["severity"], issue["action"]) for issue in issues}
    assert ("price_non_negative", 0, "soft", "set_null") in found
    assert ("rating_in_range", 1, "soft", "set_null") in found
    assert ("availability_approved", 2, "soft", "set_unknown") in found
    assert ("currency_code", 1, "soft", "set_null") in found
    assert ("product_name_present", 2, "hard", "remove_row") in found
    assert ("duplicate_source_url", 3, "hard", "remove_row") in found
    assert ("duplicate_product_id", 3, "hard", "remove_row") in found
    json.dumps(issues, allow_nan=False)
    assert broken.loc[0, "price"] == -5.0  # validation never mutates its input


def test_validate_reports_missing_required_columns() -> None:
    df = clean([book_record()]).dataframe.drop(columns=["review_text"])
    issues = validate_clean_dataset(df)
    assert issues[0]["check"] == "required_columns" and "review_text" in issues[0]["message"]


def test_soft_validation_failure_nulls_field_in_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_cleaner, "parse_price", lambda raw, hint=None: (-5.0, "USD"))
    result = clean([laptop_record()])
    row = result.dataframe.iloc[0]
    assert pd.isna(row["price"])
    assert "price" in row["data_gaps"].split("; ")
    assert any(issue["check"] == "price_non_negative" for issue in result.report["validation_issues"])


def test_hard_validation_failure_removes_row_in_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_cleaner, "make_product_id", lambda source, url: "P-collision")
    result = clean([book_record(), laptop_record()])
    assert len(result.dataframe) == 1
    assert result.report["removed_by_validation"] == 1
    assert result.rejected_records[0]["reason"] == "validation: duplicate product_id"
    assert result.rejected_records[0]["record"]["product_url"] == LAPTOP_URL


# ---------------------------------------------------------------------------------- I/O
def test_save_cleaned_csv_is_excel_friendly(tmp_path) -> None:
    df = clean([book_record(product_name="Café Crème Mystery"), laptop_record()]).dataframe
    path = save_cleaned_csv(df, tmp_path / "out" / "cleaned_products.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel
    loaded = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    assert list(loaded.columns) == list(EXPECTED_SCHEMA)
    assert loaded.loc[0, "product_name"] == "Café Crème Mystery"
    assert loaded.loc[0, "price_raw"] == "£51.77"
    assert loaded.loc[0, "scraped_at"] == "2026-10-17T12:00:00Z"
    assert loaded.loc[1, "stock_quantity"] == ""  # missing values are blank, not "nan"/"<NA>"
    assert str(loaded.loc[0, "rating"]) == "3"
    assert not list(tmp_path.glob("out/*.tmp"))


def test_load_raw_records_accepts_contract_file_and_bare_list(tmp_path) -> None:
    wrapped = tmp_path / "raw.json"
    payload = {"generated_at": "2026-10-17T12:00:00Z", "scrape_report": {}, "records": [book_record()]}
    wrapped.write_text(json.dumps(payload), encoding="utf-8")
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([laptop_record()]), encoding="utf-8")
    assert load_raw_records(wrapped)[0]["product_url"] == BOOK_URL
    assert load_raw_records(bare)[0]["product_url"] == LAPTOP_URL
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_records(broken)


def test_dataframe_to_records_converts_numpy_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "i": pd.Series([1, None], dtype="Int64"),
            "f": [1.5, math.nan],
            "s": pd.Series(["x", None], dtype="string"),
            "t": pd.Series([pd.Timestamp("2026-10-17T12:00:00", tz="UTC"), pd.NaT]),
        }
    )
    assert dataframe_to_records(frame) == [
        {"i": 1, "f": 1.5, "s": "x", "t": "2026-10-17T12:00:00Z"},
        {"i": None, "f": None, "s": None, "t": None},
    ]
