"""Pandas cleaning and validation layer for scraped competitor product records.

Turns the raw records produced by ``scraper.scrape_products`` (contract Stage 1) into the
standardised, validated DataFrame described in contract Stage 2:

* text normalisation and null-like detection (``""``, ``"N/A"``, ``"-"``, NaN ... -> null);
* locale-aware price parsing with ISO-4217 currency detection (never fabricating zeros);
* rating normalisation to integers 1-5 using round-half-up;
* availability mapping to ``in_stock`` / ``out_of_stock`` / ``unknown``;
* per-source de-duplication with ``DataFrame.drop_duplicates``;
* data-quality validation with predictable, reported actions.

Everything this module returns (records, report, rejected records) is JSON-serialisable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

__all__ = [
    "CLEAN_COLUMNS",
    "CLEAN_SCHEMA",
    "CLEANING_POLICIES",
    "CleaningResult",
    "canonicalize_url",
    "clean_description",
    "clean_product_data",
    "clean_review_text",
    "clean_text",
    "dataframe_to_records",
    "format_utc_timestamp",
    "is_null_like",
    "load_raw_records",
    "make_product_id",
    "normalize_availability",
    "normalize_currency_code",
    "normalize_rating",
    "parse_count",
    "parse_price",
    "parse_rating",
    "parse_timestamp",
    "repair_mojibake",
    "save_cleaned_csv",
    "validate_clean_dataset",
]

# --------------------------------------------------------------------------- schema
RAW_FIELDS: tuple[str, ...] = (
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

# Contract Stage 2: exact column order and dtypes of the cleaned DataFrame.
CLEAN_SCHEMA: dict[str, str] = {
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
CLEAN_COLUMNS: tuple[str, ...] = tuple(CLEAN_SCHEMA)

IDENTIFIER_COLUMNS: tuple[str, ...] = ("product_name", "product_url", "source")
# Fields counted (plus a known availability) when choosing which duplicate record to keep.
COMPLETENESS_FIELDS: tuple[str, ...] = (
    "price",
    "currency",
    "rating",
    "description",
    "review_text",
    "review_count",
    "category",
    "stock_quantity",
    "original_price",
    "review_source",
)
DATA_GAP_FIELDS: tuple[str, ...] = ("price", "currency", "rating", "description", "availability", "customer_reviews")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

IN_STOCK, OUT_OF_STOCK, UNKNOWN_AVAILABILITY = config.AVAILABILITY_VALUES

RULE_SAME_URL = "same source and canonical product_url"
RULE_SAME_CONTENT = "same source, normalised name, price, currency and description (different URLs)"

# Plain-English policy statements, copied verbatim into the README / JSON report.
CLEANING_POLICIES: dict[str, str] = {
    "deduplication": (
        "Duplicates are removed with pandas drop_duplicates inside each source only: (1) same source and "
        "canonical product URL (scheme/host lower-cased, #fragment and utm_*/click-id tracking parameters "
        "removed); (2) same source with identical normalised title, price, currency and description listed "
        "under different URLs. The most complete record is kept (ties: earliest scraped_at). The same title "
        "sold by different competitors, or by the same competitor at a different price, is kept as separate "
        "products."
    ),
    "missing_values": (
        "Empty strings and null markers ('N/A', 'none', 'null', '-', NaN) become real nulls. Records without a "
        "product name, a source or a valid absolute http(s) product URL are dropped and logged with a reason. "
        "Missing or unparseable prices, currencies, ratings, descriptions, reviews and stock counts stay null "
        "(never 0, never a placeholder) and are listed per product in data_gaps; availability that cannot be "
        "determined is labelled 'unknown'. Products without a valid price are excluded from price benchmarks."
    ),
    "rounding": (
        "Ratings are first converted to a 0-5 scale ('Three' -> 3, '8/10' -> 4.0, '92%' -> 4.6, filled star "
        "glyphs counted) and then rounded half-up to an integer 1-5 (4.5 -> 5, 3.49 -> 3); the unrounded value "
        "is kept in rating_unrounded. A rating that rounds to 0 (no stars) or exceeds the scale is treated as "
        "missing. Prices are stored exactly as parsed, without rounding."
    ),
    "price_parsing": (
        "Prices are parsed locale-aware ('1,299.99', '1.299,99', '1 299,99', '1'299.00', '12,50'). A single "
        "'.' or ',' followed by exactly three digits is a thousands separator, except '.xxx' for three-decimal "
        "currencies (KWD, BHD, OMR, JOD, TND, IQD, LYD). A price range keeps its lowest bound, the word 'free' "
        "becomes 0.0, and negative or unparseable prices become null. The original string is kept in price_raw."
    ),
    "currency": (
        "The ISO-4217 code declared in the page's structured data wins; otherwise the symbol or code next to "
        "the price is used (a bare '$' is assumed to be USD and a bare '¥' JPY). Prices in different "
        "currencies are never compared or converted."
    ),
    "validation": (
        "Every cleaned record is validated before AI analysis. Hard failures (empty name, missing source, "
        "missing or invalid URL, duplicate source+URL) remove the row; soft failures (negative price, rating "
        "outside 1-5, unapproved availability, malformed currency, non-UTC timestamp) null the offending field "
        "(availability falls back to 'unknown'). Every action is listed in the cleaning report."
    ),
}

# ---------------------------------------------------------------------- text helpers
NULL_LIKE_TOKENS = frozenset(
    {"", "n/a", "na", "n.a.", "none", "null", "nan", "nil", "undefined", "-", "--", "–", "—"}
)
_WHITESPACE_RE = re.compile(r"\s+")  # Unicode-aware: also matches NBSP, narrow NBSP, thin space ...
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")  # zero-width chars, BOM, soft hyphen
_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_MORE_MARKER_RE = re.compile(r"\s*(?:\.{3}|…)\s*more\s*$", re.IGNORECASE)
_RATING_ONLY_EXCERPT_RE = re.compile(r"\[\s*\d+(?:[.,]\d+)?\s*/\s*\d+\s*\]")
_MOJIBAKE_HINT_RE = re.compile("Â|Ã|â€|â‚")
_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("â‚¬", "€"),
    ("â‚¹", "₹"),
    ("â€™", "’"),
    ("â€˜", "‘"),
    ("â€œ", "“"),
    ("â€\x9d", "”"),
    ("â€“", "–"),
    ("â€”", "—"),
    ("â€¦", "…"),
    ("Â£", "£"),
    ("Â¥", "¥"),
    ("Â©", "©"),
    ("Â®", "®"),
    ("Â°", "°"),
    ("Â\xa0", "\xa0"),
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã¶", "ö"),
    ("Ã¼", "ü"),
    ("Ã¤", "ä"),
)
_COMPARISON_TOKEN_RE = re.compile(r"[^\W_]+")


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _INVISIBLE_RE.sub("", text)).strip()


def is_null_like(value: Any) -> bool:
    """Return True for None, NaN, pd.NA, NaT and null-marker strings such as ``""``, ``"N/A"`` or ``"-"``."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return _collapse_whitespace(value).casefold() in NULL_LIKE_TOKENS
    if isinstance(value, (list, tuple, set, frozenset, dict, np.ndarray)):
        return False
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def repair_mojibake(text: str) -> str:
    """Undo UTF-8 text that was mis-decoded as Windows-1252/Latin-1 (``"Â£51.77"`` -> ``"£51.77"``)."""
    for _ in range(3):  # tolerate text that was mis-decoded more than once
        if not _MOJIBAKE_HINT_RE.search(text):
            return text
        for codec in ("cp1252", "latin-1"):
            try:
                text = text.encode(codec).decode("utf-8")
                break
            except UnicodeError:
                continue
        else:  # mixed content: only replace well-known broken sequences
            for broken, fixed in _MOJIBAKE_REPLACEMENTS:
                text = text.replace(broken, fixed)
            return text
    return text


def _unescape_entities(text: str) -> str:
    # Only complete entities ("&amp;", "&#39;"); html.unescape alone would also rewrite "&not..." prefixes.
    return _ENTITY_RE.sub(lambda match: html.unescape(match.group(0)), text)


def clean_text(value: Any, *, strip_more_marker: bool = False) -> str | None:
    """Normalise a scraped text value; return None for null-like input.

    Repairs mojibake, decodes HTML entities, removes zero-width characters and collapses all
    whitespace (including NBSP) to single spaces. With ``strip_more_marker`` a trailing
    ``"...more"`` (appended by books.toscrape.com to descriptions) is removed.
    """
    if is_null_like(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = value if isinstance(value, str) else str(value)
    text = _collapse_whitespace(_unescape_entities(repair_mojibake(text)))
    if strip_more_marker:
        text = _MORE_MARKER_RE.sub("", text).rstrip()
    return None if is_null_like(text) else text


def clean_description(value: Any) -> str | None:
    """Clean a product description, dropping the books.toscrape.com ``"...more"`` marker."""
    return clean_text(value, strip_more_marker=True)


def clean_review_text(value: Any) -> str | None:
    """Clean ``" || "``-joined review excerpts; drop empty excerpts and bare rating prefixes like ``"[5/5]"``."""
    text = clean_text(value)
    if text is None:
        return None
    excerpts = []
    for part in text.split("||"):
        excerpt = clean_text(part)
        if excerpt and not _RATING_ONLY_EXCERPT_RE.fullmatch(excerpt):
            excerpts.append(excerpt)
    return config.REVIEW_SEPARATOR.join(excerpts) if excerpts else None


def _raw_text(value: Any) -> str | None:
    """Original value as a string for traceability columns (null-like -> None, otherwise unchanged)."""
    if is_null_like(value):
        return None
    text = value if isinstance(value, str) else str(value)
    return text.strip() or None


def _comparison_key(value: Any) -> str:
    """Case-, accent-width- and punctuation-insensitive key used for content de-duplication."""
    text = clean_text(value)
    if text is None:
        return ""
    return " ".join(_COMPARISON_TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold()))


# -------------------------------------------------------------------- price parsing
# Active ISO-4217 codes recognised inside price strings. Codes that are common English words
# (ALL, TOP, CUP, MOP, GEL, BOB, SOS, PEN) or brand names (AMD) are deliberately omitted.
ISO_CURRENCY_CODES = frozenset(
    """
    AED ARS AUD BDT BGN BHD BRL CAD CHF CLP CNY COP CZK DKK DZD EGP EUR GBP GHS HKD HUF IDR ILS INR
    IQD IRR ISK JOD JPY KES KRW KWD KZT LBP LKR LYD MAD MXN MYR NGN NOK NZD OMR PHP PKR PLN QAR RON
    RSD RUB SAR SEK SGD THB TND TRY TWD UAH USD UYU VND XAF XOF YER ZAR
    """.split()
)
CURRENCY_CODE_ALIASES: dict[str, str] = {"RMB": "CNY", "NIS": "ILS"}
NON_CURRENCY_CODES = frozenset({"XXX", "XTS"})  # ISO-4217 "no currency" / "testing" codes
# Currencies whose minor unit has three decimals: "1.299 KWD" is 1.299, not 1299.
THREE_DECIMAL_CURRENCIES = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})
# Symbols and local abbreviations -> ISO-4217. A bare "$" is assumed to be USD and "¥" JPY.
CURRENCY_SYMBOLS: dict[str, str] = {
    "US$": "USD",
    "CA$": "CAD",
    "C$": "CAD",
    "AU$": "AUD",
    "A$": "AUD",
    "NZ$": "NZD",
    "HK$": "HKD",
    "NT$": "TWD",
    "S$": "SGD",
    "R$": "BRL",
    "MX$": "MXN",
    "E£": "EGP",
    "L.E.": "EGP",
    "LE": "EGP",
    "ج.م.": "EGP",
    "ج.م": "EGP",
    "ر.س": "SAR",
    "د.إ": "AED",
    "د.ك": "KWD",
    "CN¥": "CNY",
    "JP¥": "JPY",
    "RMB": "CNY",
    "zł": "PLN",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "₺": "TRY",
    "₪": "ILS",
    "₱": "PHP",
    "₫": "VND",
    "฿": "THB",
    "₦": "NGN",
    "₴": "UAH",
    "₸": "KZT",
}
_LETTER_BEFORE = r"(?<![^\W\d_])"  # not preceded by a letter
_LETTER_AFTER = r"(?![^\W\d_])"  # not followed by a letter


def _symbol_pattern(token: str) -> str:
    pattern = re.escape(token)
    if token[0].isalpha():
        pattern = _LETTER_BEFORE + pattern
    if token[-1].isalpha():
        pattern += _LETTER_AFTER
    return pattern


_SYMBOL_ALTERNATION = "|".join(_symbol_pattern(token) for token in sorted(CURRENCY_SYMBOLS, key=len, reverse=True))
_SYMBOL_ANY_RE = re.compile(f"(?:{_SYMBOL_ALTERNATION})")
_SYMBOL_AT_END_RE = re.compile(f"(?:{_SYMBOL_ALTERNATION})$")
_SYMBOL_AT_START_RE = re.compile(f"^(?:{_SYMBOL_ALTERNATION})")
_ISO_TOKEN_RE = re.compile(_LETTER_BEFORE + r"([A-Za-z]{3})" + _LETTER_AFTER)
_ISO_AT_END_RE = re.compile(_LETTER_BEFORE + r"([A-Za-z]{3})$")
_ISO_AT_START_RE = re.compile(r"^([A-Za-z]{3})" + _LETTER_AFTER)
_ISO_CODE_RE = re.compile(r"[A-Za-z]{3}")

# A monetary amount: Indian lakh grouping, Western/European grouping (",", ".", space, apostrophe)
# or plain digits, optionally followed by a decimal part.
_AMOUNT_RE = re.compile(
    r"(?<![0-9])"
    r"(?P<int>"
    r"[0-9]{1,2}(?:,[0-9]{2})+,[0-9]{3}"
    r"|[0-9]{1,3}(?P<sep>[,.' ’])[0-9]{3}(?:(?P=sep)[0-9]{3})*"
    r"|[0-9]+"
    r")"
    r"(?:(?P<dec>[.,])(?P<frac>[0-9]+))?"
    r"(?![0-9])"
)
_FREE_RE = re.compile(r"(?:price\s*[:\-]?\s*)?free[!.]*", re.IGNORECASE)
_MINUS_SIGNS = ("-", "−")
_RANGE_SEPARATORS = frozenset({"-", "–", "—", "−", "~", "to"})
_EASTERN_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬٪", "01234567890123456789.,%")


def _normalise_numeric_text(raw: str) -> str:
    """Prepare a price/rating/count string: mojibake, NFKC (full-width digits), Eastern numerals."""
    text = unicodedata.normalize("NFKC", repair_mojibake(raw)).translate(_EASTERN_NUMERALS)
    return _collapse_whitespace(text)


def _iso_from_token(token: str) -> str | None:
    code = token.upper()
    code = CURRENCY_CODE_ALIASES.get(code, code)
    return code if code in ISO_CURRENCY_CODES else None


def normalize_currency_code(value: Any) -> str | None:
    """Normalise a declared currency (``"usd"``, ``" EUR "``, ``"£"``) to an ISO-4217 code, else None."""
    text = clean_text(value)
    if text is None:
        return None
    compact = unicodedata.normalize("NFKC", text).strip()
    alias = CURRENCY_CODE_ALIASES.get(compact.upper())
    if alias:
        return alias
    if _ISO_CODE_RE.fullmatch(compact):
        # Structured data declares ISO-4217, so any 3-letter code is trusted (e.g. "ALL" = Albanian lek).
        return compact.upper() if compact.upper() not in NON_CURRENCY_CODES else None
    return CURRENCY_SYMBOLS.get(compact)


def _currency_at_end(text: str) -> tuple[str | None, int]:
    """Currency token that ends ``text`` and the index where the token starts."""
    match = _SYMBOL_AT_END_RE.search(text)
    if match:
        return CURRENCY_SYMBOLS[match.group(0)], match.start()
    match = _ISO_AT_END_RE.search(text)
    if match and _iso_from_token(match.group(1)):
        return _iso_from_token(match.group(1)), match.start()
    return None, len(text)


def _currency_at_start(text: str) -> str | None:
    match = _SYMBOL_AT_START_RE.match(text)
    if match:
        return CURRENCY_SYMBOLS[match.group(0)]
    match = _ISO_AT_START_RE.match(text)
    return _iso_from_token(match.group(1)) if match else None


def _currency_anywhere(text: str) -> str | None:
    """First currency symbol, or upper-case ISO code, anywhere in the text."""
    candidates: list[tuple[int, str]] = []
    symbol = _SYMBOL_ANY_RE.search(text)
    if symbol:
        candidates.append((symbol.start(), CURRENCY_SYMBOLS[symbol.group(0)]))
    for match in _ISO_TOKEN_RE.finditer(text):
        code = _iso_from_token(match.group(1)) if match.group(1).isupper() else None
        if code:
            candidates.append((match.start(), code))
            break
    return min(candidates)[1] if candidates else None


def _adjacent_currency(text: str, match: re.Match[str]) -> str | None:
    """Currency written directly before (``"$9.99"``, ``"USD 249"``) or after (``"89,50 €"``) an amount."""
    prefix = text[: match.start()].rstrip()
    if prefix.endswith(_MINUS_SIGNS):
        prefix = prefix[:-1].rstrip()
    code, _ = _currency_at_end(prefix)
    return code or _currency_at_start(text[match.end() :].lstrip())


def _is_negative(text: str, match: re.Match[str]) -> bool:
    prefix = text[: match.start()].rstrip()
    code, start = _currency_at_end(prefix)
    if code is not None:
        prefix = prefix[:start].rstrip()
    if not prefix.endswith(_MINUS_SIGNS):
        return False
    before = prefix[:-1].rstrip()
    return not (before and before[-1].isdigit())  # "10 - $25" is a separator, not a sign


def _is_range_separator(between: str) -> bool:
    residue = _SYMBOL_ANY_RE.sub(" ", between)
    residue = _ISO_TOKEN_RE.sub(lambda m: " " if _iso_from_token(m.group(1)) else m.group(0), residue)
    return residue.strip().casefold() in _RANGE_SEPARATORS


def _interpret_amount(match: re.Match[str], currency: str | None) -> Decimal | None:
    """Turn a matched number into a Decimal, resolving thousands vs decimal separators."""
    integer, sep, dec, frac = match.group("int", "sep", "dec", "frac")
    if sep is not None:
        if dec is not None and dec == sep:
            return None  # "1,299,99": the same character cannot group and separate decimals
        groups = integer.split(sep)
        is_single_ambiguous_group = dec is None and len(groups) == 2 and sep in ".,"
        if is_single_ambiguous_group and (
            groups[0] == "0" or (sep == "." and currency in THREE_DECIMAL_CURRENCIES)
        ):
            return Decimal(f"{groups[0]}.{groups[1]}")
        integer = "".join(groups)
    elif "," in integer:  # Indian lakh grouping
        if dec == ",":
            return None
        integer = integer.replace(",", "")
    try:
        return Decimal(f"{integer}.{frac}") if dec is not None else Decimal(integer)
    except InvalidOperation:
        return None


def _to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return Decimal(repr(number)) if math.isfinite(number) else None
    return None


def _non_negative_float(value: Decimal | None) -> float | None:
    if value is None or not value.is_finite() or value < 0:
        return None
    return float(value)


def parse_price(raw: Any, currency_hint: Any = None) -> tuple[float | None, str | None]:
    """Parse a displayed price into ``(amount, ISO currency)``; unparseable -> ``(None, currency)``.

    * ``currency_hint`` (the structured ``currency_raw`` ISO code) wins over symbols when valid.
    * Symbols/codes may precede or follow the amount: ``"$1,299.99"``, ``"USD 249.00"``, ``"89,50 €"``.
    * ``"1.299,99"``, ``"1 299,99"``, ``"1'299.00"`` and ``"12,50"`` are understood; a lone ``.``/``,``
      followed by exactly three digits is a thousands separator unless the currency has three
      decimals (KWD, BHD, OMR, JOD, TND, IQD, LYD) and the separator is ``"."``.
    * A range (``"$34.00 – $45.00"``) returns its lowest bound; ``"free"`` returns 0.0.
    * Negative or unparseable prices return None, never 0.
    """
    hint = normalize_currency_code(currency_hint)
    if is_null_like(raw) or isinstance(raw, (bool, np.bool_)):
        return None, hint
    if isinstance(raw, (int, float, Decimal, np.integer, np.floating)):
        return _non_negative_float(_to_decimal(raw)), hint

    text = _normalise_numeric_text(str(raw))
    if _FREE_RE.fullmatch(text):
        return 0.0, hint
    matches = [m for m in _AMOUNT_RE.finditer(text) if not text[m.end() :].lstrip().startswith("%")]
    if not matches:
        return None, hint or _currency_anywhere(text)

    # Prefer the first amount that carries a currency ("2 for $10" -> 10), else the first amount.
    index = next((i for i, m in enumerate(matches) if _adjacent_currency(text, m)), 0)
    match = matches[index]
    currency = hint or _adjacent_currency(text, match) or _currency_anywhere(text)

    partner: int | None = None
    if index + 1 < len(matches) and _is_range_separator(text[match.end() : matches[index + 1].start()]):
        partner = index + 1
    elif index > 0 and _is_range_separator(text[matches[index - 1].end() : match.start()]):
        partner = index - 1
    first = matches[min(index, partner)] if partner is not None else match
    if _is_negative(text, first):
        return None, currency

    amount = _interpret_amount(match, currency)
    if amount is not None and partner is not None:
        other = _interpret_amount(matches[partner], currency)
        if other is not None:
            amount = min(amount, other)
    return _non_negative_float(amount), currency


# ------------------------------------------------------------------- rating parsing
RATING_SCALE = Decimal(5)
_RATING_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_FILLED_STARS = frozenset("★⭐")
_EMPTY_STARS = frozenset("☆✩")
_NUMBER = r"[0-9]+(?:[.,][0-9]+)?"
_RATING_FRACTION_RE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?:/|out\s+of|of)\s*(?P<scale>{_NUMBER})", re.IGNORECASE
)
_RATING_PERCENT_RE = re.compile(rf"(?P<value>{_NUMBER})\s*%")
_RATING_STARS_RE = re.compile(rf"(?P<value>{_NUMBER})\s*-?\s*stars?\b", re.IGNORECASE)
_RATING_WORD_RE = re.compile(r"\b(zero|one|two|three|four|five)\b", re.IGNORECASE)
_RATING_NUMBER_RE = re.compile(_NUMBER)
_HALF_RE = re.compile(r"([0-9])\s*½")


def _decimal_from_text(number: str) -> Decimal:
    return Decimal(number.replace(",", "."))


def _rating_on_scale(value: Decimal | None, scale: Decimal) -> float | None:
    if value is None or not value.is_finite() or scale <= 0 or value < 0 or value > scale:
        return None
    return float(value / scale * RATING_SCALE)


def parse_rating(raw: Any) -> float | None:
    """Parse a rating into a 0-5 float, or None when missing/invalid.

    Understands number words (``"Three"``, ``"star-rating Three"``), ``"4.5 out of 5"``,
    ``"Rating: 4 stars"``, star glyphs (``"★★★★☆"`` counts filled stars), fractions on any scale
    (``"3/5"``, ``"8/10"`` -> 4.0, ``"4,5/5"``), percentages (``"92%"`` -> 4.6) and bare numbers
    up to 5 (assumed to be on a 5-point scale). Values outside their scale return None.
    """
    if is_null_like(raw) or isinstance(raw, (bool, np.bool_)):
        return None
    if isinstance(raw, (int, float, Decimal, np.integer, np.floating)):
        return _rating_on_scale(_to_decimal(raw), RATING_SCALE)
    text = clean_text(raw)
    if text is None:
        return None
    if not any(ch.isdigit() for ch in text):
        filled = sum(ch in _FILLED_STARS for ch in text)
        empty = sum(ch in _EMPTY_STARS for ch in text)
        if filled + empty:
            return _rating_on_scale(Decimal(filled), Decimal(max(filled + empty, 5)))
    text = _normalise_numeric_text(_HALF_RE.sub(r"\1.5", text).replace("½", "0.5"))

    match = _RATING_FRACTION_RE.search(text)
    if match:
        return _rating_on_scale(_decimal_from_text(match["value"]), _decimal_from_text(match["scale"]))
    match = _RATING_PERCENT_RE.search(text)
    if match:
        return _rating_on_scale(_decimal_from_text(match["value"]), Decimal(100))
    match = _RATING_STARS_RE.search(text)
    if match:
        return _rating_on_scale(_decimal_from_text(match["value"]), RATING_SCALE)
    match = _RATING_WORD_RE.search(text)
    if match:
        return _rating_on_scale(Decimal(_RATING_WORDS[match.group(1).lower()]), RATING_SCALE)
    match = _RATING_NUMBER_RE.search(text)
    if match:
        return _rating_on_scale(_decimal_from_text(match.group(0)), RATING_SCALE)
    return None


def normalize_rating(value: Any) -> int | None:
    """Round a 0-5 rating half-up to an integer within ``config.RATING_MIN..RATING_MAX``.

    ``4.5 -> 5``, ``3.49 -> 3``. Strings are parsed with :func:`parse_rating` first. Missing,
    invalid, out-of-scale values and ratings that round to 0 ("not rated") return None.
    """
    if isinstance(value, str):
        value = parse_rating(value)
    number = _to_decimal(value)
    if number is None or number < 0 or number > config.RATING_MAX:
        return None
    rounded = int(number.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return rounded if config.RATING_MIN <= rounded <= config.RATING_MAX else None


# --------------------------------------------------------------- availability / count
_SCHEMA_AVAILABILITY: dict[str, str] = {
    "instock": IN_STOCK,
    "limitedavailability": IN_STOCK,
    "onlineonly": IN_STOCK,
    "outofstock": OUT_OF_STOCK,
    "soldout": OUT_OF_STOCK,
    "discontinued": OUT_OF_STOCK,
    "backorder": OUT_OF_STOCK,
    "preorder": UNKNOWN_AVAILABILITY,
    "presale": UNKNOWN_AVAILABILITY,
    "instoreonly": UNKNOWN_AVAILABILITY,
}
_SCHEMA_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?schema\.org/(?P<token>[A-Za-z]+)/?", re.IGNORECASE)
_SCHEMA_PREFIX_RE = re.compile(r"schema:(?P<token>[A-Za-z]+)", re.IGNORECASE)
_OUT_OF_STOCK_RE = re.compile(
    r"\b(?:out[\s-]*of[\s-]*stock|not\s+in\s+stock|no\s+(?:longer\s+)?(?:in\s+)?stock|not\s+available"
    r"|no\s+longer\s+available|sold[\s-]*out|unavailable|discontinued|back[\s-]*order(?:ed)?)\b",
    re.IGNORECASE,
)
_UNKNOWN_STOCK_RE = re.compile(
    r"\b(?:pre[\s-]*order(?:s|ed)?|pre[\s-]*sale|coming\s+soon|available\s+soon|in[\s-]*store\s+only)\b",
    re.IGNORECASE,
)
_IN_STOCK_RE = re.compile(
    r"\b(?:in[\s-]*stock|available|only\s+\d+\s+left|\d+\s+left|limited\s+(?:stock|availability)"
    r"|low\s+stock|ready\s+to\s+ship|ships\s+(?:today|now|within))\b",
    re.IGNORECASE,
)
_STOCK_QUANTITY_RE = re.compile(
    r"(?P<qty>\d[\d,]*)\s*(?:available|in\s+stock|left|units?|items?|pcs|pieces)\b"
    r"|stock\s*:\s*(?P<qty2>\d[\d,]*)",
    re.IGNORECASE,
)
_NO_COUNT_RE = re.compile(r"\bno\s+(?:customer\s+)?(?:reviews?|ratings?)\b", re.IGNORECASE)


def _stock_quantity(text: str) -> int | None:
    match = _STOCK_QUANTITY_RE.search(text)
    if not match:
        return None
    return int((match.group("qty") or match.group("qty2")).replace(",", ""))


def normalize_availability(raw: Any) -> tuple[str, int | None]:
    """Map raw availability to ``(status, stock_quantity)`` with status in ``config.AVAILABILITY_VALUES``.

    Negative phrases are checked first ("out of stock", "not in stock", "sold out", "unavailable",
    "discontinued"), then schema.org tokens/URLs, pre-order wording and positive phrases.
    ``"In stock (22 available)"`` -> ``("in_stock", 22)``; missing -> ``("unknown", None)``.
    """
    if isinstance(raw, (bool, np.bool_)):
        return (IN_STOCK if raw else OUT_OF_STOCK), None
    if is_null_like(raw):
        return UNKNOWN_AVAILABILITY, None
    if isinstance(raw, (int, np.integer)):
        quantity = int(raw)
        if quantity < 0:
            return UNKNOWN_AVAILABILITY, None
        return (IN_STOCK if quantity > 0 else OUT_OF_STOCK), quantity
    text = clean_text(raw)
    if text is None:
        return UNKNOWN_AVAILABILITY, None

    quantity = _stock_quantity(text)
    if _OUT_OF_STOCK_RE.search(text):
        return OUT_OF_STOCK, quantity
    schema_match = _SCHEMA_URL_RE.fullmatch(text) or _SCHEMA_PREFIX_RE.fullmatch(text)
    token = schema_match.group("token") if schema_match else (text if " " not in text else None)
    if token and token.casefold() in _SCHEMA_AVAILABILITY:
        return _SCHEMA_AVAILABILITY[token.casefold()], None
    if _UNKNOWN_STOCK_RE.search(text):
        return UNKNOWN_AVAILABILITY, quantity
    if _IN_STOCK_RE.search(text) or quantity is not None:
        if quantity == 0:
            return OUT_OF_STOCK, 0
        return IN_STOCK, quantity
    return UNKNOWN_AVAILABILITY, None


def parse_count(raw: Any) -> int | None:
    """Parse a non-negative integer count such as a review count (``"14 reviews"``, ``10``, ``"1,204"``)."""
    if is_null_like(raw) or isinstance(raw, (bool, np.bool_)):
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw) if raw >= 0 else None
    if isinstance(raw, (float, np.floating)):
        number = float(raw)
        return int(number) if math.isfinite(number) and number >= 0 and number.is_integer() else None
    text = _normalise_numeric_text(str(raw))
    if _NO_COUNT_RE.search(text):
        return 0
    for match in _AMOUNT_RE.finditer(text):
        if match.group("dec") is not None:
            continue  # "4.5 (123 reviews)": a decimal number is not a count
        if _is_negative(text, match):
            return None
        amount = _interpret_amount(match, None)
        if amount is None or amount != amount.to_integral_value():
            return None
        return int(amount)
    return None


# --------------------------------------------------------------------- URLs and time
TRACKING_QUERY_KEYS = frozenset({"gclid", "fbclid", "msclkid"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_HOST_RE = re.compile(r"[\w.-]+|[0-9A-Fa-f:.]+")


def _strip_tracking_params(query: str) -> str:
    kept = []
    for part in query.split("&"):
        if not part:
            continue
        key = unquote_plus(part.split("=", 1)[0]).strip().casefold()
        if key.startswith("utm_") or key in TRACKING_QUERY_KEYS:
            continue
        kept.append(part)  # untouched, so the remaining query keeps its original encoding
    return "&".join(kept)


def canonicalize_url(url: Any) -> str | None:
    """Canonical absolute http(s) URL, or None when missing/invalid.

    Lower-cases scheme and host, drops default ports, credentials, the fragment and tracking
    parameters (``utm_*``, ``gclid``, ``fbclid``, ``msclkid``); path and other query parameters
    are preserved exactly.
    """
    if not isinstance(url, str) or is_null_like(url):
        return None
    text = _INVISIBLE_RE.sub("", url).strip().replace(" ", "%20")
    if _WHITESPACE_RE.search(text):
        return None
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname
    if scheme not in _DEFAULT_PORTS or not host or not _HOST_RE.fullmatch(host):
        return None
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", _strip_tracking_params(parts.query), ""))


def make_product_id(source: str, product_url: str) -> str:
    """Stable join key shared by the CSV and JSON outputs: ``"P-" + sha1(source|url)[:10]``."""
    return "P-" + hashlib.sha1(f"{source}|{product_url}".encode()).hexdigest()[:10]


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp to a tz-aware UTC ``pd.Timestamp``; naive values are assumed to be UTC.

    Epoch numbers and strings without digits (e.g. ``"now"``) are rejected rather than guessed.
    """
    if is_null_like(value) or isinstance(value, (bool, np.bool_, int, float, np.number)):
        return None
    if isinstance(value, str) and not any(ch.isdigit() for ch in value):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(stamp) or not 1678 <= stamp.year <= 2261:  # stay inside the nanosecond range
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def format_utc_timestamp(value: Any) -> str | None:
    """Format a timestamp as ``"YYYY-MM-DDTHH:MM:SSZ"`` (UTC); None for missing values."""
    stamp = value if isinstance(value, pd.Timestamp) else parse_timestamp(value)
    if stamp is None or pd.isna(stamp):
        return None
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.strftime(TIMESTAMP_FORMAT)


# ------------------------------------------------------------------- JSON conversion
def _to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values to JSON-safe native Python values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, np.datetime64)):
        return format_utc_timestamp(pd.Timestamp(value)) if not pd.isna(value) else None
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, Decimal)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [_to_json_safe(item) for item in value]
    if is_null_like(value):
        return None
    return str(value)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe dicts (NaN/NA -> None, Int64 -> int, Timestamp -> ISO ``Z``)."""
    columns = [str(column) for column in df.columns]
    return [
        {column: _to_json_safe(value) for column, value in zip(columns, row)}
        for row in df.itertuples(index=False, name=None)
    ]


# ----------------------------------------------------------------- cleaning pipeline
@dataclass
class CleaningResult:
    """Output of :func:`clean_product_data`."""

    dataframe: pd.DataFrame
    report: dict[str, Any]
    rejected_records: list[dict[str, Any]] = field(default_factory=list)


def _map_values(series: pd.Series, func: Callable[[Any], Any]) -> pd.Series:
    """Apply ``func`` element-wise and keep an object Series (``Series.map`` would infer dtypes)."""
    return pd.Series([func(value) for value in series.tolist()], index=series.index, dtype=object)


def _rejection(record: Any, reason: str) -> dict[str, Any]:
    return {"record": _to_json_safe(record), "reason": reason}


def _shorten(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _build_raw_frame(records: list[Mapping[str, Any]], positions: list[int]) -> pd.DataFrame:
    """DataFrame of raw records (object dtype, index = position in the raw list), all raw keys present."""
    index = pd.Index(positions, dtype="int64")
    frame = pd.DataFrame([dict(record) for record in records], index=index, dtype=object)
    missing = [column for column in RAW_FIELDS if column not in frame.columns]
    if missing:
        frame = frame.assign(**{column: pd.Series(None, index=index, dtype=object) for column in missing})
    frame = frame.loc[:, list(RAW_FIELDS)]
    return frame.assign(_row=frame.index.to_numpy())


def _normalise_text_fields(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        product_name=_map_values(frame["product_name"], clean_text),
        source=_map_values(frame["source"], clean_text),
        category=_map_values(frame["category"], clean_text),
        review_source=_map_values(frame["review_source"], clean_text),
        description=_map_values(frame["description"], clean_description),
        review_text=_map_values(frame["review_text"], clean_review_text),
        _url_input=_map_values(frame["product_url"], _raw_text),
        product_url=_map_values(frame["product_url"], canonicalize_url),
    )


def _split_missing_identifiers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Drop rows lacking product_name, source or a valid product_url; return their reasons."""
    missing = frame[["product_name", "source", "_url_input"]].isna()
    invalid_url = frame["_url_input"].notna() & frame["product_url"].isna()
    failing = missing.any(axis=1) | invalid_url
    reasons: dict[int, str] = {}
    for row in frame.index[failing]:
        problems = [f"missing {column}" for column in ("product_name", "source") if missing.at[row, column]]
        if missing.at[row, "_url_input"]:
            problems.append("missing product_url")
        elif invalid_url.at[row]:
            url = _shorten(str(frame.at[row, "_url_input"]))
            problems.append(f"invalid product_url {url!r} (expected an absolute http(s) URL)")
        reasons[int(row)] = "; ".join(problems)
        logger.warning("Rejected raw record #%d: %s", row, reasons[int(row)])
    return frame.loc[~failing], reasons


def _parse_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse prices, currencies, ratings, availability, counts and timestamps into typed columns."""
    index = frame.index
    prices = [
        parse_price(raw, hint) for raw, hint in zip(frame["price_raw"].tolist(), frame["currency_raw"].tolist())
    ]
    currencies = [currency for _, currency in prices]
    original_prices = [
        parse_price(raw, currency)[0] for raw, currency in zip(frame["original_price_raw"].tolist(), currencies)
    ]
    unrounded = [parse_rating(raw) for raw in frame["rating_raw"].tolist()]
    ratings = [normalize_rating(value) for value in unrounded]
    unrounded = [value if rating is not None else None for value, rating in zip(unrounded, ratings)]
    availability = [normalize_availability(raw) for raw in frame["availability_raw"].tolist()]
    review_counts = [parse_count(raw) for raw in frame["review_count"].tolist()]
    stamps = [parse_timestamp(value) for value in frame["scraped_at"].tolist()]
    scraped_at = pd.to_datetime(pd.Series(stamps, index=index, dtype=object), utc=True)

    parsed = frame.assign(
        price=pd.Series([amount for amount, _ in prices], index=index, dtype="float64"),
        currency=pd.Series(currencies, index=index, dtype=object),
        original_price=pd.Series(original_prices, index=index, dtype="float64"),
        rating_unrounded=pd.Series(unrounded, index=index, dtype="float64"),
        rating=pd.Series(ratings, index=index, dtype="Int64"),
        availability=pd.Series([status for status, _ in availability], index=index, dtype=object),
        stock_quantity=pd.Series([quantity for _, quantity in availability], index=index, dtype="Int64"),
        review_count=pd.Series(review_counts, index=index, dtype="Int64"),
        scraped_at=scraped_at.astype("datetime64[ns, UTC]"),
        price_raw=_map_values(frame["price_raw"], _raw_text),
        rating_raw=_map_values(frame["rating_raw"], _raw_text),
        availability_raw=_map_values(frame["availability_raw"], _raw_text),
    )
    known_fields = parsed[list(COMPLETENESS_FIELDS)].notna().sum(axis=1)
    known_availability = parsed["availability"].ne(UNKNOWN_AVAILABILITY).astype("int64")
    return parsed.assign(_completeness=known_fields + known_availability)


def _drop_duplicate_records(
    frame: pd.DataFrame, subset: list[str], rule: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """``drop_duplicates`` on ``subset`` keeping the most complete record (ties: earliest scrape, then input order)."""
    if len(frame) < 2:
        return frame, []
    ordered = frame.sort_values(
        by=["_completeness", "scraped_at", "_row"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    )
    kept = ordered.drop_duplicates(subset=subset, keep="first")
    removed = ordered.loc[~ordered.index.isin(kept.index)]
    kept_by_key = {
        key: row for key, row in zip(kept[subset].itertuples(index=False, name=None), kept.index)
    }
    details = []
    for key, row in zip(removed[subset].itertuples(index=False, name=None), removed.index):
        kept_row = kept_by_key[key]
        details.append(
            {
                "rule": rule,
                "source": frame.at[row, "source"],
                "product_name": frame.at[row, "product_name"],
                "removed_product_url": frame.at[row, "product_url"],
                "kept_product_url": frame.at[kept_row, "product_url"],
            }
        )
        logger.debug("Duplicate removed (%s): %s", rule, frame.at[row, "product_url"])
    return frame.loc[frame.index.isin(kept.index)], details


def _drop_content_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Same source + identical normalised name, price, currency and description under different URLs."""
    eligible = frame.loc[frame["price"].notna()]  # without a price there is too little evidence
    if len(eligible) < 2:
        return frame, []
    keyed = eligible.assign(
        _name_key=_map_values(eligible["product_name"], _comparison_key),
        _description_key=_map_values(eligible["description"], _comparison_key),
        _price_key=_map_values(eligible["price"], lambda value: format(float(value), ".6f")),
        _currency_key=_map_values(eligible["currency"], lambda value: value or ""),
    )
    subset = ["source", "_name_key", "_price_key", "_currency_key", "_description_key"]
    deduplicated, details = _drop_duplicate_records(keyed, subset, RULE_SAME_CONTENT)
    removed = keyed.index.difference(deduplicated.index)
    return frame.drop(index=removed), details


def _data_gaps(frame: pd.DataFrame) -> list[str]:
    flags = pd.DataFrame(
        {
            "price": frame["price"].isna(),
            "currency": frame["currency"].isna(),
            "rating": frame["rating"].isna(),
            "description": frame["description"].isna(),
            "availability": _map_values(
                frame["availability"], lambda value: is_null_like(value) or value == UNKNOWN_AVAILABILITY
            ).astype(bool),
            "customer_reviews": frame["review_text"].isna(),
        },
        index=frame.index,
    )
    return ["; ".join(name for name in DATA_GAP_FIELDS if row[name]) for row in flags.to_dict("records")]


def _to_clean_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Add product_id and data_gaps, then select contract columns in order with contract dtypes."""
    product_ids = [make_product_id(source, url) for source, url in zip(frame["source"], frame["product_url"])]
    frame = frame.assign(
        product_id=pd.Series(product_ids, index=frame.index, dtype=object),
        data_gaps=pd.Series(_data_gaps(frame), index=frame.index, dtype=object),
    )
    return frame.loc[:, list(CLEAN_COLUMNS)].astype(CLEAN_SCHEMA)


def _apply_validation(df: pd.DataFrame, issues: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[Any, list[str]]]:
    """Apply validation actions; return the fixed frame and removed rows (index label -> messages)."""
    result = df.copy()
    removals: dict[int, list[str]] = {}
    for issue in issues:
        position, column, action = issue.get("row"), issue.get("column"), issue.get("action")
        if position is None:
            continue
        if action == REMOVE_ROW:
            removals.setdefault(position, []).append(issue["message"])
        elif action in (SET_NULL, SET_UNKNOWN) and column in result.columns:
            value = None if action == SET_NULL else UNKNOWN_AVAILABILITY
            result.iloc[position, result.columns.get_loc(column)] = value
    if not removals:
        return result, {}
    keep = np.ones(len(result), dtype=bool)
    keep[list(removals)] = False
    removed = {result.index[position]: messages for position, messages in removals.items()}
    return result.iloc[keep], removed


def _value_counts(series: pd.Series, missing_label: str | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in series.tolist():
        if is_null_like(value):
            if missing_label is not None:
                counts[missing_label] += 1
        else:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _reason_category(reason: str) -> str:
    """Rejection reason without per-record details (quoted values, explanations) for aggregation."""
    return "; ".join(part.split(" '")[0].split(" (")[0] for part in reason.split("; "))


def _parse_failures(df: pd.DataFrame, raw_column: str, parsed_column: str) -> list[dict[str, Any]]:
    """Rows whose raw value was present but could not be parsed (they stay null, never 0)."""
    failures = []
    for source, url, raw, parsed in zip(df["source"], df["product_url"], df[raw_column], df[parsed_column]):
        if not is_null_like(raw) and is_null_like(parsed):
            failures.append({"source": source, "product_url": url, raw_column: raw})
    return failures


def clean_product_data(records: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None) -> CleaningResult:
    """Clean, de-duplicate and validate raw scraped records into the contract Stage 2 DataFrame.

    Steps: normalise text and null-like values -> reject records missing identifiers -> parse
    price/currency/rating/availability/counts/timestamps -> drop duplicates within each source
    -> build product_id and data_gaps -> validate (hard failures removed, soft failures nulled).
    Accepts the list of raw records (or the whole ``raw_products.json`` payload); the input is
    not modified.
    """
    if isinstance(records, Mapping):  # the full {"generated_at", "scrape_report", "records"} payload
        records = records.get("records")
    raw_records = list(records) if records is not None else []
    rejected: list[dict[str, Any]] = []
    positions: list[int] = []
    for position, record in enumerate(raw_records):
        if isinstance(record, Mapping):
            positions.append(position)
        else:
            rejected.append(_rejection(record, "malformed record: expected a JSON object"))
            logger.warning("Rejected raw record #%d: not a JSON object (%s)", position, type(record).__name__)
    malformed_count = len(rejected)

    frame = _build_raw_frame([raw_records[position] for position in positions], positions)
    frame = _normalise_text_fields(frame)
    frame, identifier_failures = _split_missing_identifiers(frame)
    rejected.extend(_rejection(raw_records[position], reason) for position, reason in identifier_failures.items())

    frame = _parse_fields(frame)
    frame, url_duplicates = _drop_duplicate_records(frame, ["source", "product_url"], RULE_SAME_URL)
    frame, content_duplicates = _drop_content_duplicates(frame)

    clean = _to_clean_schema(frame)
    issues = validate_clean_dataset(clean)
    clean, removed_by_validation = _apply_validation(clean, issues)
    for position, messages in removed_by_validation.items():
        rejected.append(_rejection(raw_records[int(position)], "validation: " + "; ".join(messages)))
    clean = clean.assign(data_gaps=pd.Series(_data_gaps(clean), index=clean.index, dtype="string"))
    clean = clean.reset_index(drop=True)

    report = {
        "raw_records": len(raw_records),
        "rejected_malformed_records": malformed_count,
        "rejected_missing_identifiers": len(identifier_failures),
        "duplicates_removed_same_url": len(url_duplicates),
        "duplicates_removed_same_content": len(content_duplicates),
        "removed_by_validation": len(removed_by_validation),
        "final_records": len(clean),
        "records_per_source": _value_counts(clean["source"]),
        "missing_values": {column: int(count) for column, count in clean.isna().sum().items()},
        "price_parse_failures": _parse_failures(clean, "price_raw", "price"),
        "rating_parse_failures": _parse_failures(clean, "rating_raw", "rating"),
        "currency_counts": _value_counts(clean["currency"], missing_label="missing"),
        "availability_counts": {
            status: int((clean["availability"] == status).sum()) for status in config.AVAILABILITY_VALUES
        },
        "validation_issues": issues,
        "duplicate_details": url_duplicates + content_duplicates,
        "rejection_reasons": dict(Counter(_reason_category(item["reason"]) for item in rejected)),
        "policies": dict(CLEANING_POLICIES),
    }
    report = _to_json_safe(report)
    logger.info(
        "Cleaning complete: %d raw -> %d clean (%d rejected, %d same-URL and %d same-content duplicates removed, "
        "%d removed by validation, %d price and %d rating parse failures)",
        report["raw_records"],
        report["final_records"],
        malformed_count + len(identifier_failures),
        len(url_duplicates),
        len(content_duplicates),
        len(removed_by_validation),
        len(report["price_parse_failures"]),
        len(report["rating_parse_failures"]),
    )
    return CleaningResult(dataframe=clean, report=report, rejected_records=rejected)


# ------------------------------------------------------------------------ validation
HARD, SOFT = "hard", "soft"
REMOVE_ROW, SET_NULL, SET_UNKNOWN, NO_ACTION = "remove_row", "set_null", "set_unknown", "none"
_CURRENCY_CODE_RE = re.compile(r"[A-Z]{3}")


@dataclass(frozen=True)
class _Rule:
    """A validation rule; every row it flags is reported with the same check, action and message."""

    check: str
    column: str | None
    severity: str
    action: str
    message: str


def _hard(check: str, column: str, message: str) -> _Rule:
    return _Rule(check, column, HARD, REMOVE_ROW, message)


def _soft(check: str, column: str, message: str, action: str = SET_NULL) -> _Rule:
    return _Rule(check, column, SOFT, action, message)


_NAME_RULE = _hard("product_name_present", "product_name", "product name is empty")
_SOURCE_RULE = _hard("source_present", "source", "source is missing")
_URL_RULE = _hard("product_url_valid", "product_url", "product URL is missing or not an absolute http(s) URL")
_DUPLICATE_URL_RULE = _hard("duplicate_source_url", "product_url", "duplicate (source, product_url) record")
_PRODUCT_ID_RULE = _hard("product_id_present", "product_id", "product_id is missing")
_DUPLICATE_ID_RULE = _hard("duplicate_product_id", "product_id", "duplicate product_id")
_AVAILABILITY_RULE = _soft(
    "availability_approved",
    "availability",
    f"availability must be one of {', '.join(config.AVAILABILITY_VALUES)}",
    SET_UNKNOWN,
)
_CURRENCY_RULE = _soft("currency_code", "currency", "currency must be a 3-letter ISO-4217 code or null")
_TIMESTAMP_MISSING_RULE = _soft("scraped_at_present", "scraped_at", "collection timestamp is missing", NO_ACTION)
_TIMESTAMP_NAIVE_RULE = _soft("scraped_at_timezone", "scraped_at", "scraped_at must be timezone-aware")
_MIN, _MAX = config.RATING_MIN, config.RATING_MAX
# (rule, lower bound, upper bound or None, whole numbers only) for numeric columns.
_NUMERIC_RULES: tuple[tuple[_Rule, float, float | None, bool], ...] = (
    (_soft("price_non_negative", "price", "price must be a number >= 0"), 0, None, False),
    (_soft("original_price_non_negative", "original_price", "original_price must be a number >= 0"), 0, None, False),
    (_soft("rating_in_range", "rating", f"rating must be an integer {_MIN}-{_MAX}"), _MIN, _MAX, True),
    (_soft("rating_unrounded_in_range", "rating_unrounded", f"rating_unrounded must be 0-{_MAX}"), 0, _MAX, False),
    (_soft("review_count_valid", "review_count", "review_count must be a non-negative integer"), 0, None, True),
    (_soft("stock_quantity_valid", "stock_quantity", "stock_quantity must be a non-negative integer"), 0, None, True),
)


def _issue(df: pd.DataFrame, position: int | None, rule: _Rule) -> dict[str, Any]:
    row = df.iloc[position] if position is not None else None

    def field_value(name: str | None) -> Any:
        return _to_json_safe(row[name]) if row is not None and name in df.columns else None

    return {
        "check": rule.check,
        "severity": rule.severity,
        "action": rule.action,
        "row": position,
        "product_id": field_value("product_id"),
        "source": field_value("source"),
        "product_url": field_value("product_url"),
        "column": rule.column,
        "value": field_value(rule.column),
        "message": rule.message,
    }


def _column_mask(df: pd.DataFrame, column: str, predicate: Callable[[Any], bool]) -> np.ndarray:
    """Element-wise boolean mask; all False when the column is absent (reported separately)."""
    if column not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return np.array([bool(predicate(value)) for value in df[column].tolist()], dtype=bool)


def _duplicate_mask(df: pd.DataFrame, columns: list[str], eligible: np.ndarray) -> np.ndarray:
    """Second and later occurrences of ``columns`` among the ``eligible`` rows."""
    mask = np.zeros(len(df), dtype=bool)
    mask[eligible] = df.loc[eligible, columns].duplicated(keep="first").to_numpy()
    return mask


def _invalid_numbers(series: pd.Series, low: float, high: float | None, whole: bool) -> np.ndarray:
    """Present values that are non-numeric, infinite, outside ``[low, high]`` or (if ``whole``) fractional."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    present = ~np.isnan(values)
    non_numeric = series.notna().to_numpy(dtype=bool) & ~present
    with np.errstate(invalid="ignore"):
        bad = np.isinf(values) | (values < low)
        if high is not None:
            bad |= values > high
        if whole:
            bad |= values != np.round(values)
    return non_numeric | (present & bad)


def _is_malformed_currency(code: Any) -> bool:
    return not is_null_like(code) and not (isinstance(code, str) and _CURRENCY_CODE_RE.fullmatch(code))


def _is_aware_timestamp(value: Any) -> bool:
    return isinstance(value, (pd.Timestamp, dt.datetime)) and value.tzinfo is not None


def _identifier_rules(df: pd.DataFrame) -> list[tuple[_Rule, np.ndarray]]:
    """Hard failures: the record cannot be identified or traced, so the row is removed."""
    name_blank = _column_mask(df, "product_name", is_null_like)
    source_blank = _column_mask(df, "source", is_null_like)
    url_invalid = _column_mask(df, "product_url", lambda url: canonicalize_url(url) is None)
    findings = [(_NAME_RULE, name_blank), (_SOURCE_RULE, source_blank), (_URL_RULE, url_invalid)]
    if {"source", "product_url"} <= set(df.columns):
        identified = ~(name_blank | source_blank | url_invalid)
        findings.append((_DUPLICATE_URL_RULE, _duplicate_mask(df, ["source", "product_url"], identified)))
    if "product_id" in df.columns:
        id_blank = _column_mask(df, "product_id", is_null_like)
        findings.append((_PRODUCT_ID_RULE, id_blank))
        findings.append((_DUPLICATE_ID_RULE, _duplicate_mask(df, ["product_id"], ~id_blank)))
    return findings


def _value_rules(df: pd.DataFrame) -> list[tuple[_Rule, np.ndarray]]:
    """Soft failures: the record is kept and only the offending value is nulled (or set to unknown)."""
    findings = [
        (rule, _invalid_numbers(df[rule.column], low, high, whole))
        for rule, low, high, whole in _NUMERIC_RULES
        if rule.column in df.columns
    ]
    if "availability" in df.columns:
        approved = df["availability"].isin(config.AVAILABILITY_VALUES).fillna(False).to_numpy(dtype=bool)
        findings.append((_AVAILABILITY_RULE, ~approved))
    findings.append((_CURRENCY_RULE, _column_mask(df, "currency", _is_malformed_currency)))
    if "scraped_at" in df.columns:
        missing = _column_mask(df, "scraped_at", is_null_like)
        aware = _column_mask(df, "scraped_at", _is_aware_timestamp)
        findings.append((_TIMESTAMP_MISSING_RULE, missing))
        findings.append((_TIMESTAMP_NAIVE_RULE, ~missing & ~aware))
    return findings


def _json_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Confirm every record survives ``json.dumps(..., allow_nan=False)``; pinpoint offending fields."""
    records = dataframe_to_records(df)
    try:
        json.dumps(records, allow_nan=False)
        return []
    except (TypeError, ValueError):
        pass
    issues = []
    for position, record in enumerate(records):
        for column, value in record.items():
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                hard = column in IDENTIFIER_COLUMNS
                rule = _Rule(
                    "json_serialisable", column, HARD if hard else SOFT, REMOVE_ROW if hard else SET_NULL,
                    f"{column} is not JSON-serialisable",
                )
                issues.append(_issue(df, position, rule))
    return issues


def validate_clean_dataset(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check a cleaned DataFrame before AI processing; return JSON-safe issue dicts.

    Each issue names its ``check``, ``severity`` (``"hard"``/``"soft"``), the ``action`` the cleaner
    applies (``"remove_row"``, ``"set_null"``, ``"set_unknown"`` or ``"none"``), the 0-based ``row``
    position, the product identifiers, the offending ``column``/``value`` and a message. Hard
    failures: empty name, missing source, missing/invalid URL, duplicate (source, product_url) or
    product_id. Soft failures: negative/non-numeric prices, ratings outside 1-5, invalid counts,
    unapproved availability, malformed currency, missing or naive timestamps, non-JSON values.
    An empty list means the dataset passed. ``df`` is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_clean_dataset expects a pandas DataFrame, got {type(df).__name__}")
    issues: list[dict[str, Any]] = []
    missing_columns = [column for column in config.REQUIRED_CLEAN_COLUMNS if column not in df.columns]
    if missing_columns:
        message = f"missing required column(s): {', '.join(missing_columns)}"
        issues.append(_issue(df, None, _Rule("required_columns", None, HARD, NO_ACTION, message)))
    if not df.empty:
        for rule, mask in (*_identifier_rules(df), *_value_rules(df)):
            issues.extend(_issue(df, int(position), rule) for position in np.flatnonzero(mask))
        issues.extend(_json_issues(df))
    if issues:
        logger.warning("Validation found %d issue(s) in the cleaned dataset", len(issues))
    return issues


# ------------------------------------------------------------------------------- I/O
def load_raw_records(path: str | os.PathLike[str] = config.RAW_PRODUCTS_FILE) -> list[dict[str, Any]]:
    """Load raw records from ``raw_products.json`` (``{"records": [...]}`` or a bare list)."""
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file_path} is not valid JSON: {exc}") from exc
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{file_path} does not contain a list of raw product records")
    return records


def save_cleaned_csv(df: pd.DataFrame, path: str | os.PathLike[str] = config.CLEANED_PRODUCTS_FILE) -> Path:
    """Write the cleaned dataset as Excel-friendly UTF-8 CSV (BOM, ISO-8601 ``Z`` timestamps, blank nulls)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = _map_values(output[column], format_utc_timestamp)
    temporary = file_path.with_name(file_path.name + ".tmp")
    try:
        output.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, file_path)  # atomic: never leave a half-written CSV behind
    except OSError:
        logger.exception("Could not write cleaned CSV to %s", file_path)
        temporary.unlink(missing_ok=True)
        raise
    logger.info("Saved %d cleaned records to %s", len(df), file_path)
    return file_path
