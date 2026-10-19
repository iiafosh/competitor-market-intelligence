"""Deterministic, currency-safe competitor price benchmark (contract Stage 2b).

For every cleaned product this module selects comparable competitor offers with pandas and
summarises where the product's price sits among them. The benchmark is deliberately
conservative so the AI stage only reasons over defensible evidence:

* prices are compared only within one currency (no exchange-rate conversion, ever);
* only products with a valid (present, finite, non-negative) price take part;
* the preferred peer group is "same category and currency" (category matched
  case-insensitively); with fewer than ``MIN_CATEGORY_PEERS`` such peers the benchmark falls
  back to "same source and currency" and says so in the note;
* similar products are ranked by token Jaccard similarity on name + description, preferring
  peers from other sources when similarity ties.

All returned values are plain JSON-serialisable Python types.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pandas as pd

import config
from data_cleaner import is_null_like

logger = logging.getLogger(__name__)

__all__ = [
    "BASIS_CATEGORY",
    "BASIS_NONE",
    "BASIS_SOURCE",
    "MIN_CATEGORY_PEERS",
    "build_all_contexts",
    "build_competitor_context",
    "jaccard_similarity",
    "tokenize",
]

BASIS_CATEGORY = "same category and currency"
BASIS_SOURCE = "same source and currency (broader benchmark)"
BASIS_NONE = "none"
MIN_CATEGORY_PEERS = 3  # below this the category group is too thin and the source group is tried
PRICE_TIE_TOLERANCE = 1e-9  # prices closer than this count as ties in the percentile rank

STOP_WORDS = frozenset(
    """
    an and are as at be but by for from has have in into is it its of on or our so than that the
    their this to was we were will with you your
    """.split()
)
_TOKEN_RE = re.compile(r"[^\W_]+")  # Unicode letters and digits
_REQUIRED_COLUMNS = ("product_id", "product_name", "price", "currency", "source")


def tokenize(text: str | None) -> frozenset[str]:
    """Lower-case alphanumeric tokens of length >= 2, minus a small stop-word list."""
    if not isinstance(text, str) or not text:
        return frozenset()
    normalised = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(token for token in _TOKEN_RE.findall(normalised) if len(token) >= 2 and token not in STOP_WORDS)


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """|A ∩ B| / |A ∪ B|; 0.0 when both token sets are empty."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _round(value: float, digits: int) -> float:
    """Round half-up (consistent with the rating policy) and normalise -0.0 to 0.0."""
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)) + 0.0


def _text(value: Any) -> str | None:
    if is_null_like(value):
        return None
    text = str(value).strip()
    return text or None


def _rating(value: Any) -> int | None:
    if is_null_like(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _column_values(df: pd.DataFrame, column: str) -> list[Any]:
    return df[column].tolist() if column in df.columns else [None] * len(df)


def _prepare_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised working copy with one row per product, a validity flag and token sets."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {', '.join(missing)}")

    product_ids = [_text(value) for value in df["product_id"].tolist()]
    if any(product_id is None for product_id in product_ids):
        raise ValueError("product_id must not be null")
    duplicates = sorted(product_id for product_id, count in Counter(product_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"product_id values must be unique; duplicated: {', '.join(duplicates[:5])}")

    names = [_text(value) or "" for value in df["product_name"].tolist()]
    descriptions = [_text(value) or "" for value in _column_values(df, "description")]
    categories = [_text(value) for value in _column_values(df, "category")]
    category_keys = [category.casefold() if category else None for category in categories]
    prices = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    currencies = [_text(value) for value in df["currency"].tolist()]

    catalogue = pd.DataFrame(
        {
            "product_id": pd.Series(product_ids, dtype=object),
            "product_name": pd.Series(names, dtype=object),
            "source": pd.Series([_text(value) for value in df["source"].tolist()], dtype=object),
            "currency": pd.Series([code.upper() if code else None for code in currencies], dtype=object),
            "category": pd.Series(categories, dtype=object),
            "category_key": pd.Series(category_keys, dtype=object),
            "price": pd.Series(prices, dtype="float64"),
            "rating": pd.Series([_rating(value) for value in _column_values(df, "rating")], dtype=object),
            "availability": pd.Series(
                [_text(value) or "unknown" for value in _column_values(df, "availability")], dtype=object
            ),
            "tokens": pd.Series(
                [tokenize(f"{name} {description}") for name, description in zip(names, descriptions)], dtype=object
            ),
        }
    )
    with np.errstate(invalid="ignore"):
        catalogue["valid_price"] = np.isfinite(prices) & (prices >= 0)
    return catalogue


def _empty_context(category: str | None, currency: str | None, note: str) -> dict[str, Any]:
    return {
        "comparison_basis": BASIS_NONE,
        "category": category,
        "currency": currency,
        "peer_count": 0,
        "price_stats": None,
        "price_position": None,
        "similar_products": [],
        "note": note,
    }


def _price_stats(prices: pd.Series) -> dict[str, float]:
    return {
        "min": _round(prices.min(), 2),
        "median": _round(prices.median(), 2),
        "mean": _round(prices.mean(), 2),
        "max": _round(prices.max(), 2),
    }


def _price_position(price: float, prices: pd.Series) -> dict[str, Any]:
    """Where ``price`` sits among peer prices; percentile = % cheaper peers + half of tied peers."""
    median = float(prices.median())
    cheaper = int((prices < price - PRICE_TIE_TOLERANCE).sum())
    pricier = int((prices > price + PRICE_TIE_TOLERANCE).sum())
    ties = len(prices) - cheaper - pricier
    return {
        "difference_vs_median": _round(price - median, 2),
        # Undefined when every peer is free (median 0): reported as None rather than infinity.
        "difference_vs_median_pct": _round((price - median) / median * 100, 1) if median else None,
        "percentile_rank": _round((cheaper + 0.5 * ties) / len(prices) * 100, 1),
        "cheaper_peer_count": cheaper,
        "pricier_peer_count": pricier,
    }


def _similar_products(product: pd.Series, peers: pd.DataFrame, max_peers: int) -> list[dict[str, Any]]:
    """Peers ranked by similarity, then cross-source first, then closest price, then product_id."""
    if max_peers <= 0 or peers.empty:
        return []
    tokens, source, price = product["tokens"], product["source"], float(product["price"])
    scored = [(jaccard_similarity(tokens, peer.tokens), peer) for peer in peers.itertuples(index=False)]
    scored.sort(key=lambda item: (-item[0], item[1].source == source, abs(item[1].price - price), item[1].product_id))
    return [
        {
            "product_id": peer.product_id,
            "product_name": peer.product_name,
            "source": peer.source,
            "price": float(peer.price),
            "currency": peer.currency,
            "rating": peer.rating,
            "availability": peer.availability,
            "similarity": _round(similarity, 3),
        }
        for similarity, peer in scored[:max_peers]
    ]


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _benchmark_note(basis: str, product: pd.Series, peers: pd.DataFrame, category_peer_count: int) -> str:
    count, currency, category = len(peers), product["currency"], product["category"]
    sources = ", ".join(sorted(set(peers["source"])))
    if basis == BASIS_CATEGORY:
        peers_text = _plural(count, "other product")
        lead = f"Benchmarked against {peers_text} in the '{category}' category priced in {currency}"
    else:
        reason = (
            f"only {_plural(category_peer_count, 'other priced product')} in the '{category}' category use {currency}"
            if category
            else "the product has no category"
        )
        lead = (
            f"Broader benchmark because {reason}: compared with {_plural(count, 'other product')} from the same "
            f"source ({product['source']}) priced in {currency}, which may not be direct substitutes"
        )
    parts = [f"{lead} (peer sources: {sources})."]
    if count < MIN_CATEGORY_PEERS:
        parts.append(f"With only {_plural(count, 'comparable product')}, treat the price position as indicative.")
    parts.append("Prices are compared only within one currency; no exchange-rate conversion is applied.")
    return " ".join(parts)


def _context_for(catalogue: pd.DataFrame, position: int, max_peers: int) -> dict[str, Any]:
    product = catalogue.iloc[position]
    category, currency = product["category"], product["currency"]
    if not product["valid_price"]:
        return _empty_context(
            category,
            currency,
            "No price benchmark: this product has no valid price, so it is excluded from price comparisons.",
        )
    if currency is None:
        return _empty_context(
            category,
            currency,
            "No price benchmark: the product's currency is unknown and prices are never compared across currencies.",
        )

    comparable = (
        catalogue["valid_price"]
        & catalogue["currency"].eq(currency)
        & catalogue["product_id"].ne(product["product_id"])
    )
    if product["category_key"] is None:
        category_mask = pd.Series(False, index=catalogue.index)
    else:
        category_mask = comparable & catalogue["category_key"].eq(product["category_key"])
    source_mask = comparable & catalogue["source"].eq(product["source"])
    category_count, source_count = int(category_mask.sum()), int(source_mask.sum())

    if category_count >= MIN_CATEGORY_PEERS or (category_count and category_count >= source_count):
        basis, peers = BASIS_CATEGORY, catalogue.loc[category_mask]
    elif source_count:
        basis, peers = BASIS_SOURCE, catalogue.loc[source_mask]
    else:
        return _empty_context(
            category,
            currency,
            f"No price benchmark: no other product with a valid price in {currency} was found in the same "
            "category or source, and prices are never compared across currencies.",
        )

    price = float(product["price"])
    peer_prices = peers["price"].astype("float64")
    return {
        "comparison_basis": basis,
        "category": category,
        "currency": currency,
        "peer_count": len(peers),
        "price_stats": _price_stats(peer_prices),
        "price_position": _price_position(price, peer_prices),
        "similar_products": _similar_products(product, peers, max_peers),
        "note": _benchmark_note(basis, product, peers, category_count),
    }


def _check_max_peers(max_peers: int) -> int:
    if isinstance(max_peers, bool) or not isinstance(max_peers, (int, np.integer)) or max_peers < 0:
        raise ValueError(f"max_peers must be a non-negative integer, got {max_peers!r}")
    return int(max_peers)


def build_competitor_context(
    df: pd.DataFrame, product_id: str, max_peers: int = config.PROMPT_MAX_PEERS
) -> dict[str, Any]:
    """Competitor price context for one product of the cleaned DataFrame.

    Raises ``KeyError`` for an unknown ``product_id`` and ``ValueError`` when required columns
    are missing or product ids are not unique.
    """
    max_peers = _check_max_peers(max_peers)
    catalogue = _prepare_catalogue(df)
    positions = np.flatnonzero(catalogue["product_id"].to_numpy() == product_id)
    if positions.size == 0:
        raise KeyError(f"unknown product_id: {product_id!r}")
    return _context_for(catalogue, int(positions[0]), max_peers)


def build_all_contexts(df: pd.DataFrame, max_peers: int = config.PROMPT_MAX_PEERS) -> dict[str, dict[str, Any]]:
    """Competitor price context for every product, keyed by ``product_id`` in DataFrame order."""
    max_peers = _check_max_peers(max_peers)
    catalogue = _prepare_catalogue(df)
    contexts = {
        product_id: _context_for(catalogue, position, max_peers)
        for position, product_id in enumerate(catalogue["product_id"].tolist())
    }
    bases = Counter(context["comparison_basis"] for context in contexts.values())
    logger.info("Built competitor contexts for %d products (%s)", len(contexts), dict(sorted(bases.items())))
    return contexts
