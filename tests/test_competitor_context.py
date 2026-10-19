"""Tests for the deterministic competitor price benchmark (contract Stage 2b)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
from competitor_context import (
    BASIS_CATEGORY,
    BASIS_NONE,
    BASIS_SOURCE,
    build_all_contexts,
    build_competitor_context,
    jaccard_similarity,
    tokenize,
)
from data_cleaner import clean_product_data

CONTEXT_KEYS = set(
    "comparison_basis category currency peer_count price_stats price_position similar_products note".split()
)
PEER_KEYS = set("product_id product_name source price currency rating availability similarity".split())


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Minimal cleaned-schema frame with the dtypes produced by data_cleaner."""
    defaults = {
        "product_name": "Item",
        "price": np.nan,
        "currency": None,
        "rating": None,
        "availability": "unknown",
        "description": None,
        "source": "shop_a",
        "category": None,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    return frame.astype(
        {
            "product_id": "string",
            "product_name": "string",
            "price": "float64",
            "currency": "string",
            "rating": "Int64",
            "availability": "string",
            "description": "string",
            "source": "string",
            "category": "string",
        }
    )


def product(product_id: str, price: float, **fields: object) -> dict:
    return {"product_id": product_id, "price": price, "currency": "USD", "category": "Laptops", **fields}


# ----------------------------------------------------------------------- price maths
def test_price_stats_position_and_percentile_math() -> None:
    df = make_frame(
        [
            product("P1", 10.0),
            product("P2", 20.0, rating=4),
            product("P3", 20.0),
            product("P4", 30.0),
            product("P5", 40.0),
        ]
    )
    context = build_competitor_context(df, "P2")
    assert set(context) == CONTEXT_KEYS
    assert context["comparison_basis"] == BASIS_CATEGORY
    assert (context["category"], context["currency"], context["peer_count"]) == ("Laptops", "USD", 4)
    assert context["price_stats"] == {"min": 10.0, "median": 25.0, "mean": 25.0, "max": 40.0}
    assert context["price_position"] == {
        "difference_vs_median": -5.0,
        "difference_vs_median_pct": -20.0,
        "percentile_rank": 37.5,  # 1 cheaper peer + 0.5 x 1 tied peer, out of 4
        "cheaper_peer_count": 1,
        "pricier_peer_count": 2,
    }


def test_stats_are_rounded_to_two_decimals() -> None:
    df = make_frame([product("A", 10.0), product("B", 10.01), product("C", 10.02), product("D", 13.333)])
    context = build_competitor_context(df, "D")
    assert context["price_stats"] == {"min": 10.0, "median": 10.01, "mean": 10.01, "max": 10.02}
    assert context["price_position"]["difference_vs_median"] == 3.32  # 13.333 - 10.01 = 3.323
    assert context["price_position"]["difference_vs_median_pct"] == 33.2
    assert context["price_position"]["percentile_rank"] == 100.0


def test_free_peers_give_no_percentage_difference() -> None:
    df = make_frame([product("A", 0.0), product("B", 0.0), product("C", 0.0), product("D", 5.0)])
    position = build_competitor_context(df, "D")["price_position"]
    assert position["difference_vs_median"] == 5.0
    assert position["difference_vs_median_pct"] is None


# ------------------------------------------------------------------- peer selection
def test_prices_in_different_currencies_are_never_compared() -> None:
    df = make_frame(
        [
            product("U1", 300.0, source="shop_a"),
            product("U2", 400.0, source="shop_a"),
            product("U3", 500.0, source="shop_b"),
            product("U4", 600.0, source="shop_b"),
            product("G1", 350.0, currency="GBP", source="shop_c"),
        ]
    )
    contexts = build_all_contexts(df)
    usd = contexts["U1"]
    assert usd["comparison_basis"] == BASIS_CATEGORY and usd["peer_count"] == 3
    assert {peer["currency"] for peer in usd["similar_products"]} == {"USD"}
    assert "G1" not in {peer["product_id"] for peer in usd["similar_products"]}
    gbp = contexts["G1"]
    assert gbp["comparison_basis"] == BASIS_NONE
    assert gbp["peer_count"] == 0 and gbp["price_stats"] is None and gbp["price_position"] is None
    assert gbp["similar_products"] == []
    assert "GBP" in gbp["note"] and "currenc" in gbp["note"]


def test_falls_back_to_same_source_when_category_group_is_thin() -> None:
    df = make_frame(
        [
            product("H1", 15.0, category="Household", source="web_scraping_dev"),
            product("H2", 25.0, category="Household", source="web_scraping_dev"),
            product("C1", 5.0, category="Consumables", source="web_scraping_dev"),
            product("C2", 9.0, category="Consumables", source="web_scraping_dev"),
            product("A1", 30.0, category="Apparel", source="web_scraping_dev"),
            product("X1", 99.0, category="Household", source="other_shop"),
            product("G1", 12.0, category="Household", source="web_scraping_dev", currency="GBP"),
        ]
    )
    context = build_competitor_context(df, "H1")
    assert context["comparison_basis"] == BASIS_SOURCE
    assert context["peer_count"] == 4  # H2, C1, C2, A1: same source and currency, any category
    assert {peer["product_id"] for peer in context["similar_products"]} == {"H2", "C1", "C2", "A1"}
    assert context["price_stats"] == {"min": 5.0, "median": 17.0, "mean": 17.25, "max": 30.0}
    assert context["note"].startswith("Broader benchmark")


def test_small_category_group_is_used_when_source_group_is_not_larger() -> None:
    df = make_frame(
        [
            product("M1", 10.0, category="Mystery", currency="GBP", source="books_toscrape"),
            product("M2", 20.0, category="Mystery", currency="GBP", source="books_toscrape"),
            product("M3", 30.0, category="Mystery", currency="GBP", source="books_toscrape"),
        ]
    )
    context = build_competitor_context(df, "M1")
    assert context["comparison_basis"] == BASIS_CATEGORY
    assert context["peer_count"] == 2
    assert "indicative" in context["note"]


def test_category_matching_is_case_insensitive() -> None:
    df = make_frame(
        [
            product("A", 10.0, category="Mystery", source="s1"),
            product("B", 11.0, category="mystery ", source="s2"),
            product("C", 12.0, category="MYSTERY", source="s3"),
            product("D", 13.0, category="Mystery", source="s4"),
        ]
    )
    context = build_competitor_context(df, "A")
    assert context["comparison_basis"] == BASIS_CATEGORY and context["peer_count"] == 3


def test_peers_without_a_valid_price_are_excluded() -> None:
    df = make_frame(
        [
            product("T", 50.0),
            product("NO_PRICE", np.nan),
            product("NEGATIVE", -1.0),
            product("INF", np.inf),
            product("OK1", 40.0),
            product("OK2", 60.0),
            product("OK3", 70.0),
        ]
    )
    context = build_competitor_context(df, "T")
    assert context["peer_count"] == 3
    assert {peer["product_id"] for peer in context["similar_products"]} == {"OK1", "OK2", "OK3"}


@pytest.mark.parametrize(
    ("rows", "note_fragment"),
    [
        ([product("T", np.nan), product("P", 10.0)], "no valid price"),
        ([product("T", 10.0, currency=None), product("P", 10.0)], "currency is unknown"),
        ([product("T", 10.0), product("P", 10.0, currency="EUR")], "no other product"),
    ],
)
def test_none_basis_explains_why(rows: list[dict], note_fragment: str) -> None:
    context = build_competitor_context(make_frame(rows), "T")
    assert context["comparison_basis"] == BASIS_NONE
    assert context["peer_count"] == 0
    assert context["price_stats"] is None and context["price_position"] is None
    assert context["similar_products"] == []
    assert note_fragment in context["note"]


# ---------------------------------------------------------------- similar products
def test_similar_products_are_ranked_and_limited() -> None:
    def food(product_id: str, price: float, name: str, description: str | None, source: str = "shop_b", **extra):
        fields = {"product_name": name, "description": description, "source": source, "category": "Food"}
        return product(product_id, price, **fields, **extra)

    dark = ("Dark Chocolate Bar", "Rich dark chocolate with orange")
    df = make_frame(
        [
            food("T", 5.0, *dark, source="shop_a"),
            food("SAME_SRC", 5.5, *dark, source="shop_a"),
            food("OTHER_SRC", 6.0, *dark),
            food("MID", 4.0, "Milk Chocolate Bar", "Creamy milk chocolate", rating=4),
            food("LOW", 3.0, "Orange Juice", "Fresh orange juice"),
            food("ZERO", 30.0, "Laptop Stand", None),
        ]
    )
    similar = build_competitor_context(df, "T", max_peers=3)["similar_products"]
    assert [peer["product_id"] for peer in similar] == ["OTHER_SRC", "SAME_SRC", "MID"]  # tie -> cross-source first
    assert [peer["similarity"] for peer in similar] == [1.0, 1.0, 0.286]
    assert set(similar[2]) == PEER_KEYS
    assert similar[2]["rating"] == 4 and similar[0]["rating"] is None
    assert similar[2]["availability"] == "unknown"
    full = build_competitor_context(df, "T", max_peers=10)["similar_products"]
    assert [peer["product_id"] for peer in full][-2:] == ["LOW", "ZERO"]
    assert build_competitor_context(df, "T", max_peers=0)["similar_products"] == []


def test_default_peer_limit_comes_from_config() -> None:
    rows = [product(f"P{i}", float(10 + i)) for i in range(config.PROMPT_MAX_PEERS + 4)]
    context = build_competitor_context(make_frame(rows), "P0")
    assert context["peer_count"] == config.PROMPT_MAX_PEERS + 3
    assert len(context["similar_products"]) == config.PROMPT_MAX_PEERS


def test_tokenize_and_jaccard() -> None:
    assert tokenize("The Box of Chocolate Candy!") == frozenset({"box", "chocolate", "candy"})
    assert tokenize(None) == frozenset()
    assert jaccard_similarity(frozenset({"a1", "b2"}), frozenset({"b2", "c3"})) == pytest.approx(1 / 3)
    assert jaccard_similarity(frozenset(), frozenset()) == 0.0


# ------------------------------------------------------------------ contracts / errors
def test_all_contexts_keyed_by_product_id_and_json_serialisable() -> None:
    df = make_frame(
        [
            product("P1", 10.0, rating=3),
            product("P2", 20.0, rating=5),
            product("P3", 30.0),
            product("P4", np.nan),
        ]
    )
    contexts = build_all_contexts(df)
    assert list(contexts) == ["P1", "P2", "P3", "P4"]
    json.dumps(contexts, allow_nan=False)
    for context in contexts.values():
        assert set(context) == CONTEXT_KEYS
        assert isinstance(context["peer_count"], int) and context["note"]
        for peer in context["similar_products"]:
            assert isinstance(peer["price"], float)
            assert peer["rating"] is None or type(peer["rating"]) is int


def test_errors_for_unknown_product_bad_columns_and_duplicate_ids() -> None:
    df = make_frame([product("P1", 10.0)])
    with pytest.raises(KeyError):
        build_competitor_context(df, "missing")
    with pytest.raises(ValueError, match="price"):
        build_all_contexts(df.drop(columns=["price"]))
    with pytest.raises(ValueError, match="unique"):
        build_all_contexts(pd.concat([df, df], ignore_index=True))
    with pytest.raises(ValueError, match="max_peers"):
        build_all_contexts(df, max_peers=-1)
    assert build_all_contexts(df.iloc[0:0]) == {}


def test_contexts_from_cleaned_scrape_never_mix_currencies() -> None:
    books = "https://books.toscrape.com/catalogue/"
    laptops = "https://webscraper.io/test-sites/e-commerce/static/product/"
    store = "https://web-scraping.dev/product/"

    def raw(name: str, price: str, source: str, url: str, category: str, currency: str | None = None) -> dict:
        # Deliberately partial records: the cleaner must add the missing Stage 1 keys itself.
        return {
            "product_name": name,
            "price_raw": price,
            "currency_raw": currency,
            "rating_raw": "4",
            "availability_raw": "In stock",
            "description": f"{name} description",
            "source": source,
            "product_url": url,
            "scraped_at": "2026-10-17T12:00:00Z",
            "category": category,
        }

    records = [
        raw("Sharp Objects", "£47.82", "books_toscrape", f"{books}sharp-objects_997/index.html", "Mystery"),
        raw("In a Dark, Dark Wood", "£19.63", "books_toscrape", f"{books}in-a-dark-dark-wood_963/", "Mystery"),
        raw("The Past Never Ends", "£56.50", "books_toscrape", f"{books}the-past-never-ends_942/", "Mystery"),
        raw("Packard 255 G2", "$416.99", "webscraper_io", f"{laptops}31", "Laptops", "USD"),
        raw("Aspire E1-510", "$306.99", "webscraper_io", f"{laptops}32", "Laptops", "USD"),
        raw("Box of Chocolate Candy", "$9.99", "web_scraping_dev", f"{store}1", "Consumables", "USD"),
        raw("Dark Red Energy Potion", "$4.99", "web_scraping_dev", f"{store}2", "Consumables", "USD"),
        raw("Teal Energy Potion", "$4.99", "web_scraping_dev", f"{store}3", "Consumables", "USD"),
        raw("Mystery Box", "Call for price", "web_scraping_dev", f"{store}4", "Household", "USD"),
    ]
    df = clean_product_data(records).dataframe
    contexts = build_all_contexts(df)
    assert set(contexts) == set(df["product_id"])
    for product_id, context in contexts.items():
        own = df.loc[df["product_id"] == product_id].iloc[0]
        assert context["currency"] == (None if pd.isna(own["currency"]) else own["currency"])
        assert {peer["currency"] for peer in context["similar_products"]} <= {context["currency"]}
    unpriced = df.loc[df["product_name"] == "Mystery Box", "product_id"].iloc[0]
    assert contexts[unpriced]["comparison_basis"] == BASIS_NONE
    laptop = df.loc[df["product_name"] == "Packard 255 G2", "product_id"].iloc[0]
    assert contexts[laptop]["comparison_basis"] == BASIS_CATEGORY  # 1 category peer = 1 source peer
    json.dumps(contexts, allow_nan=False)
