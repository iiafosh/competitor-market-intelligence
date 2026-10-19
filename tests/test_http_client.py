"""Offline tests for the polite HTTP client: robots.txt, retries, rate limiting, encoding, caching."""

from __future__ import annotations

import codecs
import json
import logging
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any

import pytest
import requests

import config
from http_client import (
    FetchError,
    PermanentHTTPError,
    PoliteHttpClient,
    RetryExhaustedError,
    RobotsDisallowedError,
    parse_retry_after,
    parse_robots_txt,
)

# The real robots.txt files of the two sources that publish one (fetched 2026-10-18).
WEBSCRAPER_IO_ROBOTS = """User-agent: *
Disallow: /test-sites/product/
Disallow: /test-sites/pagination/
Disallow: /test-sites/pagination*?page=
Disallow: /test-sites/scroll/
Disallow: /test-sites/load-more/

Sitemap: https://webscraper.io/sitemap.xml
"""

WEB_SCRAPING_DEV_ROBOTS = """User-agent: *
Disallow: /robots-disallowed
Crawl-delay: 2

User-agent: Googlebot
Disallow: /api/
Disallow: /robots-disallowed
Crawl-delay: 2

User-agent: Bingbot
Disallow: /api/
Disallow: /robots-disallowed
Crawl-delay: 2

Sitemap: https://web-scraping.dev/sitemap.xml
Sitemap: https://web-scraping.dev/sitemap-index.xml
"""

SHOP = "https://shop.test"


# --------------------------------------------------------------------------- test doubles
def make_response(
    url: str, status: int = 200, body: bytes = b"", headers: dict[str, str] | None = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.headers.update(headers or {"Content-Type": "text/html; charset=utf-8"})
    response.url = url
    response.reason = HTTPStatus(status).phrase
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    return response


class FakeSession:
    """Serves queued responses/exceptions per URL (the last one repeats); unknown URLs get 404."""

    def __init__(self, routes: dict[str, Any] | None = None) -> None:
        self.routes = {
            url: list(items) if isinstance(items, list) else [items] for url, items in (routes or {}).items()
        }
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, str] | None = None, timeout: Any = None) -> requests.Response:
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        queue = self.routes.get(url)
        if not queue:
            return make_response(url, 404)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def urls(self) -> list[str]:
        return [call["url"] for call in self.calls]

    def close(self) -> None:
        pass


class FakeClock:
    """Monotonic clock that only moves when the client sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


def make_client(session: FakeSession, **overrides: Any) -> tuple[PoliteHttpClient, FakeClock]:
    clock = FakeClock()
    settings: dict[str, Any] = {
        "default_delay": 1.5,
        "max_retries": 3,
        "backoff_base": 2.0,
        "backoff_max": 30.0,
        "max_retry_after": 60.0,
        "sleep": clock.sleep,
        "clock": clock,
        "jitter": lambda: 0.0,
    }
    settings.update(overrides)
    return PoliteHttpClient(session, **settings), clock


def html(body: str) -> bytes:
    return f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>".encode()


def ok(url: str, body: str = "ok") -> requests.Response:
    return make_response(url, 200, html(body))


def plain_text(url: str, text: str) -> requests.Response:
    return make_response(url, 200, text.encode("utf-8"), {"Content-Type": "text/plain"})


# --------------------------------------------------------------------------- robots.txt parsing
def test_webscraper_io_rules_including_wildcard() -> None:
    policy = parse_robots_txt(WEBSCRAPER_IO_ROBOTS, "MarketScopeCapstone")

    assert policy.matched_group == "*"
    assert policy.crawl_delay is None
    assert policy.can_fetch("https://webscraper.io/test-sites/e-commerce/static/computers/laptops")
    assert policy.can_fetch("https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page=2")
    assert policy.can_fetch("https://webscraper.io/test-sites/e-commerce/static/product/31")
    assert not policy.can_fetch("https://webscraper.io/test-sites/product/31")
    assert not policy.can_fetch("https://webscraper.io/test-sites/pagination/page-1")
    # "/test-sites/pagination*?page=" uses a wildcard that urllib.robotparser does not understand
    assert not policy.can_fetch("https://webscraper.io/test-sites/pagination-ajax?page=2")
    assert policy.can_fetch("https://webscraper.io/test-sites/pagination-ajax")


def test_web_scraping_dev_crawl_delay_and_group_selection() -> None:
    ours = parse_robots_txt(WEB_SCRAPING_DEV_ROBOTS, "MarketScopeCapstone")
    googlebot = parse_robots_txt(WEB_SCRAPING_DEV_ROBOTS, "Googlebot")

    assert ours.matched_group == "*"
    assert ours.crawl_delay == 2.0
    assert not ours.can_fetch("https://web-scraping.dev/robots-disallowed")
    assert ours.can_fetch("https://web-scraping.dev/products?category=apparel&page=2")
    assert ours.can_fetch("https://web-scraping.dev/api/reviews")  # /api/ is only closed to Googlebot/Bingbot
    assert googlebot.matched_group == "Googlebot"
    assert not googlebot.can_fetch("https://web-scraping.dev/api/reviews?product_id=1")


def test_specific_group_wins_case_insensitively_and_groups_merge() -> None:
    text = (
        "User-agent: *\nDisallow: /\n\n"
        "User-agent: marketscopecapstone/2.0\nAllow: /\nCrawl-delay: 5\n\n"
        "User-agent: MARKETSCOPECAPSTONE\nDisallow: /private\n"
    )
    policy = parse_robots_txt(text, "MarketScopeCapstone")

    assert policy.matched_group == "MarketScopeCapstone"
    assert policy.crawl_delay == 5.0
    assert policy.can_fetch("https://x.test/catalogue")
    assert not policy.can_fetch("https://x.test/private/page")


def test_longest_match_allow_wins_ties_and_end_anchor() -> None:
    text = (
        "User-agent: *\n"
        "Disallow: /shop\n"
        "Allow: /shop/public\n"
        "Allow: /tie\n"
        "Disallow: /tie\n"
        "Disallow: /*.pdf$\n"
        "Disallow: /search*q=\n"
    )
    policy = parse_robots_txt(text, "MarketScopeCapstone")

    assert not policy.can_fetch("https://x.test/shop/private")
    assert policy.can_fetch("https://x.test/shop/public/item")
    assert policy.can_fetch("https://x.test/tie/page")
    assert not policy.can_fetch("https://x.test/files/report.pdf")
    assert policy.can_fetch("https://x.test/files/report.pdf?download=1")
    assert not policy.can_fetch("https://x.test/search?page=1&q=boots")
    assert policy.can_fetch("https://x.test/search?page=1")
    assert policy.can_fetch("https://x.test/robots.txt")


def test_rules_outside_groups_and_empty_disallow_are_ignored() -> None:
    policy = parse_robots_txt("Disallow: /everything\nUser-agent: *\nDisallow:\n# comment\n", "MarketScopeCapstone")

    assert policy.can_fetch("https://x.test/everything")
    assert policy.rules == ()


def test_no_matching_group_allows_everything() -> None:
    policy = parse_robots_txt("User-agent: Googlebot\nDisallow: /\n", "MarketScopeCapstone")

    assert policy.matched_group is None
    assert policy.can_fetch("https://x.test/anything")


# --------------------------------------------------------------------------- robots.txt fetching
def test_robots_4xx_means_no_restrictions() -> None:
    robots_url = f"{SHOP}/robots.txt"
    session = FakeSession({robots_url: make_response(robots_url, 404), f"{SHOP}/p": ok(f"{SHOP}/p")})
    client, _ = make_client(session)

    assert client.get(f"{SHOP}/p").status_code == 200
    assert client.stats_as_dict()["robots_txt"][SHOP]["status"] == "unavailable"


@pytest.mark.parametrize(
    "robots_answer",
    [make_response(f"{SHOP}/robots.txt", 503), requests.exceptions.ConnectionError("connection refused")],
    ids=["http-503", "network-error"],
)
def test_unreachable_robots_disallows_everything(robots_answer: Any, caplog: pytest.LogCaptureFixture) -> None:
    session = FakeSession({f"{SHOP}/robots.txt": robots_answer, f"{SHOP}/p": ok(f"{SHOP}/p")})
    client, _ = make_client(session)

    with caplog.at_level(logging.ERROR, logger="http_client"), pytest.raises(RobotsDisallowedError) as excinfo:
        client.get(f"{SHOP}/p")

    assert session.urls() == [f"{SHOP}/robots.txt"] * 4  # retried, then gave up; the page itself was never requested
    assert "unreachable" in str(excinfo.value)
    assert client.stats.robots_blocked == 1
    assert client.stats_as_dict()["robots_txt"][SHOP]["status"] == "unreachable"
    assert any("unreachable" in record.getMessage() for record in caplog.records)


def test_disallowed_url_raises_without_sending_request() -> None:
    robots_url = "https://webscraper.io/robots.txt"
    session = FakeSession({robots_url: plain_text(robots_url, WEBSCRAPER_IO_ROBOTS)})
    client, _ = make_client(session)

    with pytest.raises(RobotsDisallowedError) as excinfo:
        client.get("https://webscraper.io/test-sites/product/31")

    assert session.urls() == [robots_url]
    assert "Disallow: /test-sites/product/" in str(excinfo.value)
    assert excinfo.value.summary.startswith("blocked by robots.txt")


def test_robots_with_byte_order_mark_is_parsed() -> None:
    robots_url = f"{SHOP}/robots.txt"
    body = codecs.BOM_UTF8 + b"User-agent: *\nDisallow: /private\n"
    session = FakeSession({robots_url: make_response(robots_url, 200, body, {"Content-Type": "text/plain"})})
    client, _ = make_client(session)

    with pytest.raises(RobotsDisallowedError):
        client.get(f"{SHOP}/private/area")

    assert client.robots_policy(SHOP).matched_group == "*"


def test_robots_is_fetched_once_per_host() -> None:
    session = FakeSession({f"{SHOP}/a": ok(f"{SHOP}/a"), f"{SHOP}/b": ok(f"{SHOP}/b")})
    client, _ = make_client(session)

    client.get(f"{SHOP}/a")
    client.get(f"{SHOP}/b")

    assert session.urls().count(f"{SHOP}/robots.txt") == 1


# --------------------------------------------------------------------------- retries
def test_retries_transient_503_with_exponential_backoff() -> None:
    page = f"{SHOP}/p"
    session = FakeSession({page: [make_response(page, 503), make_response(page, 503), ok(page)]})
    client, clock = make_client(session)

    assert client.get(page).status_code == 200
    # 1.5 s politeness gap after robots.txt, then backoff 2 s and 4 s (jitter disabled)
    assert clock.sleeps == [1.5, 2.0, 4.0]
    assert client.stats.retries == 2
    assert client.stats.failures == 0


def test_backoff_is_capped() -> None:
    page = f"{SHOP}/p"
    session = FakeSession({page: requests.exceptions.ConnectTimeout("slow")})
    client, clock = make_client(session, max_retries=4, backoff_max=5.0)

    with pytest.raises(RetryExhaustedError):
        client.get(page)

    assert clock.sleeps == [1.5, 2.0, 4.0, 5.0, 5.0]


def test_no_retry_on_404() -> None:
    page = f"{SHOP}/missing"
    session = FakeSession({page: make_response(page, 404)})
    client, _ = make_client(session)

    with pytest.raises(PermanentHTTPError) as excinfo:
        client.get(page)

    assert session.urls().count(page) == 1
    assert excinfo.value.status_code == 404
    assert str(excinfo.value) == f"GET {page} failed: HTTP 404 Not Found"
    assert client.stats.retries == 0
    assert client.stats.failures == 1


def test_timeouts_exhaust_bounded_retries() -> None:
    page = f"{SHOP}/slow"
    session = FakeSession({page: requests.exceptions.ReadTimeout("read timed out")})
    client, clock = make_client(session)

    with pytest.raises(RetryExhaustedError) as excinfo:
        client.get(page)

    assert session.urls().count(page) == 1 + 3
    assert excinfo.value.attempts == 4
    assert str(excinfo.value) == f"GET {page} failed after 4 attempts: ReadTimeout"
    assert clock.sleeps == [1.5, 2.0, 4.0, 8.0]
    assert client.stats.failures == 1


def test_retry_after_delta_seconds_is_honoured_and_host_slows_down() -> None:
    page = f"{SHOP}/p"
    session = FakeSession({page: [make_response(page, 429, headers={"Retry-After": "7"}), ok(page)]})
    client, clock = make_client(session)

    assert client.get(page).status_code == 200
    assert clock.sleeps == [1.5, 7.0]  # Retry-After (7 s) wins over the 2 s backoff
    assert client.stats_as_dict()["request_gap_seconds"][SHOP] == 3.0  # 429 doubled the 1.5 s gap


def test_retry_after_beyond_limit_gives_up_instead_of_retrying_early() -> None:
    page = f"{SHOP}/p"
    session = FakeSession({page: make_response(page, 429, headers={"Retry-After": "3600"})})
    client, _ = make_client(session)

    with pytest.raises(RetryExhaustedError) as excinfo:
        client.get(page)

    assert session.urls().count(page) == 1
    assert "after 1 attempt:" in str(excinfo.value)
    assert "3600" in str(excinfo.value)


def test_parse_retry_after_formats() -> None:
    now = datetime(2026, 10, 21, 7, 27, 30, tzinfo=timezone.utc)

    assert parse_retry_after("120") == 120.0
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT", now=now) == 30.0
    assert parse_retry_after("Wed, 21 Oct 2026 07:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after("²") is None  # a Unicode digit must not crash float()
    assert parse_retry_after(None) is None


# --------------------------------------------------------------------------- rate limiting
def test_rate_limit_is_tracked_per_host_with_the_largest_delay() -> None:
    session = FakeSession(
        {
            "https://a.test/1": make_response("https://a.test/1", 200, html("a1")),
            "https://a.test/2": make_response("https://a.test/2", 200, html("a2")),
            "https://b.test/1": make_response("https://b.test/1", 200, html("b1")),
        }
    )
    client, clock = make_client(session)

    client.get("https://a.test/1", min_delay=2.0)  # robots.txt first, then the page 2.0 s later
    client.get("https://b.test/1")  # new host: robots.txt immediately, page 1.5 s later
    client.get("https://a.test/2", min_delay=2.0)  # a.test was last hit 1.5 s ago: wait 0.5 s

    assert clock.sleeps == [2.0, 1.5, 0.5]


def test_crawl_delay_raises_the_request_gap() -> None:
    robots_url = f"{SHOP}/robots.txt"
    session = FakeSession(
        {
            robots_url: plain_text(robots_url, "User-agent: *\nCrawl-delay: 5\n"),
            f"{SHOP}/1": ok(f"{SHOP}/1"),
            f"{SHOP}/2": ok(f"{SHOP}/2"),
        }
    )
    client, clock = make_client(session)

    client.get(f"{SHOP}/1", min_delay=2.0)
    client.get(f"{SHOP}/2", min_delay=2.0)

    assert clock.sleeps == [5.0, 5.0]
    assert client.stats_as_dict()["request_gap_seconds"][SHOP] == 5.0


# --------------------------------------------------------------------------- content handling
def test_get_soup_decodes_pound_sign_when_header_has_no_charset() -> None:
    page = "https://books.test/item"
    body = '<html><head><meta charset="utf-8"></head><body><p class="price_color">£51.77</p></body></html>'
    response = make_response(page, 200, body.encode("utf-8"), {"Content-Type": "text/html"})
    session = FakeSession({page: response})
    client, _ = make_client(session)

    assert "Â£51.77" in response.text  # the pitfall: requests assumes ISO-8859-1 for text/html
    assert client.get_soup(page).select_one("p.price_color").get_text() == "£51.77"


def test_get_soup_prefers_charset_declared_in_http_header() -> None:
    page = "https://legacy.test/item"
    body = "<html><body><p>£12.00</p></body></html>".encode("cp1252")
    session = FakeSession({page: make_response(page, 200, body, {"Content-Type": "text/html; charset=windows-1252"})})
    client, _ = make_client(session)

    assert client.get_soup(page).p.get_text() == "£12.00"


def test_cache_prevents_second_download() -> None:
    page = f"{SHOP}/p"
    session = FakeSession({page: make_response(page, 200, html("ok"))})
    client, _ = make_client(session)

    client.get_soup(page)
    client.get_soup(page + "#reviews")  # same document, different fragment

    assert session.urls().count(page) == 1
    assert client.stats.cache_hits == 1


def test_every_request_identifies_itself_and_has_a_timeout() -> None:
    session = FakeSession({f"{SHOP}/p": make_response(f"{SHOP}/p", 200, html("ok"))})
    client = PoliteHttpClient(session, sleep=lambda _: None)

    client.get(f"{SHOP}/p")

    assert session.calls, "no request sent"
    for call in session.calls:
        assert call["headers"]["User-Agent"] == config.USER_AGENT
        assert call["timeout"] == config.REQUEST_TIMEOUT


def test_stats_are_json_serialisable() -> None:
    session = FakeSession({f"{SHOP}/p": make_response(f"{SHOP}/p", 200, html("ok"))})
    client, _ = make_client(session)
    client.get(f"{SHOP}/p")

    stats = json.loads(json.dumps(client.stats_as_dict()))

    assert stats["requests_sent"] == 2  # robots.txt + page
    assert stats["successful_responses"] == 1
    assert stats["bytes_received"] > 0
    assert stats["robots_txt"][SHOP]["http_status"] == 404


@pytest.mark.parametrize("url", ["ftp://shop.test/file", "/relative/path", "not a url"])
def test_rejects_non_http_urls(url: str) -> None:
    client, _ = make_client(FakeSession())

    with pytest.raises(FetchError):
        client.get(url)


def test_client_requires_a_user_agent() -> None:
    with pytest.raises(ValueError):
        PoliteHttpClient(FakeSession(), headers={"Accept": "text/html"})
