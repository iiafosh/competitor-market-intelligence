"""Polite HTTP client used by the scraping stage.

Everything that touches the network lives here, so the page parsers stay pure and testable:

* truthful identification (custom ``User-Agent``) and a timeout on every request;
* robots.txt compliance per RFC 9309 (product-token group selection, ``*``/``$`` wildcards,
  longest match, ``Crawl-delay``); a disallowed URL is never requested;
* per-host rate limiting with ``time.sleep`` (injectable, together with the clock, for tests);
* bounded retries with exponential backoff + jitter for transient failures, honouring ``Retry-After``;
* an in-memory per-URL cache so the same page is never downloaded twice in one run;
* request statistics for the execution report.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any
from collections.abc import Callable, Mapping
from urllib.parse import quote, urldefrag, urlsplit

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_EXCEPTIONS: tuple[type[requests.RequestException], ...] = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)
JITTER_MAX_SECONDS = 1.0  # random extra wait added to each backoff to avoid synchronised retries
RATE_LIMIT_SLOWDOWN_FACTOR = 2.0  # an HTTP 429 doubles the host's request gap for the rest of the run
ROBOTS_MAX_BYTES = 500 * 1024  # RFC 9309 section 2.5: parse at least the first 500 KiB

_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_AGENT_TOKEN_RE = re.compile(r"\*|[A-Za-z_-]+")
# Characters left untouched when normalising robots.txt paths: RFC 3986 reserved + unreserved + "%".
_URL_SAFE_CHARS = "/:?#[]@!$&'()*+,;=-._~%"


# --------------------------------------------------------------------------- errors
class FetchError(Exception):
    """A URL could not be retrieved. Base class of every error raised by the client."""

    def __init__(self, url: str, reason: str, status_code: int | None = None, attempts: int = 1) -> None:
        self.url = url
        self.reason = reason
        self.status_code = status_code
        self.attempts = attempts
        super().__init__(self._describe())

    @property
    def summary(self) -> str:
        """Short, URL-free description suitable for per-record warnings."""
        return self.reason

    def _describe(self) -> str:
        return f"GET {self.url} failed: {self.reason}"


class RobotsDisallowedError(FetchError):
    """robots.txt forbids the URL for our user agent; no request was sent."""

    @property
    def summary(self) -> str:
        return f"blocked by robots.txt ({self.reason})"

    def _describe(self) -> str:
        return f"GET {self.url} not sent: blocked by robots.txt ({self.reason})"


class PermanentHTTPError(FetchError):
    """The server answered with a non-retryable HTTP error such as 403 or 404."""


class RetryExhaustedError(FetchError):
    """A transient failure (timeout, connection error, 429/5xx) outlasted every permitted attempt."""

    @property
    def summary(self) -> str:
        return f"{self.reason} (after {_plural(self.attempts, 'attempt')})"

    def _describe(self) -> str:
        return f"GET {self.url} failed after {_plural(self.attempts, 'attempt')}: {self.reason}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# --------------------------------------------------------------------------- robots.txt
@dataclass(frozen=True)
class RobotsRule:
    """One ``Allow``/``Disallow`` line of the group that applies to us."""

    allow: bool
    pattern: str
    regex: re.Pattern[str] = field(repr=False, compare=False)

    @classmethod
    def build(cls, allow: bool, pattern: str) -> RobotsRule:
        normalised = _normalise_octets(pattern)
        return cls(allow=allow, pattern=normalised, regex=_compile_robots_pattern(normalised))

    def matches(self, target: str) -> bool:
        return self.regex.match(target) is not None

    def __str__(self) -> str:
        return f"{'Allow' if self.allow else 'Disallow'}: {self.pattern}"


@dataclass(frozen=True)
class RobotsPolicy:
    """The robots.txt rules that apply to our product token on one host."""

    rules: tuple[RobotsRule, ...] = ()
    crawl_delay: float | None = None
    status: str = "parsed"  # parsed | unavailable (4xx: no restrictions) | unreachable (disallow all)
    http_status: int | None = None
    matched_group: str | None = None  # user-agent of the group obeyed ("*" or our token)

    @classmethod
    def unavailable(cls, http_status: int | None = None) -> RobotsPolicy:
        """robots.txt answered 4xx: RFC 9309 section 2.3.1.3 allows crawling without restrictions."""
        return cls(status="unavailable", http_status=http_status)

    @classmethod
    def unreachable(cls, http_status: int | None = None) -> RobotsPolicy:
        """robots.txt failed with 5xx/network errors: RFC 9309 section 2.3.1.4 requires complete disallow."""
        return cls(rules=(RobotsRule.build(False, "/"),), status="unreachable", http_status=http_status)

    def decide(self, url: str) -> tuple[bool, RobotsRule | None]:
        """Return ``(allowed, deciding_rule)``: the longest matching rule wins, Allow wins ties."""
        target = _robots_target(url)
        if target == "/robots.txt":  # always implicitly allowed (RFC 9309 section 2.2.2)
            return True, None
        best: RobotsRule | None = None
        for rule in self.rules:
            if not rule.matches(target):
                continue
            if best is None or len(rule.pattern) > len(best.pattern):
                best = rule
            elif len(rule.pattern) == len(best.pattern) and rule.allow and not best.allow:
                best = rule  # equally specific: Allow wins
        return (best is None or best.allow), best

    def can_fetch(self, url: str) -> bool:
        return self.decide(url)[0]

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary for the scrape report."""
        return {
            "status": self.status,
            "http_status": self.http_status,
            "matched_group": self.matched_group,
            "rule_count": len(self.rules),
            "crawl_delay": self.crawl_delay,
        }


@dataclass
class _RobotsGroup:
    agents: list[str] = field(default_factory=list)
    rules: list[RobotsRule] = field(default_factory=list)
    crawl_delay: float | None = None


def parse_robots_txt(
    text: str,
    product_token: str = config.ROBOTS_USER_AGENT_TOKEN,
    *,
    http_status: int | None = None,
) -> RobotsPolicy:
    """Parse robots.txt and keep only the group(s) that apply to ``product_token``.

    Groups naming the token (case-insensitive) win; otherwise the ``*`` groups apply; with neither,
    everything is allowed. Several matching groups are merged, as RFC 9309 requires.
    """
    groups = _parse_robots_groups(text)
    token = product_token.lower()
    selected = [group for group in groups if token in group.agents]
    matched_group: str | None = product_token
    if not selected:
        selected = [group for group in groups if "*" in group.agents]
        matched_group = "*" if selected else None
    delays = [group.crawl_delay for group in selected if group.crawl_delay is not None]
    return RobotsPolicy(
        rules=tuple(rule for group in selected for rule in group.rules),
        crawl_delay=max(delays) if delays else None,
        status="parsed",
        http_status=http_status,
        matched_group=matched_group,
    )


def _parse_robots_groups(text: str) -> list[_RobotsGroup]:
    groups: list[_RobotsGroup] = []
    current: _RobotsGroup | None = None
    reading_agents = False  # consecutive user-agent lines share one group
    for raw_line in text.splitlines():
        key, separator, value = raw_line.split("#", 1)[0].partition(":")
        if not separator:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if current is None or not reading_agents:
                current = _RobotsGroup()
                groups.append(current)
            agent = _AGENT_TOKEN_RE.match(value)
            if agent:
                current.agents.append(agent.group(0).lower())
            reading_agents = True
        elif key in {"allow", "disallow", "crawl-delay"}:
            reading_agents = False
            if current is None:  # rules before the first user-agent line belong to no group
                continue
            if key == "crawl-delay":
                delay = _parse_crawl_delay(value)
                if delay is not None:
                    current.crawl_delay = max(delay, current.crawl_delay or 0.0)
            elif value:  # an empty Disallow/Allow matches nothing
                current.rules.append(RobotsRule.build(key == "allow", value))
    return groups


def _parse_crawl_delay(value: str) -> float | None:
    try:
        delay = float(value)
    except ValueError:
        return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def _compile_robots_pattern(pattern: str) -> re.Pattern[str]:
    """Translate a robots.txt path pattern (``*`` = any characters, trailing ``$`` = end) to a regex."""
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + (r"\Z" if anchored else ""), re.DOTALL)


def _normalise_octets(value: str) -> str:
    """Percent-encode non-ASCII characters and upper-case existing escapes so paths compare byte-wise."""
    encoded = quote(value, safe=_URL_SAFE_CHARS)
    return _PERCENT_ESCAPE_RE.sub(lambda match: match.group(0).upper(), encoded)


def _robots_target(url: str) -> str:
    """The part of a URL that robots.txt rules match against: path plus query."""
    parts = urlsplit(url)
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    return _normalise_octets(target)


# --------------------------------------------------------------------------- helpers
def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait according to a ``Retry-After`` header (delta-seconds or HTTP-date).

    Returns None when the header is absent or unparseable.
    """
    if not value or not value.strip():
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():  # str.isdigit alone accepts digits such as "²"
        return float(value)
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (moment - current).total_seconds())


def declared_charset(response: requests.Response) -> str | None:
    """Charset declared in the HTTP ``Content-Type`` header, or None when the header has none."""
    match = _CHARSET_RE.search(response.headers.get("Content-Type", ""))
    return match.group(1).lower() if match else None


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _status_text(response: requests.Response) -> str:
    return f"HTTP {response.status_code} {response.reason or ''}".strip()


@dataclass
class HttpStats:
    """Counters for the execution report."""

    requests_sent: int = 0  # every HTTP attempt, robots.txt and retries included
    successful_responses: int = 0  # responses with status < 400
    retries: int = 0  # extra attempts after a transient failure
    failures: int = 0  # fetches that ended in an error (after any retries)
    robots_blocked: int = 0  # URLs not requested because robots.txt disallows them
    cache_hits: int = 0  # URLs served from the in-memory cache
    bytes_received: int = 0


# --------------------------------------------------------------------------- client
class PoliteHttpClient:
    """A ``requests.Session`` wrapper that is polite by construction.

    The gap between two requests to the same host is
    ``max(default_delay, min_delay of the call, robots.txt Crawl-delay, HTTP 429 slowdown)``.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        headers: Mapping[str, str] = config.DEFAULT_HEADERS,
        timeout: float | tuple[float, float] = config.REQUEST_TIMEOUT,
        default_delay: float = config.REQUEST_DELAY_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        backoff_base: float = config.BACKOFF_BASE_SECONDS,
        backoff_max: float = config.BACKOFF_MAX_SECONDS,
        max_retry_after: float = config.MAX_RETRY_AFTER_SECONDS,
        robots_token: str = config.ROBOTS_USER_AGENT_TOKEN,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not headers.get("User-Agent"):
            raise ValueError("A descriptive User-Agent header is required")
        if max_retries < 0 or min(default_delay, backoff_base, backoff_max, max_retry_after) < 0:
            raise ValueError("Retry counts and delays must not be negative")
        self._owns_session = session is None
        self.session = session if session is not None else requests.Session()
        self.headers = dict(headers)
        self.timeout = timeout
        self.default_delay = default_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_retry_after = max_retry_after
        self.robots_token = robots_token
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self.stats = HttpStats()
        self._cache: dict[str, requests.Response] = {}
        self._robots: dict[str, RobotsPolicy] = {}
        self._last_request_at: dict[str, float] = {}
        self._gap_used: dict[str, float] = {}
        self._slowdown: dict[str, float] = {}

    # ----------------------------------------------------------------- public API
    def get(self, url: str, *, min_delay: float | None = None) -> requests.Response:
        """GET ``url`` politely and return the successful response.

        ``min_delay`` is the caller's (per-source) politeness floor for this host, in seconds.
        Raises :class:`RobotsDisallowedError`, :class:`PermanentHTTPError`,
        :class:`RetryExhaustedError` or :class:`FetchError`.
        """
        url = self._validated(url)
        cached = self._cache.get(url)
        if cached is not None:
            self.stats.cache_hits += 1
            logger.debug("Cache hit, not downloading again: %s", url)
            return cached
        self._enforce_robots(url, min_delay)
        try:
            response = self._send(url, min_delay)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise PermanentHTTPError(url, _status_text(response), status_code=response.status_code) from exc
        except FetchError:
            self.stats.failures += 1
            raise
        self._cache[url] = response
        return response

    def get_soup(self, url: str, *, min_delay: float | None = None) -> BeautifulSoup:
        """GET ``url`` and parse it with ``html.parser``.

        The raw bytes are parsed (not ``response.text``) so a ``<meta charset>`` is honoured when the
        HTTP header declares no charset; requests would otherwise assume ISO-8859-1 and turn "£" into "Â£".
        """
        response = self.get(url, min_delay=min_delay)
        return BeautifulSoup(response.content, "html.parser", from_encoding=declared_charset(response))

    def robots_policy(self, url: str, *, min_delay: float | None = None) -> RobotsPolicy:
        """The robots.txt policy for the URL's host, fetched once per scheme+host and cached."""
        origin = _origin(url)
        policy = self._robots.get(origin)
        if policy is None:
            policy = self._load_robots(origin, min_delay)
            self._robots[origin] = policy
        return policy

    def stats_as_dict(self) -> dict[str, Any]:
        """JSON-safe request statistics, robots.txt outcomes and the request gap used per host."""
        data: dict[str, Any] = asdict(self.stats)
        data["robots_txt"] = {origin: policy.describe() for origin, policy in sorted(self._robots.items())}
        data["request_gap_seconds"] = {origin: round(gap, 2) for origin, gap in sorted(self._gap_used.items())}
        return data

    def close(self) -> None:
        """Close the underlying session if this client created it."""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> PoliteHttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ----------------------------------------------------------------- robots.txt
    def _enforce_robots(self, url: str, min_delay: float | None) -> None:
        policy = self.robots_policy(url, min_delay=min_delay)
        allowed, rule = policy.decide(url)
        if allowed:
            return
        self.stats.robots_blocked += 1
        if policy.status == "unreachable":
            reason = "robots.txt unreachable, so every URL on this host is treated as disallowed"
        else:
            reason = f"rule '{rule}' in group '{policy.matched_group}'"
        logger.debug("robots.txt disallows %s (%s)", url, reason)
        raise RobotsDisallowedError(url, reason)

    def _load_robots(self, origin: str, min_delay: float | None) -> RobotsPolicy:
        robots_url = f"{origin}/robots.txt"
        try:
            response = self._send(robots_url, min_delay)
        except FetchError as exc:
            self.stats.failures += 1
            logger.error(
                "robots.txt at %s is unreachable (%s); treating all of %s as disallowed (RFC 9309)",
                robots_url, exc.summary, origin,
            )
            return RobotsPolicy.unreachable(exc.status_code)
        status = response.status_code
        if 200 <= status < 300:
            text = response.content[:ROBOTS_MAX_BYTES].decode("utf-8-sig", errors="replace")  # -sig drops a BOM
            policy = parse_robots_txt(text, self.robots_token, http_status=status)
            logger.info(
                "robots.txt %s: group '%s', %d rule(s), crawl-delay %s",
                robots_url, policy.matched_group or "none", len(policy.rules), policy.crawl_delay,
            )
            return policy
        if 300 <= status < 500:
            logger.info(
                "robots.txt %s: HTTP %d (unavailable) - no crawl restrictions apply (RFC 9309)", robots_url, status
            )
            return RobotsPolicy.unavailable(status)
        self.stats.failures += 1
        logger.error(
            "robots.txt at %s answered HTTP %d; treating all of %s as disallowed (RFC 9309)", robots_url, status, origin
        )
        return RobotsPolicy.unreachable(status)

    # ----------------------------------------------------------------- transport
    def _send(self, url: str, min_delay: float | None) -> requests.Response:
        """Send the GET with throttling and bounded retries.

        Returns the first response whose status is not retryable (success or permanent error);
        raises :class:`RetryExhaustedError` when transient failures persist.
        """
        origin = _origin(url)
        max_attempts = 1 + self.max_retries
        attempt = 0
        failure = "no attempt made"
        status_code: int | None = None
        while attempt < max_attempts:
            attempt += 1
            self._throttle(origin, min_delay)
            retry_after: float | None = None
            try:
                self.stats.requests_sent += 1
                response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            except requests.exceptions.SSLError as exc:  # certificate problems do not fix themselves
                raise FetchError(url, f"SSLError: {exc}") from exc
            except TRANSIENT_EXCEPTIONS as exc:
                failure, status_code = type(exc).__name__, None
            except requests.RequestException as exc:  # invalid URL, too many redirects, ...
                raise FetchError(url, f"{type(exc).__name__}: {exc}") from exc
            else:
                self.stats.bytes_received += len(response.content)
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    if response.status_code < 400:
                        self.stats.successful_responses += 1
                    return response
                failure, status_code = _status_text(response), response.status_code
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                if status_code == 429:
                    self._slow_down(origin, min_delay)
            finally:
                self._last_request_at[origin] = self._clock()

            if attempt >= max_attempts:
                break
            delay = self._retry_delay(attempt, retry_after)
            if delay is None:
                failure = f"{failure}; server asked to wait {retry_after:.0f}s (limit {self.max_retry_after:.0f}s)"
                break
            self.stats.retries += 1
            logger.warning(
                "GET %s attempt %d/%d failed (%s); retrying in %.1fs", url, attempt, max_attempts, failure, delay
            )
            self._sleep(delay)
        raise RetryExhaustedError(url, failure, status_code=status_code, attempts=attempt)

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float | None:
        """Backoff before the next attempt; None if the server asks us to wait longer than we accept."""
        backoff = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1) + self._jitter() * JITTER_MAX_SECONDS)
        if retry_after is None:
            return backoff
        if retry_after > self.max_retry_after:
            return None  # never retry earlier than the server asked; give up on this URL instead
        return max(backoff, retry_after)

    def _throttle(self, origin: str, min_delay: float | None) -> None:
        """Sleep until the minimum gap since the previous request to ``origin`` has elapsed."""
        gap = self._request_gap(origin, min_delay)
        self._gap_used[origin] = gap
        last = self._last_request_at.get(origin)
        if last is None:
            logger.debug("First request to %s; request gap for this host is %.2fs", origin, gap)
            return
        wait = last + gap - self._clock()
        if wait > 0:
            logger.debug("Rate limit %s: sleeping %.2fs to keep a %.2fs gap", origin, wait, gap)
            self._sleep(wait)
        else:
            logger.debug("Rate limit %s: %.2fs gap already elapsed", origin, gap)

    def _request_gap(self, origin: str, min_delay: float | None) -> float:
        policy = self._robots.get(origin)
        crawl_delay = policy.crawl_delay if policy is not None and policy.crawl_delay is not None else 0.0
        return max(self.default_delay, min_delay or 0.0, crawl_delay, self._slowdown.get(origin, 0.0))

    def _slow_down(self, origin: str, min_delay: float | None) -> None:
        """React to HTTP 429 by widening the gap for this host for the rest of the run."""
        widened = min(self.backoff_max, self._request_gap(origin, min_delay) * RATE_LIMIT_SLOWDOWN_FACTOR)
        if widened > self._slowdown.get(origin, 0.0):
            self._slowdown[origin] = widened
            logger.warning("%s answered HTTP 429; slowing down to one request every %.1fs", origin, widened)

    @staticmethod
    def _validated(url: str) -> str:
        """Strip the fragment and reject anything that is not an absolute http(s) URL."""
        cleaned = urldefrag(url.strip()).url
        parts = urlsplit(cleaned)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise FetchError(url, "not an absolute http(s) URL")
        return cleaned
