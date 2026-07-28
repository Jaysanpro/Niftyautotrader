"""Polite HTTP helpers.

Indian data portals (NSE especially) are picky: they require a browser-like
User-Agent and a warm cookie jar obtained by visiting the homepage first. This
module centralises that so individual fetchers stay readable, and enforces a
minimum delay between hits to the same host.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Connection": "keep-alive",
}

NSE_HEADERS = {
    **DEFAULT_HEADERS,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}

MIN_INTERVAL_SECONDS = 1.0

_lock = threading.Lock()
_last_hit: dict[str, float] = {}
_sessions: dict[str, requests.Session] = {}


def _throttle(host: str) -> None:
    with _lock:
        last = _last_hit.get(host, 0.0)
        wait = MIN_INTERVAL_SECONDS - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        _last_hit[host] = time.time()


def get_session(host: str, headers: dict[str, str] | None = None) -> requests.Session:
    """One keep-alive session per host, so cookies persist across calls."""
    with _lock:
        session = _sessions.get(host)
        if session is None:
            session = requests.Session()
            session.headers.update(headers or DEFAULT_HEADERS)
            _sessions[host] = session
        return session


def get(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    retries: int = 2,
) -> requests.Response | None:
    """GET with throttling and bounded retries. Returns None on failure."""
    host = urlparse(url).netloc
    session = get_session(host, headers)
    for attempt in range(retries + 1):
        _throttle(host)
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            # 401/403 from NSE usually means the cookie jar went cold.
            if resp.status_code in (401, 403) and "nseindia" in host:
                warm_nse(session)
                continue
            log.debug("GET %s -> HTTP %s", url, resp.status_code)
        except requests.RequestException as exc:
            log.debug("GET %s failed (attempt %d): %s", url, attempt + 1, exc)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return None


def get_text(url: str, **kwargs: Any) -> str | None:
    resp = get(url, **kwargs)
    return resp.text if resp is not None else None


def get_json(url: str, **kwargs: Any) -> Any | None:
    kwargs.setdefault("headers", NSE_HEADERS)
    resp = get(url, **kwargs)
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        log.debug("non-JSON response from %s", url)
        return None


def warm_nse(session: requests.Session | None = None) -> requests.Session:
    """Visit the NSE homepage to pick up the cookies its API endpoints demand."""
    session = session or get_session("www.nseindia.com", NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", headers=DEFAULT_HEADERS, timeout=10)
    except requests.RequestException as exc:
        log.debug("NSE cookie warm-up failed: %s", exc)
    return session


def nse_api(path: str, params: dict[str, Any] | None = None, timeout: int = 15) -> Any | None:
    """Call an NSE public API path, warming cookies first.

    NSE actively discourages programmatic access and changes these endpoints
    without notice, so every caller must treat None as an expected outcome.
    """
    session = get_session("www.nseindia.com", NSE_HEADERS)
    if not session.cookies:
        warm_nse(session)
    url = f"https://www.nseindia.com/api/{path.lstrip('/')}"
    _throttle("www.nseindia.com")
    try:
        resp = session.get(url, params=params, headers=NSE_HEADERS, timeout=timeout)
        if resp.status_code in (401, 403):
            warm_nse(session)
            _throttle("www.nseindia.com")
            resp = session.get(url, params=params, headers=NSE_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("NSE api %s failed: %s", path, exc)
        return None
