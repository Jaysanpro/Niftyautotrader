"""Indian financial news retrieval.

Public RSS feeds from the mainstream Indian business press plus company
announcements. RSS is used rather than scraping article pages: it is the
publisher-sanctioned interface, it is stable, and headlines carry most of the
signal an agent needs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any
from xml.etree import ElementTree

from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.http import get_text
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import IST, now_ist, to_ist, as_of_moment

log = get_logger(__name__)

# Publisher -> RSS endpoint.
FEEDS: dict[str, str] = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Economic Times Economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "Moneycontrol Business": "https://www.moneycontrol.com/rss/business.xml",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Moneycontrol Results": "https://www.moneycontrol.com/rss/results.xml",
    "Business Standard Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Business Standard Economy": "https://www.business-standard.com/rss/economy-102.rss",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
    "LiveMint Companies": "https://www.livemint.com/rss/companies",
    "The Hindu BusinessLine": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
}

# Feeds most relevant to the macro analyst specifically.
MACRO_FEEDS = (
    "Economic Times Economy",
    "Business Standard Economy",
    "LiveMint Markets",
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass
class NewsItem:
    title: str
    link: str
    published: datetime | None
    source: str
    summary: str = ""

    @property
    def age_days(self) -> float | None:
        return self.age_days_at(None)

    def age_days_at(self, reference: datetime | None) -> float | None:
        """Age against a given moment, defaulting to now.

        A dated run has to measure staleness from its own session; against the
        wall clock every item in a 2024 backtest looks ancient and the decay
        applied to it is meaningless.
        """
        if self.published is None:
            return None
        anchor = reference or now_ist()
        return (to_ist(anchor) - to_ist(self.published)).total_seconds() / 86400

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published"] = self.published.isoformat() if self.published else None
        return data

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        stamp = f"{to_ist(self.published):%d-%b %H:%M}" if self.published else "undated"
        return f"[{stamp}] {self.title} ({self.source})"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()


def _parse_pubdate(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        from email.utils import parsedate_to_datetime  # noqa: PLC0415

        parsed = parsedate_to_datetime(raw)
        return to_ist(parsed) if parsed else None
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return to_ist(datetime.strptime(raw.strip(), fmt))
        except ValueError:
            continue
    return None


def _parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    """Parse RSS 2.0 / Atom without requiring feedparser."""
    items: list[NewsItem] = []
    try:
        root = ElementTree.fromstring(xml_text.encode("utf-8", errors="ignore"))
    except ElementTree.ParseError as exc:
        log.debug("bad feed XML from %s: %s", source, exc)
        return items

    # RSS 2.0
    for node in root.iter("item"):
        title = _clean(node.findtext("title"))
        if not title:
            continue
        items.append(
            NewsItem(
                title=title,
                link=(node.findtext("link") or "").strip(),
                published=_parse_pubdate(node.findtext("pubDate")),
                source=source,
                summary=_clean(node.findtext("description"))[:400],
            )
        )
    if items:
        return items

    # Atom fallback
    ns = "{http://www.w3.org/2005/Atom}"
    for node in root.iter(f"{ns}entry"):
        title = _clean(node.findtext(f"{ns}title"))
        if not title:
            continue
        link_el = node.find(f"{ns}link")
        items.append(
            NewsItem(
                title=title,
                link=(link_el.get("href") if link_el is not None else "") or "",
                published=_parse_pubdate(node.findtext(f"{ns}updated")),
                source=source,
                summary=_clean(node.findtext(f"{ns}summary"))[:400],
            )
        )
    return items


def fetch_feed(name: str, url: str, config: dict[str, Any] | None = None) -> list[NewsItem]:
    cache = get_cache(config)
    hit = cache.get("rss", {"url": url})
    if hit is not None:
        return hit
    text = get_text(url, timeout=12)
    if not text:
        return []
    items = _parse_rss(text, name)
    if items:
        cache.set("rss", {"url": url}, items)
    return items


def fetch_news(
    feeds: dict[str, str] | None = None,
    lookback_days: int = 7,
    limit: int = 60,
    config: dict[str, Any] | None = None,
) -> list[NewsItem]:
    """Pull all configured feeds and return the freshest items, newest first."""
    if config is not None and not config.get("online_tools", True):
        return []
    chosen = feeds or FEEDS
    # On a dated run the window ends at that session, not at today. Without
    # the upper bound a backtest reads the coverage *of* the move it is about
    # to be scored on, which is the single most effective way to invent alpha
    # that does not exist.
    anchor = as_of_moment(config) or now_ist()
    dated_run = bool((config or {}).get("as_of"))
    cutoff = anchor - timedelta(days=lookback_days)

    collected: list[NewsItem] = []
    for name, url in chosen.items():
        try:
            collected.extend(fetch_feed(name, url, config))
        except Exception as exc:  # one dead feed must not stop the rest
            log.debug("feed %s failed: %s", name, exc)

    fresh: list[NewsItem] = []
    for item in collected:
        if item.published is None:
            # An undated item cannot be shown to predate the session, so a
            # dated run drops it rather than gambling on it. A live run keeps
            # it: there, "undated" just means a sloppy feed.
            if not dated_run:
                fresh.append(item)
            continue
        when = to_ist(item.published)
        if cutoff <= when <= anchor:
            fresh.append(item)
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in fresh:
        key = item.title.lower()[:90]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(key=lambda i: to_ist(i.published) if i.published else datetime.min.replace(tzinfo=IST),
                reverse=True)
    return unique[:limit]


def _match_terms(company_name: str | None, base_symbol: str) -> list[str]:
    """Search terms for a company: the ticker plus meaningful name tokens."""
    terms = {base_symbol.lower()}
    if company_name:
        cleaned = re.sub(
            r"\b(limited|ltd|india|indian|the|company|co|corporation|corp|inc|and|&)\b",
            " ",
            company_name.lower(),
        )
        for token in cleaned.split():
            if len(token) >= 4:
                terms.add(token)
    return sorted(terms)


def filter_for_symbol(
    items: list[NewsItem], base_symbol: str, company_name: str | None = None, limit: int = 12
) -> list[NewsItem]:
    """Headlines mentioning this company. Word-boundary matched to avoid
    'ITC' hitting 'switch' and similar false positives."""
    terms = _match_terms(company_name, base_symbol)
    patterns = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms]
    hits = [
        item
        for item in items
        if any(p.search(item.title) or p.search(item.summary) for p in patterns)
    ]
    return hits[:limit]


def fetch_company_announcements(
    base_symbol: str, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Corporate announcements filed with NSE for this scrip.

    NSE's endpoints are unofficial and frequently blocked; an empty list is a
    normal outcome and is reported as a missing source, not an error.
    """
    if config is not None and not config.get("online_tools", True):
        return []
    from tradingagents_in.dataflows.http import nse_api

    cache = get_cache(config)
    key = {"symbol": base_symbol, "kind": "announcements"}
    hit = cache.get("nse_ann", key)
    if hit is not None:
        return hit

    payload = nse_api("corporate-announcements", {"index": "equities", "symbol": base_symbol})
    if not isinstance(payload, list):
        return []
    out = [
        {
            "subject": row.get("desc") or row.get("subject"),
            "detail": (row.get("attchmntText") or "")[:400],
            "date": row.get("an_dt") or row.get("exchdisstime"),
        }
        for row in payload[:20]
    ]
    cache.set("nse_ann", key, out)
    return out


def summarise_headlines(items: list[NewsItem], limit: int = 15) -> str:
    if not items:
        return "(no headlines retrieved)"
    return "\n".join(f"- {item}" for item in items[:limit])
