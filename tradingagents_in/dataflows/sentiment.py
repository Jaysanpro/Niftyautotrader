"""Sentiment aggregation across Indian financial media and social venues.

Design notes:
  * News-headline sentiment is computed with a finance-tuned lexicon, not a
    generic one. "Beats estimates", "block deal", "pledge", "QIP" and "circuit"
    mean specific things in an Indian market context.
  * Social sources (X, Reddit, StockTwits) are only queried when credentials
    are present. Anything missing is reported as a failed source so the agent
    discounts its own confidence rather than inventing a number.
  * The LLM analyst gets the raw items *and* this deterministic score. The
    score anchors it; a model-only read of sentiment drifts badly.
"""

from __future__ import annotations

import os
import re
from typing import Any

from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.http import get_json
from tradingagents_in.dataflows.news import NewsItem
from tradingagents_in.models.analyst import SentimentSource
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

# Finance-specific polarity lexicon. Weights are deliberately blunt; precision
# here is false comfort, direction and magnitude are what matter.
POSITIVE_TERMS: dict[str, float] = {
    "beats": 1.0, "beat": 1.0, "outperform": 1.0, "upgrade": 1.2, "upgrades": 1.2,
    "buy rating": 1.2, "target raised": 1.2, "record high": 1.0, "lifetime high": 1.1,
    "surge": 0.9, "surges": 0.9, "rally": 0.8, "rallies": 0.8, "jumps": 0.8,
    "gains": 0.6, "profit rises": 1.0, "strong results": 1.1, "order win": 1.0,
    "bags order": 1.0, "wins contract": 1.0, "expansion": 0.6, "capex": 0.4,
    "dividend": 0.5, "bonus issue": 0.7, "buyback": 0.9, "stake buy": 0.7,
    "fii buying": 0.8, "block deal buy": 0.6, "approval": 0.7, "usfda approval": 1.2,
    "margin expansion": 1.0, "guidance raised": 1.2, "multibagger": 0.5,
    "turnaround": 0.8, "debt reduction": 0.9, "deleveraging": 0.9,
}

NEGATIVE_TERMS: dict[str, float] = {
    "misses": -1.0, "miss": -1.0, "downgrade": -1.2, "downgrades": -1.2,
    "sell rating": -1.2, "target cut": -1.2, "plunge": -1.1, "plunges": -1.1,
    "slumps": -1.0, "crash": -1.3, "falls": -0.6, "declines": -0.6,
    "profit falls": -1.0, "loss widens": -1.2, "weak results": -1.1,
    "margin pressure": -1.0, "guidance cut": -1.2, "resigns": -0.8,
    "cfo resigns": -1.2, "auditor resigns": -1.5, "auditor quits": -1.5,
    "pledge": -1.0, "pledged shares": -1.1, "promoter selling": -1.2,
    "stake sale": -0.6, "sebi probe": -1.5, "sebi order": -1.2, "sebi bans": -1.5,
    "income tax raid": -1.5, "ed raid": -1.5, "cbi": -1.2, "fraud": -1.6,
    "insolvency": -1.5, "nclt": -1.1, "default": -1.4, "downgraded rating": -1.2,
    "lower circuit": -1.2, "asm": -0.9, "gsm": -1.1, "fno ban": -0.8,
    "fii selling": -0.8, "qip": -0.4, "dilution": -0.8, "recall": -0.9,
    "usfda observation": -1.1, "import alert": -1.4, "warning letter": -1.3,
    "layoffs": -0.6, "strike": -0.7, "penalty": -0.9, "gst notice": -0.9,
    "tax demand": -0.9,
}

NEGATORS = ("no ", "not ", "never ", "denies ", "denied ", "rules out ")

_WORD = re.compile(r"[a-z]+")


def score_text(text: str) -> float:
    """Polarity in [-1, 1] for one headline."""
    lowered = text.lower()
    total = 0.0
    hits = 0
    for term, weight in {**POSITIVE_TERMS, **NEGATIVE_TERMS}.items():
        idx = lowered.find(term)
        if idx < 0:
            continue
        hits += 1
        # Crude negation window: "does not beat estimates" should not read bullish.
        window = lowered[max(0, idx - 18) : idx]
        total += -weight if any(neg in window for neg in NEGATORS) else weight
    if not hits:
        return 0.0
    return max(-1.0, min(1.0, total / (hits ** 0.5) / 1.5))


def score_headlines(items: list[NewsItem]) -> tuple[float, int]:
    """Recency-weighted mean polarity across headlines, plus the count scored."""
    scored: list[tuple[float, float]] = []
    for item in items:
        polarity = score_text(f"{item.title} {item.summary}")
        if polarity == 0.0:
            continue
        age = item.age_days if item.age_days is not None else 3.0
        weight = 1.0 / (1.0 + max(0.0, age) / 3.0)  # ~half-weight at 3 days old
        scored.append((polarity, weight))
    if not scored:
        return 0.0, 0
    num = sum(p * w for p, w in scored)
    den = sum(w for _, w in scored)
    return round(num / den, 3), len(scored)


# --------------------------------------------------------------- social APIs
def fetch_stocktwits(base_symbol: str, config: dict[str, Any] | None = None) -> SentimentSource | None:
    """StockTwits India coverage. Public endpoint, no key, often sparse for NSE."""
    if config is not None and not config.get("online_tools", True):
        return None
    cache = get_cache(config)
    key = {"symbol": base_symbol}
    hit = cache.get("stocktwits", key)
    if hit is not None:
        return hit

    payload = get_json(
        f"https://api.stocktwits.com/api/2/streams/symbol/{base_symbol}.NSE.json", timeout=10
    )
    messages = (payload or {}).get("messages")
    if not messages:
        return None

    bullish = bearish = 0
    for msg in messages:
        basic = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
        if basic == "Bullish":
            bullish += 1
        elif basic == "Bearish":
            bearish += 1
    labelled = bullish + bearish
    score = (bullish - bearish) / labelled if labelled else 0.0
    result = SentimentSource(
        source="StockTwits India",
        score=round(score, 3),
        volume=len(messages),
        notes=f"{bullish} bullish / {bearish} bearish of {len(messages)} messages",
    )
    cache.set("stocktwits", key, result)
    return result


def fetch_reddit(base_symbol: str, config: dict[str, Any] | None = None) -> SentimentSource | None:
    """r/IndiaInvestments + r/IndianStockMarket mentions via Reddit's public search."""
    if config is not None and not config.get("online_tools", True):
        return None
    cache = get_cache(config)
    key = {"symbol": base_symbol}
    hit = cache.get("reddit", key)
    if hit is not None:
        return hit

    posts: list[dict[str, Any]] = []
    for sub in ("IndiaInvestments", "IndianStockMarket", "DalalStreetTalks"):
        payload = get_json(
            f"https://www.reddit.com/r/{sub}/search.json",
            params={"q": base_symbol, "restrict_sr": "on", "sort": "new",
                    "t": "week", "limit": 25},
            timeout=12,
        )
        children = ((payload or {}).get("data") or {}).get("children") or []
        posts.extend(child.get("data", {}) for child in children)

    if not posts:
        return None
    scores = [
        score_text(f"{p.get('title', '')} {p.get('selftext', '')[:300]}") for p in posts
    ]
    non_zero = [s for s in scores if s != 0.0]
    result = SentimentSource(
        source="Reddit (r/IndiaInvestments, r/IndianStockMarket)",
        score=round(sum(non_zero) / len(non_zero), 3) if non_zero else 0.0,
        volume=len(posts),
        notes=f"{len(posts)} posts in the last week",
    )
    cache.set("reddit", key, result)
    return result


def fetch_x(base_symbol: str, config: dict[str, Any] | None = None) -> SentimentSource | None:
    """X/Twitter cashtag sentiment. Requires X_BEARER_TOKEN (paid API tier)."""
    token = os.getenv("X_BEARER_TOKEN")
    if not token or (config is not None and not config.get("online_tools", True)):
        return None

    from tradingagents_in.dataflows.http import get  # noqa: PLC0415

    query = f"(${base_symbol} OR #{base_symbol} OR #Nifty {base_symbol}) lang:en -is:retweet"
    resp = get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": query, "max_results": 50, "tweet.fields": "public_metrics"},
        timeout=15,
        retries=1,
    )
    if resp is None:
        return None
    try:
        tweets = resp.json().get("data") or []
    except ValueError:
        return None
    if not tweets:
        return None

    scores = [score_text(t.get("text", "")) for t in tweets]
    non_zero = [s for s in scores if s != 0.0]
    return SentimentSource(
        source="X/Twitter",
        score=round(sum(non_zero) / len(non_zero), 3) if non_zero else 0.0,
        volume=len(tweets),
        notes=f"{len(tweets)} recent cashtag posts",
    )


# ------------------------------------------------------------- aggregation
# Media outlets get more weight than anonymous social feeds.
SOURCE_WEIGHTS = {
    "Indian financial press": 1.0,
    "Company announcements": 0.9,
    "StockTwits India": 0.4,
    "Reddit (r/IndiaInvestments, r/IndianStockMarket)": 0.35,
    "X/Twitter": 0.4,
}


def aggregate(sources: list[SentimentSource]) -> float:
    """Weighted mean of per-source scores, weighted by trust x log volume."""
    import math

    num = den = 0.0
    for src in sources:
        trust = SOURCE_WEIGHTS.get(src.source, 0.3)
        weight = trust * math.log1p(max(1, src.volume))
        num += src.score * weight
        den += weight
    return round(num / den, 3) if den else 0.0


def buzz_level(sources: list[SentimentSource], baseline: int = 20) -> str:
    """How loud the chatter is relative to a normal week."""
    total = sum(s.volume for s in sources)
    if total < baseline * 0.4:
        return "quiet"
    if total < baseline * 1.5:
        return "normal"
    if total < baseline * 4:
        return "elevated"
    return "frenzied"


def contrarian_flag(score: float, buzz: str, price_change_pct: float | None) -> str | None:
    """Crowding check.

    Euphoria on a name that has already run, or capitulation after a crash, is
    where Indian retail flow is most often the wrong side of the trade.
    """
    if buzz in ("elevated", "frenzied") and score > 0.55 and (price_change_pct or 0) > 12:
        return (
            "Crowded long: sentiment is euphoric after a sharp run. Retail flow into an "
            "extended move is a common local top pattern - size down and demand a pullback entry."
        )
    if buzz in ("elevated", "frenzied") and score < -0.55 and (price_change_pct or 0) < -12:
        return (
            "Capitulation: uniformly negative chatter after a sharp fall. Often near-term "
            "exhaustion, but only tradable once price confirms - do not catch it on sentiment alone."
        )
    return None
