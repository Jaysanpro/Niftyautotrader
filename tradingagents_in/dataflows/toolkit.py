"""The data toolkit: one fetch pass per run, shared by every agent.

Agents do not call data sources directly. The graph runs :meth:`Toolkit.gather`
once and every analyst reads from the resulting bundle. That gives:

  * one consistent snapshot - the bull and the bear argue over the *same*
    numbers, which is the whole point of the debate;
  * bounded network cost and rate-limit exposure;
  * an explicit, inspectable record of what was and was not available.
"""

from __future__ import annotations

import time
from datetime import date
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradingagents_in.dataflows import (
    corporate_actions,
    fundamentals,
    indicators,
    macro,
    market_data,
    microstructure,
    news,
    sentiment,
)
from tradingagents_in.dataflows.brokers import get_broker
from tradingagents_in.dataflows.calendar_in import SessionContext, session_context
from tradingagents_in.dataflows.fno import OptionChain, fetch_option_chain, resolve_expiry
from tradingagents_in.dataflows.symbols import resolve
from tradingagents_in.models.analyst import FlowSnapshot, MacroIndicator, SentimentSource
from tradingagents_in.models.base import DataQuality, SymbolRef
from tradingagents_in.models.risk import MarketMicrostructure
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import as_of_moment

log = get_logger(__name__)


@dataclass
class MarketBundle:
    """Everything the agents get to reason over, gathered once."""

    ref: SymbolRef
    session: SessionContext
    quote: dict[str, Any] = field(default_factory=dict)
    prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    enriched: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: pd.DataFrame | None = None

    fundamentals: dict[str, Any] = field(default_factory=dict)
    peers: list = field(default_factory=list)
    corporate_actions: list = field(default_factory=list)

    news_items: list = field(default_factory=list)
    company_news: list = field(default_factory=list)
    announcements: list = field(default_factory=list)
    sentiment_sources: list[SentimentSource] = field(default_factory=list)

    macro_indicators: list[MacroIndicator] = field(default_factory=list)
    india_indicators: list[MacroIndicator] = field(default_factory=list)
    flows: FlowSnapshot = field(default_factory=FlowSnapshot)
    upcoming_events: list[str] = field(default_factory=list)

    micro: MarketMicrostructure = field(default_factory=MarketMicrostructure)
    option_chain: OptionChain | None = None

    sources_used: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    # ------------------------------------------------------------- helpers
    @property
    def daily(self) -> pd.DataFrame:
        return self.enriched.get("1d", self.prices.get("1d", pd.DataFrame()))

    @property
    def last_price(self) -> float | None:
        price = self.quote.get("last_price")
        if price is not None:
            return float(price)
        daily = self.daily
        return float(daily["close"].iloc[-1]) if not daily.empty else None

    def quality(self, expected: list[str]) -> DataQuality:
        """Data-quality record scoped to what a given analyst needed."""
        have = set(self.sources_used)
        missing = [item for item in expected if item not in have]
        completeness = (len(expected) - len(missing)) / len(expected) if expected else 1.0
        return DataQuality(
            sources_used=sorted(have & set(expected)),
            sources_failed=sorted(set(self.sources_failed) & set(expected)),
            missing_fields=missing,
            completeness=round(max(0.0, min(1.0, completeness)), 3),
        )


class Toolkit:
    """Fetches and caches every input for one symbol, for one run."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.online = bool(config.get("online_tools", True))
        self._broker = None
        if self.online and config.get("data_provider", "yfinance") != "yfinance":
            try:
                self._broker = get_broker(config)
            except Exception as exc:
                log.warning("broker feed unavailable, using yfinance: %s", exc)

    # ------------------------------------------------------------- gather
    def gather(self, symbol: str, as_of: str | date | None = None) -> MarketBundle:
        """Gather every input for one symbol, optionally as of a past session.

        ``as_of`` arrives per call rather than per Toolkit because the graph
        builds its gather node once and reuses it: a walk-forward run sweeps
        hundreds of dates through this one instance.
        """
        started = time.time()
        # Every dataflow below reads the date off the config it is handed, so
        # the override is applied once, here, instead of at each call.
        config = {**self.config, "as_of": as_of} if as_of else self.config

        if str(config.get("data_provider", "")).lower() == "synthetic":
            # Deterministic, network-free tape for CI, demos and air-gapped runs.
            from tradingagents_in.dataflows.synthetic import build_synthetic_bundle

            bundle = build_synthetic_bundle(symbol, config)
            bundle.elapsed_seconds = round(time.time() - started, 2)
            log.info("gathered %s from the synthetic provider", bundle.ref.yf_symbol)
            return bundle

        ref = resolve(symbol)
        exchange = ref.exchange.value if hasattr(ref.exchange, "value") else str(ref.exchange)
        bundle = MarketBundle(
            ref=ref,
            session=session_context(when=as_of_moment(config), exchange=exchange),
        )

        self._step(bundle, "prices", lambda: self._load_prices(bundle, config))
        self._step(bundle, "quote", lambda: self._load_quote(bundle, config))
        self._step(bundle, "benchmark", lambda: self._load_benchmark(bundle, config))

        if not ref.is_index:
            self._step(bundle, "fundamentals", lambda: self._load_fundamentals(bundle, config))
            self._step(bundle, "peers", lambda: self._load_peers(bundle, config))
            self._step(bundle, "corporate_actions", lambda: self._load_actions(bundle, config))

        self._step(bundle, "news", lambda: self._load_news(bundle, config))
        self._step(bundle, "sentiment", lambda: self._load_sentiment(bundle, config))
        self._step(bundle, "macro", lambda: self._load_macro(bundle, config))
        self._step(bundle, "flows", lambda: self._load_flows(bundle, config))
        self._step(bundle, "microstructure", lambda: self._load_micro(bundle, config))

        if config.get("allow_fno") and (ref.fno_eligible or ref.is_index):
            self._step(bundle, "option_chain", lambda: self._load_chain(bundle, config))

        bundle.elapsed_seconds = round(time.time() - started, 2)
        log.info(
            "gathered %s in %.1fs (sources ok: %s, failed: %s)",
            ref.yf_symbol, bundle.elapsed_seconds,
            len(bundle.sources_used), len(bundle.sources_failed),
        )
        return bundle

    def _step(self, bundle: MarketBundle, name: str, fn) -> None:
        """Run one fetch step; a failure marks the source, never aborts the run."""
        try:
            ok = fn()
            (bundle.sources_used if ok else bundle.sources_failed).append(name)
        except Exception as exc:
            log.warning("data step %r failed: %s", name, exc)
            bundle.sources_failed.append(name)

    # -------------------------------------------------------------- steps
    def _load_prices(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        timeframes = list(config.get("technical_timeframes", ["1d"]))
        if "1d" not in timeframes:
            timeframes.insert(0, "1d")
        bundle.prices = market_data.fetch_multi_timeframe(
            bundle.ref, timeframes,
            lookback_days=int(config.get("lookback_days", 400)),
            config=config,
        )
        for tf, frame in bundle.prices.items():
            if len(frame) < 20:
                continue
            bundle.enriched[tf] = indicators.compute_all(frame, intraday=tf not in ("1d", "1wk"))
        return bool(bundle.prices)

    def _load_quote(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        if self._broker is not None:
            quote = self._broker.quote(bundle.ref.base_symbol)
            if quote.get("last_price"):
                bundle.quote = quote
                return True
        bundle.quote = market_data.fetch_quote(bundle.ref, config)
        return bool(bundle.quote.get("last_price"))

    def _load_benchmark(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        symbol = config.get("benchmark_symbol", "^NSEI")
        bundle.benchmark = market_data.fetch_ohlcv(
            symbol, "1d", lookback_days=int(config.get("lookback_days", 400)),
            config=config,
        )
        return bundle.benchmark is not None and not bundle.benchmark.empty

    def _load_fundamentals(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        yf_data = fundamentals.fetch_yf_fundamentals(bundle.ref, config)
        screener = fundamentals.fetch_screener(bundle.ref, config)
        bundle.fundamentals = fundamentals.merge_sources(yf_data, screener)
        if bundle.fundamentals.get("name") and not bundle.ref.name:
            bundle.ref.name = bundle.fundamentals["name"]
        return bool(bundle.fundamentals.get("pe") or bundle.fundamentals.get("market_cap_inr"))

    def _load_peers(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        bundle.peers = fundamentals.fetch_peers(
            bundle.ref, limit=int(config.get("peer_count", 5)), config=config
        )
        return bool(bundle.peers)

    def _load_actions(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        bundle.corporate_actions = corporate_actions.fetch_corporate_actions(
            bundle.ref, lookback_days=int(config.get("lookback_days", 400)),
            config=config,
        )
        return True  # "no actions" is a valid, informative answer

    def _load_news(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        bundle.news_items = news.fetch_news(
            lookback_days=int(config.get("news_lookback_days", 7)),
            limit=int(config.get("max_news_items", 40)),
            config=config,
        )
        bundle.company_news = news.filter_for_symbol(
            bundle.news_items, bundle.ref.base_symbol, bundle.ref.name
        )
        if not bundle.ref.is_index:
            bundle.announcements = news.fetch_company_announcements(
                bundle.ref.base_symbol, config
            )
        return bool(bundle.news_items)

    def _load_sentiment(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        sources: list[SentimentSource] = []
        score, count = sentiment.score_headlines(
            bundle.company_news or bundle.news_items[:20], reference=bundle.session.now
        )
        if count:
            sources.append(
                SentimentSource(
                    source="Indian financial press",
                    score=score,
                    volume=count,
                    notes="ET, Moneycontrol, Business Standard, Mint, BusinessLine headlines",
                )
            )
        if not bundle.ref.is_index and self.online:
            for fetcher in (sentiment.fetch_stocktwits, sentiment.fetch_reddit, sentiment.fetch_x):
                try:
                    result = fetcher(bundle.ref.base_symbol, config)
                    if result:
                        sources.append(result)
                except Exception as exc:
                    log.debug("social source failed: %s", exc)
        bundle.sentiment_sources = sources
        return bool(sources)

    def _load_macro(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        bundle.macro_indicators = macro.fetch_macro_snapshot(config)
        bundle.india_indicators = macro.fetch_india_indicators(config)
        bundle.upcoming_events = macro.upcoming_events(as_of=bundle.session.trade_date)
        return bool(bundle.macro_indicators)

    def _load_flows(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        bundle.flows = macro.fetch_fii_dii_flows(config)
        return bundle.flows.fii_net_inr_cr is not None or bundle.flows.dii_net_inr_cr is not None

    def _load_micro(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        index_change = None
        for indicator in bundle.macro_indicators:
            if indicator.name == "nifty":
                index_change = indicator.change_pct
                break
        bundle.micro = microstructure.build_microstructure(
            bundle.ref, bundle.quote, bundle.prices.get("1d"), index_change, config
        )
        return True

    def _load_chain(self, bundle: MarketBundle, config: dict[str, Any]) -> bool:
        expiry = resolve_expiry(bundle.ref, when=bundle.session.trade_date)
        bundle.option_chain = fetch_option_chain(bundle.ref, expiry, config)
        return bundle.option_chain is not None
