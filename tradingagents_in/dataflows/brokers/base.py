"""Broker feed interface.

Adapters are strictly *read-only* market-data sources. Order placement is
deliberately out of scope: this framework produces research and a proposed
trade, and a human (or a separate, audited execution system) places it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class BrokerFeed(ABC):
    """Read-only market-data adapter for an Indian broker API."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate. Returns False rather than raising when creds are absent."""

    @abstractmethod
    def ohlcv(
        self, symbol: str, interval: str, start: date, end: date
    ) -> pd.DataFrame:
        """Historical candles, normalised to lowercase OHLCV in IST."""

    @abstractmethod
    def quote(self, symbol: str) -> dict[str, Any]:
        """Last traded price and session stats."""

    def option_chain(self, underlying: str, expiry: date | None = None) -> Any | None:
        """Optional: live option chain. Default None means 'not supported'."""
        return None

    def margin_required(self, symbol: str, quantity: int, product: str) -> float | None:
        """Optional: broker-computed margin, which beats our approximation."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name} connected={self._connected}>"


class BrokerUnavailable(RuntimeError):
    """Raised when an adapter is selected but cannot be initialised."""


def get_broker(config: dict[str, Any]) -> BrokerFeed | None:
    """Instantiate the configured broker feed, or None for the free path."""
    provider = str(config.get("data_provider", "yfinance")).lower()
    if provider in ("yfinance", "", "none"):
        return None

    if provider == "angelone":
        from tradingagents_in.dataflows.brokers.angelone import AngelOneFeed

        feed: BrokerFeed = AngelOneFeed(config)
    elif provider in ("zerodha", "kite"):
        from tradingagents_in.dataflows.brokers.zerodha import ZerodhaFeed

        feed = ZerodhaFeed(config)
    else:
        raise BrokerUnavailable(f"unknown data_provider {provider!r}")

    return feed if feed.connect() else None
