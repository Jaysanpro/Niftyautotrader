"""Zerodha Kite Connect feed (optional).

Requires ``pip install 'tradingagents-in[brokers]'`` plus KITE_API_KEY and a
valid KITE_ACCESS_TOKEN. Kite access tokens expire daily and are minted through
an interactive login flow, so this adapter reads an existing token rather than
attempting to log in on its own.

Historical data on Kite requires the paid historical-data subscription.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pandas as pd

from tradingagents_in.dataflows.brokers.base import BrokerFeed
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import IST

log = get_logger(__name__)

_INTERVALS = {
    "1m": "minute", "3m": "3minute", "5m": "5minute", "15m": "15minute",
    "30m": "30minute", "1h": "60minute", "60m": "60minute", "1d": "day",
}


class ZerodhaFeed(BrokerFeed):
    name = "zerodha"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._kite: Any = None
        self._instruments: dict[str, int] = {}

    def connect(self) -> bool:
        api_key = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            log.info("Kite credentials missing; falling back to yfinance")
            return False
        try:
            from kiteconnect import KiteConnect  # noqa: PLC0415
        except ImportError:
            log.warning("kiteconnect not installed; falling back to yfinance")
            return False

        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            kite.profile()  # cheapest call that proves the token is live
            self._kite = kite
            self._connected = True
            log.info("Zerodha Kite connected")
            return True
        except Exception as exc:
            # Expired daily token is the overwhelmingly common case here.
            log.warning("Kite connection failed (token may have expired): %s", exc)
            return False

    def _instrument_token(self, symbol: str, exchange: str = "NSE") -> int | None:
        key = f"{exchange}:{symbol.upper()}"
        if key in self._instruments:
            return self._instruments[key]
        try:
            for row in self._kite.instruments(exchange):
                if row.get("tradingsymbol") == symbol.upper():
                    token = int(row["instrument_token"])
                    self._instruments[key] = token
                    return token
        except Exception as exc:
            log.debug("Kite instrument lookup failed for %s: %s", symbol, exc)
        return None

    def ohlcv(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        if not self._connected:
            return pd.DataFrame()
        token = self._instrument_token(symbol)
        if token is None:
            return pd.DataFrame()
        try:
            candles = self._kite.historical_data(
                token, start, end, _INTERVALS.get(interval, "day")
            )
        except Exception as exc:
            log.warning("Kite historical_data failed for %s: %s", symbol, exc)
            return pd.DataFrame()
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles).rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(IST)
        keep = ["open", "high", "low", "close", "volume"]
        return df.set_index("datetime")[keep].astype(float).sort_index()

    def quote(self, symbol: str) -> dict[str, Any]:
        if not self._connected:
            return {}
        key = f"NSE:{symbol.upper()}"
        try:
            payload = self._kite.quote([key])
        except Exception as exc:
            log.debug("Kite quote failed for %s: %s", symbol, exc)
            return {}

        data = (payload or {}).get(key) or {}
        ohlc = data.get("ohlc") or {}
        return {
            "symbol": symbol,
            "last_price": data.get("last_price"),
            "open": ohlc.get("open"),
            "day_high": ohlc.get("high"),
            "day_low": ohlc.get("low"),
            "previous_close": ohlc.get("close"),
            "volume": data.get("volume"),
            "oi": data.get("oi"),
            "source": "zerodha",
            "fetched_at": datetime.now(tz=IST).isoformat(),
        }

    def margin_required(self, symbol: str, quantity: int, product: str) -> float | None:
        """Broker-computed margin beats our SPAN approximation when available."""
        if not self._connected:
            return None
        try:
            payload = self._kite.order_margins(
                [
                    {
                        "exchange": "NSE",
                        "tradingsymbol": symbol.upper(),
                        "transaction_type": "BUY",
                        "variety": "regular",
                        "product": product,
                        "order_type": "MARKET",
                        "quantity": quantity,
                    }
                ]
            )
            return float(payload[0]["total"]) if payload else None
        except Exception as exc:
            log.debug("Kite order_margins failed: %s", exc)
            return None
