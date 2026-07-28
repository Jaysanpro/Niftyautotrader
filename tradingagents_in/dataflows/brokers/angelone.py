"""Angel One SmartAPI feed (optional).

Requires ``pip install 'tradingagents-in[brokers]'`` and these env vars:
ANGELONE_API_KEY, ANGELONE_CLIENT_ID, ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET.

SmartAPI gives clean 1-minute history well past yfinance's 60-day intraday
window, which is the main reason to wire it up.
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

# Our interval labels -> SmartAPI's.
_INTERVALS = {
    "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR",
    "60m": "ONE_HOUR", "1d": "ONE_DAY",
}

INSTRUMENT_LIST_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)


class AngelOneFeed(BrokerFeed):
    name = "angelone"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._client: Any = None
        self._tokens: dict[str, str] = {}

    def connect(self) -> bool:
        api_key = os.getenv("ANGELONE_API_KEY")
        client_id = os.getenv("ANGELONE_CLIENT_ID")
        password = os.getenv("ANGELONE_PASSWORD")
        totp_secret = os.getenv("ANGELONE_TOTP_SECRET")
        if not all((api_key, client_id, password, totp_secret)):
            log.info("Angel One credentials incomplete; falling back to yfinance")
            return False

        try:
            import pyotp  # noqa: PLC0415
            from SmartApi import SmartConnect  # noqa: PLC0415
        except ImportError:
            log.warning("smartapi-python/pyotp not installed; falling back to yfinance")
            return False

        try:
            client = SmartConnect(api_key=api_key)
            session = client.generateSession(
                client_id, password, pyotp.TOTP(totp_secret).now()
            )
            if not session or not session.get("status"):
                log.warning("Angel One login rejected: %s", (session or {}).get("message"))
                return False
            self._client = client
            self._connected = True
            log.info("Angel One SmartAPI connected")
            return True
        except Exception as exc:
            log.warning("Angel One connection failed: %s", exc)
            return False

    def _token_for(self, symbol: str, exchange: str = "NSE") -> str | None:
        """Resolve an NSE trading symbol to its SmartAPI instrument token."""
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._tokens:
            return self._tokens[cache_key]

        from tradingagents_in.dataflows.cache import get_cache
        from tradingagents_in.dataflows.http import get_json

        cache = get_cache(self.config)
        instruments = cache.get("angel_instruments", {"v": 1})
        if instruments is None:
            instruments = get_json(INSTRUMENT_LIST_URL, timeout=60)
            if instruments:
                cache.set("angel_instruments", {"v": 1}, instruments)
        if not instruments:
            return None

        wanted = f"{symbol.upper()}-EQ"
        for row in instruments:
            if row.get("exch_seg") == exchange and row.get("symbol") == wanted:
                token = str(row.get("token"))
                self._tokens[cache_key] = token
                return token
        return None

    def ohlcv(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        if not self._connected:
            return pd.DataFrame()
        token = self._token_for(symbol)
        if token is None:
            log.debug("no Angel One token for %s", symbol)
            return pd.DataFrame()

        try:
            payload = self._client.getCandleData(
                {
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": _INTERVALS.get(interval, "ONE_DAY"),
                    "fromdate": f"{start:%Y-%m-%d} 09:15",
                    "todate": f"{end:%Y-%m-%d} 15:30",
                }
            )
        except Exception as exc:
            log.warning("Angel One candle fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        rows = (payload or {}).get("data") or []
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(IST)
        return df.set_index("datetime").astype(float).sort_index()

    def quote(self, symbol: str) -> dict[str, Any]:
        if not self._connected:
            return {}
        token = self._token_for(symbol)
        if token is None:
            return {}
        try:
            payload = self._client.ltpData("NSE", f"{symbol.upper()}-EQ", token)
        except Exception as exc:
            log.debug("Angel One ltpData failed for %s: %s", symbol, exc)
            return {}

        data = (payload or {}).get("data") or {}
        return {
            "symbol": symbol,
            "last_price": _f(data.get("ltp")),
            "open": _f(data.get("open")),
            "day_high": _f(data.get("high")),
            "day_low": _f(data.get("low")),
            "previous_close": _f(data.get("close")),
            "source": "angelone",
            "fetched_at": datetime.now(tz=IST).isoformat(),
        }


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
