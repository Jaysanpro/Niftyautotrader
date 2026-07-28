"""OHLCV retrieval for NSE/BSE instruments.

yfinance is the default because it is free and covers both ``.NS`` and ``.BO``.
Broker feeds (Angel One, Zerodha, TrueData) plug in through
:mod:`tradingagents_in.dataflows.brokers` and are preferred when configured,
since yfinance intraday history is capped at 60 days and can be gappy.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.calendar_in import previous_trading_day
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import IST

log = get_logger(__name__)

# yfinance intraday retention limits, in calendar days.
_INTRADAY_MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730}

_COLUMN_MAP = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}


class DataUnavailable(RuntimeError):
    """Raised when no usable price history could be retrieved."""


def _import_yf():
    try:
        import yfinance as yf  # noqa: PLC0415

        return yf
    except ImportError as exc:  # pragma: no cover - optional extra
        raise DataUnavailable(
            "yfinance is not installed. `pip install 'tradingagents-in[data]'`"
        ) from exc


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance frame to lowercase OHLCV in IST."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    # Single-ticker downloads still come back column-multi-indexed in newer
    # yfinance versions; flatten to the price field.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=_COLUMN_MAP)
    keep = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
    df = df[keep]
    df = df.loc[:, ~df.columns.duplicated()]

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = (
            df.index.tz_localize("UTC").tz_convert(IST)
            if df.index.tz is None
            else df.index.tz_convert(IST)
        )
    df.index.name = "datetime"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # A zero-volume, zero-range row is a placeholder, not a session.
    df = df.dropna(subset=[c for c in ("open", "high", "low", "close") if c in df.columns])
    return df


def _clamp_period(interval: str, start: date, end: date) -> date:
    cap = _INTRADAY_MAX_DAYS.get(interval)
    if cap is None:
        return start
    earliest = end - timedelta(days=cap - 1)
    return max(start, earliest)


def fetch_ohlcv(
    symbol: str | SymbolRef,
    interval: str = "1d",
    lookback_days: int = 400,
    end: date | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV for one instrument.

    Returns an empty frame rather than raising when the symbol is simply not
    covered, so a single bad ticker in a batch never kills the run.
    """
    yf_symbol = symbol.yf_symbol if isinstance(symbol, SymbolRef) else str(symbol)
    end_date = end or date.today()
    start_date = _clamp_period(interval, end_date - timedelta(days=lookback_days), end_date)

    cache = get_cache(config)
    params = {
        "symbol": yf_symbol, "interval": interval,
        "start": str(start_date), "end": str(end_date),
    }
    hit = cache.get("ohlcv", params)
    if hit is not None:
        return hit

    yf = _import_yf()
    try:
        raw = yf.download(
            yf_symbol,
            start=start_date,
            end=end_date + timedelta(days=1),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        log.warning("yfinance download failed for %s (%s): %s", yf_symbol, interval, exc)
        return pd.DataFrame()

    df = _tidy(raw)
    if df.empty:
        log.warning("no %s data for %s", interval, yf_symbol)
        return df
    cache.set("ohlcv", params, df)
    return df


def fetch_multi_timeframe(
    symbol: str | SymbolRef,
    timeframes: list[str],
    lookback_days: int = 400,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch several timeframes, skipping any that come back empty."""
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        # Intraday frames need far less calendar history than daily ones.
        days = lookback_days if tf in ("1d", "1wk", "1mo") else min(lookback_days, 59)
        df = fetch_ohlcv(symbol, interval=tf, lookback_days=days, config=config)
        if not df.empty:
            out[tf] = df
    return out


def fetch_quote(symbol: str | SymbolRef, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Last price and session stats. Falls back to the daily bar if needed."""
    yf_symbol = symbol.yf_symbol if isinstance(symbol, SymbolRef) else str(symbol)
    cache = get_cache(config)
    hit = cache.get("quote", {"symbol": yf_symbol})
    if hit is not None:
        return hit

    quote: dict[str, Any] = {"symbol": yf_symbol}
    yf = _import_yf()
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        quote.update(
            {
                "last_price": _f(getattr(info, "last_price", None)),
                "open": _f(getattr(info, "open", None)),
                "day_high": _f(getattr(info, "day_high", None)),
                "day_low": _f(getattr(info, "day_low", None)),
                "previous_close": _f(getattr(info, "previous_close", None)),
                "volume": _f(getattr(info, "last_volume", None)),
                "market_cap": _f(getattr(info, "market_cap", None)),
                "year_high": _f(getattr(info, "year_high", None)),
                "year_low": _f(getattr(info, "year_low", None)),
                "currency": getattr(info, "currency", "INR"),
            }
        )
    except Exception as exc:
        log.debug("fast_info unavailable for %s: %s", yf_symbol, exc)

    if quote.get("last_price") is None:
        daily = fetch_ohlcv(yf_symbol, "1d", lookback_days=10, config=config)
        if not daily.empty:
            row = daily.iloc[-1]
            quote.update(
                {
                    "last_price": float(row["close"]),
                    "open": float(row["open"]),
                    "day_high": float(row["high"]),
                    "day_low": float(row["low"]),
                    "volume": float(row.get("volume", 0) or 0),
                    "previous_close": float(daily["close"].iloc[-2]) if len(daily) > 1 else None,
                }
            )

    prev, last = quote.get("previous_close"), quote.get("last_price")
    if prev and last:
        quote["change_pct"] = round((last - prev) / prev * 100, 3)
    quote["as_of_session"] = str(previous_trading_day(date.today() + timedelta(days=1)))
    cache.set("quote", {"symbol": yf_symbol}, quote)
    return quote


def average_turnover_inr(df: pd.DataFrame, window: int = 20) -> float | None:
    """Average daily traded value in INR - the liquidity gate for sizing."""
    if df.empty or "volume" not in df.columns:
        return None
    recent = df.tail(window)
    if recent.empty:
        return None
    typical = (recent["high"] + recent["low"] + recent["close"]) / 3
    turnover = (typical * recent["volume"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(turnover.mean()) if not turnover.empty else None


def relative_strength(
    stock: pd.DataFrame, benchmark: pd.DataFrame, window: int = 60
) -> float | None:
    """Stock return minus benchmark return over `window` sessions, in pct points."""
    if stock.empty or benchmark.empty:
        return None
    joined = pd.concat(
        [stock["close"].rename("s"), benchmark["close"].rename("b")], axis=1
    ).dropna()
    if len(joined) < max(5, window // 4):
        return None
    tail = joined.tail(window)
    s_ret = (tail["s"].iloc[-1] / tail["s"].iloc[0] - 1) * 100
    b_ret = (tail["b"].iloc[-1] / tail["b"].iloc[0] - 1) * 100
    return round(float(s_ret - b_ret), 2)


def realised_volatility_pct(df: pd.DataFrame, window: int = 20) -> float | None:
    """Annualised realised volatility from daily closes (252 sessions)."""
    if df.empty or len(df) < window + 1:
        return None
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().tail(window)
    if returns.empty:
        return None
    return round(float(returns.std(ddof=1) * np.sqrt(252) * 100), 2)


def drawdown_from_high(df: pd.DataFrame, window: int = 252) -> float | None:
    """Current drawdown from the rolling high, in percent (negative)."""
    if df.empty:
        return None
    tail = df["close"].tail(window)
    peak = float(tail.max())
    if not peak:
        return None
    return round((float(tail.iloc[-1]) - peak) / peak * 100, 2)


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return None if np.isnan(out) else out
    except (TypeError, ValueError):
        return None
