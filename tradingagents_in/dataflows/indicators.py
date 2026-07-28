"""Technical indicators, implemented on pandas so there is no TA-Lib build step.

Conventions:
  * every function takes an OHLCV DataFrame with columns
    ``open, high, low, close, volume`` and a DatetimeIndex;
  * every function returns a Series/DataFrame aligned to that index;
  * warm-up periods are NaN rather than back-filled, so a signal is never
    manufactured out of insufficient history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")


def _validate(df: pd.DataFrame, needs: tuple[str, ...] = OHLCV) -> None:
    missing = [c for c in needs if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")


# ------------------------------------------------------------------ averages
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length, min_periods=length).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


# --------------------------------------------------------------- oscillators
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI (smoothed, not the simple-average variant)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # avg_loss == 0 means an unbroken run of gains -> RSI 100 by definition.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    _validate(df, ("high", "low", "close"))
    lowest = df["low"].rolling(k, min_periods=k).min()
    highest = df["high"].rolling(k, min_periods=k).max()
    percent_k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0.0, np.nan)
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_k.rolling(d).mean()})


# --------------------------------------------------------------- volatility
def true_range(df: pd.DataFrame) -> pd.Series:
    _validate(df, ("high", "low", "close"))
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, length)
    dev = series.rolling(length, min_periods=length).std(ddof=0)
    upper, lower = mid + std * dev, mid - std * dev
    width = (upper - lower) / mid.replace(0.0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percent_b": percent_b,
        }
    )


def keltner(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], length)
    rng = atr(df, length)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + mult * rng, "kc_lower": mid - mult * rng})


# ------------------------------------------------------------------- trend
def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend - the default trailing-stop indicator on most Indian charts.

    Returns ``supertrend`` (the line) and ``direction`` (+1 long, -1 short).
    """
    _validate(df, ("high", "low", "close"))
    hl2 = (df["high"] + df["low"]) / 2
    rng = atr(df, length)
    upper_basic = hl2 + multiplier * rng
    lower_basic = hl2 - multiplier * rng

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    close = df["close"].to_numpy()
    ub = upper_basic.to_numpy(copy=True)
    lb = lower_basic.to_numpy(copy=True)

    # Bands only tighten while price stays on the same side; they reset on a flip.
    for i in range(1, len(df)):
        if np.isnan(ub[i]) or np.isnan(ub[i - 1]):
            continue
        if ub[i] > ub[i - 1] and close[i - 1] <= ub[i - 1]:
            ub[i] = ub[i - 1]
        if lb[i] < lb[i - 1] and close[i - 1] >= lb[i - 1]:
            lb[i] = lb[i - 1]
    upper[:] = ub
    lower[:] = lb

    direction = np.full(len(df), np.nan)
    line = np.full(len(df), np.nan)
    trend = 1
    for i in range(len(df)):
        if np.isnan(ub[i]):
            continue
        if np.isnan(line[i - 1]) if i > 0 else True:
            trend = 1 if close[i] > ub[i] else -1
        elif trend == 1 and close[i] < lb[i]:
            trend = -1
        elif trend == -1 and close[i] > ub[i]:
            trend = 1
        direction[i] = trend
        line[i] = lb[i] if trend == 1 else ub[i]

    return pd.DataFrame(
        {"supertrend": line, "supertrend_dir": direction}, index=df.index
    )


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI."""
    _validate(df, ("high", "low", "close"))
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1 / length
    tr_s = true_range(df).ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_s = pd.Series(plus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False, min_periods=length
    ).mean()
    minus_s = pd.Series(minus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False, min_periods=length
    ).mean()

    safe_tr = tr_s.replace(0.0, np.nan)
    plus_di = 100 * plus_s / safe_tr
    minus_di = 100 * minus_s / safe_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "adx": dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean(),
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    _validate(df, ("high", "low"))
    upper = df["high"].rolling(length, min_periods=length).max()
    lower = df["low"].rolling(length, min_periods=length).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2})


# ------------------------------------------------------------------ volume
def vwap(df: pd.DataFrame, anchor: str | None = "D") -> pd.Series:
    """Volume-weighted average price.

    Intraday VWAP resets each session (``anchor="D"``), which is how every
    Indian trading terminal draws it. Pass ``anchor=None`` for a cumulative
    VWAP over the whole frame (useful on daily bars).
    """
    _validate(df, ("high", "low", "close", "volume"))
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    if anchor and isinstance(df.index, pd.DatetimeIndex):
        groups = df.index.normalize()
        cum_pv = pv.groupby(groups).cumsum()
        cum_vol = df["volume"].groupby(groups).cumsum()
    else:
        cum_pv, cum_vol = pv.cumsum(), df["volume"].cumsum()
    return cum_pv / cum_vol.replace(0.0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    _validate(df, ("close", "volume"))
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Latest volume relative to its rolling average - the 'volume spike' read."""
    _validate(df, ("volume",))
    avg = df["volume"].rolling(length, min_periods=max(2, length // 2)).mean()
    return df["volume"] / avg.replace(0.0, np.nan)


# ------------------------------------------------------------------ pivots
def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """Floor-trader pivots computed off the previous session - the Indian default."""
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "r2": pivot + (high - low),
        "r3": high + 2 * (pivot - low),
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
        "s3": low - 2 * (high - pivot),
    }


def camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    rng = high - low
    return {
        "h4": close + rng * 1.1 / 2,
        "h3": close + rng * 1.1 / 4,
        "l3": close - rng * 1.1 / 4,
        "l4": close - rng * 1.1 / 2,
    }


def cpr(high: float, low: float, close: float) -> dict[str, float]:
    """Central Pivot Range. A narrow CPR is read as a breakout setup in India."""
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pivot - bc
    top, bottom = max(tc, bc), min(tc, bc)
    return {
        "cpr_pivot": pivot,
        "cpr_top": top,
        "cpr_bottom": bottom,
        "cpr_width_pct": (top - bottom) / pivot * 100 if pivot else float("nan"),
    }


# ----------------------------------------------------------- swing levels
def swing_levels(
    df: pd.DataFrame, lookback: int = 120, window: int = 5, max_levels: int = 4
) -> tuple[list[float], list[float]]:
    """Fractal swing highs/lows near the current price, as support/resistance.

    Returns ``(supports, resistances)`` sorted by proximity to the last close.
    """
    _validate(df, ("high", "low", "close"))
    recent = df.tail(lookback)
    if len(recent) < window * 2 + 1:
        return [], []

    highs, lows = recent["high"].to_numpy(), recent["low"].to_numpy()
    pivots_hi: list[float] = []
    pivots_lo: list[float] = []
    for i in range(window, len(recent) - window):
        seg_hi = highs[i - window : i + window + 1]
        seg_lo = lows[i - window : i + window + 1]
        if highs[i] == seg_hi.max():
            pivots_hi.append(float(highs[i]))
        if lows[i] == seg_lo.min():
            pivots_lo.append(float(lows[i]))

    last = float(recent["close"].iloc[-1])
    supports = sorted({round(p, 2) for p in pivots_lo if p < last}, key=lambda p: last - p)
    resistances = sorted({round(p, 2) for p in pivots_hi if p > last}, key=lambda p: p - last)
    return supports[:max_levels], resistances[:max_levels]


# ------------------------------------------------------------ aggregation
def compute_all(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Attach the full indicator set used by the Technical Analyst."""
    _validate(df)
    out = df.copy()
    close = out["close"]

    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)
    out["sma_50"] = sma(close, 50)
    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close))
    out = out.join(bollinger(close))
    out = out.join(supertrend(out))
    out = out.join(adx(out))
    out = out.join(donchian(out))
    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / close.replace(0.0, np.nan) * 100
    out["vwap"] = vwap(out, anchor="D" if intraday else None)
    out["obv"] = obv(out)
    out["volume_ratio_20"] = volume_ratio(out, 20)
    out["return_1"] = close.pct_change() * 100
    return out


def latest_values(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, float | None]:
    """Last non-NaN value per column, as plain floats for Pydantic."""
    if df.empty:
        return {}
    cols = columns or [c for c in df.columns if c not in OHLCV]
    row = df.iloc[-1]
    result: dict[str, float | None] = {}
    for col in cols:
        if col not in df.columns:
            continue
        value = row.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            result[col] = None
        else:
            result[col] = float(value)
    return result


def classify_trend(df: pd.DataFrame) -> str:
    """Trend label from EMA stack + ADX. Deliberately conservative."""
    if df.empty or "ema_20" not in df.columns:
        return "sideways"
    row = df.iloc[-1]
    ema20, ema50, close = row.get("ema_20"), row.get("ema_50"), row.get("close")
    adx_val = row.get("adx")
    if any(pd.isna(x) for x in (ema20, ema50, close)):
        return "sideways"
    # Below ADX 20 the market is ranging and EMA order is noise.
    if not pd.isna(adx_val) and adx_val < 20:
        return "sideways"
    if close > ema20 > ema50:
        return "uptrend"
    if close < ema20 < ema50:
        return "downtrend"
    return "sideways"


def volatility_regime(df: pd.DataFrame) -> str:
    """ATR% of price versus its own one-year distribution."""
    if "atr_pct" not in df.columns or df["atr_pct"].dropna().empty:
        return "normal"
    series = df["atr_pct"].dropna()
    current = float(series.iloc[-1])
    window = series.tail(250)
    if len(window) < 30:
        return "normal"
    p25, p75, p95 = (float(window.quantile(q)) for q in (0.25, 0.75, 0.95))
    if current >= p95:
        return "extreme"
    if current >= p75:
        return "high"
    if current <= p25:
        return "low"
    return "normal"
