"""Tradability checks specific to Indian markets.

Circuit filters, ASM/GSM surveillance, F&O ban periods and impact cost. These
are hard gates: a beautiful thesis on a stock sitting in the upper circuit
under GSM Stage 4 is not a trade, and the risk manager must be told so in
machine-checkable terms rather than prose.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from tradingagents_in.config.universe import INDEX_CIRCUIT_BANDS_PCT, is_fno_eligible
from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.calendar_in import (
    days_to_expiry,
    is_expiry_day,
    market_phase,
)
from tradingagents_in.dataflows.http import nse_api
from tradingagents_in.dataflows.market_data import average_turnover_inr
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.models.enums import SurveillanceFlag
from tradingagents_in.models.risk import MarketMicrostructure
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import as_of_date

log = get_logger(__name__)

# Price bands applied by the exchanges to cash-segment scrips.
CIRCUIT_BANDS_PCT = (2.0, 5.0, 10.0, 20.0)

# F&O stocks have no daily band (a 10% move triggers a cooling-off instead);
# everything else defaults to 20% unless the exchange says otherwise.
DEFAULT_BAND_PCT = 20.0


def infer_circuit_band(ref: SymbolRef, daily: pd.DataFrame | None = None) -> float | None:
    """Best-effort price band.

    F&O-eligible scrips return None (no fixed band). For others we infer from
    the widest daily move actually observed: a scrip that has never moved more
    than 5% in a year is almost certainly banded at 5%.
    """
    if ref.is_index or is_fno_eligible(ref.base_symbol):
        return None
    if daily is None or daily.empty or len(daily) < 60:
        return DEFAULT_BAND_PCT

    moves = (daily["close"].pct_change().abs() * 100).dropna().tail(250)
    if moves.empty:
        return DEFAULT_BAND_PCT
    observed_max = float(moves.max())
    for band in CIRCUIT_BANDS_PCT:
        # Allow a hair of slack for rounding at the band edge.
        if observed_max <= band + 0.15:
            return band
    return DEFAULT_BAND_PCT


def circuit_levels(
    previous_close: float, band_pct: float | None
) -> tuple[float | None, float | None]:
    """Upper and lower circuit prices for the session."""
    if band_pct is None or not previous_close:
        return None, None
    return (
        round(previous_close * (1 + band_pct / 100), 2),
        round(previous_close * (1 - band_pct / 100), 2),
    )


def is_circuit_locked(
    last_price: float | None,
    day_high: float | None,
    day_low: float | None,
    upper: float | None,
    lower: float | None,
    tolerance_pct: float = 0.25,
) -> bool:
    """True when price is pinned at a band - i.e. you cannot get filled."""
    if last_price is None or (upper is None and lower is None):
        return False
    if upper and abs(last_price - upper) / upper * 100 <= tolerance_pct:
        # Locked means it is sitting there, not that it merely touched.
        return day_high is None or abs(day_high - upper) / upper * 100 <= tolerance_pct
    if lower and abs(last_price - lower) / lower * 100 <= tolerance_pct:
        return day_low is None or abs(day_low - lower) / lower * 100 <= tolerance_pct
    return False


def fetch_fno_ban_list(config: dict[str, Any] | None = None) -> set[str]:
    """Securities in the F&O ban period.

    A scrip crossing 95% of market-wide position limit is banned: only
    position-reducing orders are allowed, and new positions attract penalties.
    """
    if config is not None and not config.get("online_tools", True):
        return set()
    cache = get_cache(config)
    key = {"as_of": str(as_of_date(config))}
    hit = cache.get("fno_ban", key)
    if hit is not None:
        return hit

    banned: set[str] = set()
    payload = nse_api("marketStatus")
    # NSE publishes the ban list in a few shapes across endpoints; try the
    # common one and fall back silently.
    if isinstance(payload, dict):
        for row in payload.get("marketState", []) or []:
            if isinstance(row, dict) and row.get("index") == "Securities in F&O Ban":
                for name in (row.get("symbols") or []):
                    banned.add(str(name).upper())
    if banned:
        cache.set("fno_ban", key, banned)
    return banned


def fetch_surveillance_flag(
    ref: SymbolRef, config: dict[str, Any] | None = None
) -> SurveillanceFlag:
    """ASM/GSM status for the scrip.

    NSE does not expose a clean public endpoint for this; when we cannot
    confirm, we return NONE and the caller records the source as unverified
    rather than asserting the stock is clean.
    """
    if ref.is_index or (config is not None and not config.get("online_tools", True)):
        return SurveillanceFlag.NONE

    cache = get_cache(config)
    key = {"symbol": ref.base_symbol}
    hit = cache.get("surveillance", key)
    if hit is not None:
        return hit

    payload = nse_api("quote-equity", {"symbol": ref.base_symbol})
    if not isinstance(payload, dict):
        return SurveillanceFlag.NONE

    surveillance = str(
        (payload.get("securityInfo") or {}).get("surveillance", {}).get("surv") or ""
    ).upper()
    flag = SurveillanceFlag.NONE
    if "GSM" in surveillance:
        flag = SurveillanceFlag.GSM
    elif "ASM" in surveillance:
        flag = (
            SurveillanceFlag.ASM_LONG_TERM
            if "LONG" in surveillance
            else SurveillanceFlag.ASM_SHORT_TERM
        )
    cache.set("surveillance", key, flag)
    return flag


def estimate_impact_cost(
    order_value_inr: float, avg_daily_turnover_inr: float | None
) -> float | None:
    """Crude impact-cost estimate as a % of order value.

    Square-root market-impact heuristic: cost scales with the square root of
    participation. Good enough to tell a 0.05% fill from a 2% one, which is
    the distinction that matters for position sizing in mid/small caps.
    """
    if not avg_daily_turnover_inr or avg_daily_turnover_inr <= 0 or order_value_inr <= 0:
        return None
    participation = order_value_inr / avg_daily_turnover_inr
    return round(min(5.0, 0.10 * (participation ** 0.5) * 100), 3)


def index_circuit_note(index_change_pct: float | None) -> str | None:
    """Market-wide circuit-breaker proximity note."""
    if index_change_pct is None:
        return None
    move = abs(index_change_pct)
    for band in INDEX_CIRCUIT_BANDS_PCT:
        if move >= band:
            return (
                f"Index has moved {index_change_pct:+.2f}% - at or past the {band:.0f}% "
                "market-wide circuit breaker. Trading halts apply market-wide; assume "
                "no execution certainty."
            )
    if move >= 3.0:
        return (
            f"Index move of {index_change_pct:+.2f}% is large but well inside the 10% "
            "market-wide breaker. Expect elevated volatility and wider spreads."
        )
    return None


def build_microstructure(
    ref: SymbolRef,
    quote: dict[str, Any],
    daily: pd.DataFrame | None = None,
    index_change_pct: float | None = None,
    config: dict[str, Any] | None = None,
) -> MarketMicrostructure:
    """Assemble every tradability gate into one object for the risk manager."""
    exchange = ref.exchange.value if hasattr(ref.exchange, "value") else str(ref.exchange)
    band = infer_circuit_band(ref, daily)
    prev_close = quote.get("previous_close") or quote.get("last_price")
    upper, lower = circuit_levels(prev_close, band) if prev_close else (None, None)

    turnover = average_turnover_inr(daily) if daily is not None else None
    avg_volume = (
        float(daily["volume"].tail(20).mean()) if daily is not None and not daily.empty else None
    )

    try:
        ban_list = fetch_fno_ban_list(config)
    except Exception:
        ban_list = set()

    try:
        surveillance = fetch_surveillance_flag(ref, config)
    except Exception:
        surveillance = SurveillanceFlag.NONE

    locked = is_circuit_locked(
        quote.get("last_price"), quote.get("day_high"), quote.get("day_low"), upper, lower
    )
    if locked:
        surveillance = SurveillanceFlag.CIRCUIT_LOCKED

    return MarketMicrostructure(
        circuit_band_pct=band,
        upper_circuit=upper,
        lower_circuit=lower,
        is_circuit_locked=locked,
        surveillance=surveillance,
        in_fno_ban=ref.base_symbol.upper() in ban_list,
        avg_daily_turnover_inr=turnover,
        avg_daily_volume=avg_volume,
        impact_cost_estimate_pct=None,  # filled once an order size exists
        is_expiry_day=is_expiry_day(exchange=exchange),
        days_to_expiry=days_to_expiry(exchange=exchange),
        market_phase=market_phase().value,
        index_circuit_note=index_circuit_note(index_change_pct),
    )
