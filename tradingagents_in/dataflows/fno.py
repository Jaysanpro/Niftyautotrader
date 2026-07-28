"""Futures & options helpers for the Indian derivatives market.

Covers what the trader agent needs to write an executable F&O order:
lot sizes, strike ladders, expiry selection, an option-chain read where one is
available, and a Black-76 fallback premium so a proposal always carries a
sanity-checked cost even when no live chain can be fetched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from tradingagents_in.config.universe import lot_size, strike_step
from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.calendar_in import (
    next_monthly_expiry,
    trading_days_between,
    weekly_expiry,
)
from tradingagents_in.dataflows.http import nse_api
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.models.enums import OptionType
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

# Index options settle weekly; single stocks are monthly-only in India.
INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}

# Rough SPAN+exposure margin as a fraction of contract value. Real margin comes
# from the broker; this is only used to check affordability before proposing.
APPROX_FUTURES_MARGIN_PCT = {"index": 0.12, "stock": 0.20}


@dataclass
class OptionQuote:
    strike: float
    option_type: OptionType
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    implied_volatility: float | None = None
    open_interest: int | None = None
    change_in_oi: int | None = None
    volume: int | None = None

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask:
            return round((self.bid + self.ask) / 2, 2)
        return self.last_price

    @property
    def spread_pct(self) -> float | None:
        """Bid-ask spread as % of mid - the practical liquidity test for options."""
        if not (self.bid and self.ask) or self.ask <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        return round((self.ask - self.bid) / mid * 100, 2) if mid else None


@dataclass
class OptionChain:
    underlying: str
    spot: float | None
    expiry: str
    calls: list[OptionQuote]
    puts: list[OptionQuote]
    source: str = "NSE"

    def atm_strike(self) -> float | None:
        if self.spot is None or not self.calls:
            return None
        return min((c.strike for c in self.calls), key=lambda s: abs(s - self.spot))

    def max_pain(self) -> float | None:
        """Strike where total option-writer loss is minimised.

        Widely watched around Indian expiry; useful as context, not as a signal
        on its own.
        """
        strikes = sorted({q.strike for q in self.calls} | {q.strike for q in self.puts})
        if not strikes:
            return None
        oi_call = {q.strike: (q.open_interest or 0) for q in self.calls}
        oi_put = {q.strike: (q.open_interest or 0) for q in self.puts}
        best, best_pain = None, math.inf
        for settle in strikes:
            pain = sum(max(0.0, settle - k) * oi_call.get(k, 0) for k in strikes)
            pain += sum(max(0.0, k - settle) * oi_put.get(k, 0) for k in strikes)
            if pain < best_pain:
                best, best_pain = settle, pain
        return best

    def pcr_oi(self) -> float | None:
        """Put-call ratio by open interest."""
        call_oi = sum(q.open_interest or 0 for q in self.calls)
        put_oi = sum(q.open_interest or 0 for q in self.puts)
        return round(put_oi / call_oi, 3) if call_oi else None

    def quote(self, strike: float, option_type: OptionType) -> OptionQuote | None:
        pool = self.calls if option_type == OptionType.CE else self.puts
        for q in pool:
            if abs(q.strike - strike) < 1e-6:
                return q
        return None


def is_index_underlying(base_symbol: str) -> bool:
    return base_symbol.upper() in INDEX_UNDERLYINGS


def resolve_expiry(
    ref: SymbolRef, horizon_days: int = 10, when: date | None = None
) -> date:
    """Pick a sensible expiry for the intended holding period.

    Single stocks only have monthly expiries. For indices we take the weekly
    unless the trade needs to live past it - buying a weekly for a two-week
    idea is how retail books donate premium to theta.
    """
    today = when or date.today()
    exchange = ref.exchange.value if hasattr(ref.exchange, "value") else str(ref.exchange)

    if not is_index_underlying(ref.base_symbol) and not ref.is_index:
        monthly = next_monthly_expiry(today, exchange)
        # Never propose a contract with fewer than 3 sessions left.
        if trading_days_between(today, monthly) < 3:
            following = next_monthly_expiry(monthly.replace(day=28), exchange)
            return following
        return monthly

    weekly = weekly_expiry(today, exchange)
    if trading_days_between(today, weekly) >= min(horizon_days, 3):
        return weekly
    monthly = next_monthly_expiry(today, exchange)
    return monthly if trading_days_between(today, weekly) < 2 else weekly


def round_to_strike(price: float, base_symbol: str, spot: float | None = None) -> float:
    step = strike_step(base_symbol, spot or price)
    return round(price / step) * step


def strike_ladder(
    spot: float, base_symbol: str, count: int = 5
) -> list[float]:
    """ATM plus `count` strikes either side."""
    step = strike_step(base_symbol, spot)
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-count, count + 1)]


def fetch_option_chain(
    ref: SymbolRef, expiry: date | None = None, config: dict[str, Any] | None = None
) -> OptionChain | None:
    """Live NSE option chain. Returns None when unavailable (a normal outcome)."""
    if config is not None and not config.get("online_tools", True):
        return None

    symbol = ref.base_symbol.upper()
    if symbol == "NIFTY50":
        symbol = "NIFTY"

    cache = get_cache(config)
    key = {"symbol": symbol, "expiry": str(expiry) if expiry else "nearest"}
    hit = cache.get("option_chain", key)
    if hit is not None:
        return hit

    endpoint = "option-chain-indices" if is_index_underlying(symbol) else "option-chain-equities"
    payload = nse_api(endpoint, {"symbol": symbol})
    records = (payload or {}).get("records") if isinstance(payload, dict) else None
    if not records:
        return None

    spot = records.get("underlyingValue")
    expiries = records.get("expiryDates") or []
    target = expiry.strftime("%d-%b-%Y") if expiry else (expiries[0] if expiries else "")

    calls: list[OptionQuote] = []
    puts: list[OptionQuote] = []
    for row in records.get("data", []):
        if target and row.get("expiryDate") != target:
            continue
        for side, bucket, opt_type in (
            ("CE", calls, OptionType.CE),
            ("PE", puts, OptionType.PE),
        ):
            leg = row.get(side)
            if not leg:
                continue
            bucket.append(
                OptionQuote(
                    strike=float(row.get("strikePrice", 0)),
                    option_type=opt_type,
                    last_price=_f(leg.get("lastPrice")),
                    bid=_f(leg.get("bidprice")),
                    ask=_f(leg.get("askPrice")),
                    implied_volatility=_f(leg.get("impliedVolatility")),
                    open_interest=_i(leg.get("openInterest")),
                    change_in_oi=_i(leg.get("changeinOpenInterest")),
                    volume=_i(leg.get("totalTradedVolume")),
                )
            )

    if not calls and not puts:
        return None
    chain = OptionChain(
        underlying=symbol, spot=_f(spot), expiry=target, calls=calls, puts=puts
    )
    cache.set("option_chain", key, chain)
    return chain


# --------------------------------------------------------------- pricing
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    days_to_expiry: float,
    volatility_pct: float,
    option_type: OptionType,
    rate_pct: float = 6.5,
) -> float:
    """Fallback premium estimate when no live chain is available.

    The 6.5% default rate approximates the Indian short-end risk-free rate.
    This is for sizing sanity only - never quote it as a tradable price.
    """
    t = max(days_to_expiry, 0.5) / 365.0
    sigma = max(volatility_pct, 1.0) / 100.0
    r = rate_pct / 100.0
    if spot <= 0 or strike <= 0:
        return 0.0

    d1 = (math.log(spot / strike) + (r + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == OptionType.CE:
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return round(max(price, 0.05), 2)


def contract_value(spot: float, base_symbol: str, lots: int = 1) -> float | None:
    size = lot_size(base_symbol)
    return round(spot * size * lots, 2) if size else None


def approx_futures_margin(
    spot: float, base_symbol: str, lots: int = 1, is_index: bool = False
) -> float | None:
    """Rough SPAN+exposure requirement. The broker's number is authoritative."""
    value = contract_value(spot, base_symbol, lots)
    if value is None:
        return None
    pct = APPROX_FUTURES_MARGIN_PCT["index" if is_index else "stock"]
    return round(value * pct, 2)


def max_affordable_lots(
    capital_inr: float, spot: float, base_symbol: str, is_index: bool = False
) -> int:
    """Lots the capital can margin. Returns 0 when even one lot is unaffordable."""
    per_lot = approx_futures_margin(spot, base_symbol, 1, is_index)
    if not per_lot or per_lot <= 0:
        return 0
    return int(capital_inr // per_lot)


def theta_warning(days_to_expiry: int, is_long_option: bool) -> str | None:
    """Expiry-decay caution, which is where most retail option buying dies."""
    if not is_long_option:
        return None
    if days_to_expiry <= 1:
        return (
            "Expiry day: long options are almost pure theta and gamma. Premium can go to "
            "zero within hours. Do not hold a directional long option into the close."
        )
    if days_to_expiry <= 3:
        return (
            f"Only {days_to_expiry} sessions to expiry - theta decay accelerates sharply in "
            "the final week. Prefer the next expiry or a debit spread to cap the bleed."
        )
    return None


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
