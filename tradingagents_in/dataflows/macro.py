"""India macro context: global cues, USDINR, crude, and FII/DII flows.

The macro block is what separates an India-aware model from a generic one.
A Nifty view that ignores crude, the rupee and foreign flows is not a view.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tradingagents_in.config.universe import MACRO_SYMBOLS
from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.http import get_json, nse_api
from tradingagents_in.dataflows.market_data import fetch_ohlcv
from tradingagents_in.models.analyst import FlowSnapshot, MacroIndicator
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import as_of_date

log = get_logger(__name__)

# How each macro instrument transmits into Indian equities. This is the
# domain knowledge the LLM should not have to guess at.
TRANSMISSION: dict[str, str] = {
    "usdinr": (
        "A weaker rupee lifts IT/pharma exporters and hurts importers (OMCs, "
        "capital goods) and FII returns; sharp depreciation usually coincides "
        "with foreign outflows."
    ),
    "brent_crude": (
        "India imports ~85% of its crude. Higher Brent widens the current-account "
        "deficit, pressures the rupee and inflation, hurts OMCs/paints/aviation, "
        "and helps upstream (ONGC, Oil India)."
    ),
    "us_10y": (
        "Higher US yields compress the carry on EM allocations and typically "
        "coincide with FII selling in Indian cash equities."
    ),
    "dollar_index": "A strong dollar is a headwind for EM flows including India.",
    "us_sp500": "Overnight Wall Street direction sets the SGX/GIFT Nifty gap at 09:15 IST.",
    "us_nasdaq": "Nasdaq direction leads Nifty IT most reliably of all global cues.",
    "us_vix": "Elevated VIX signals global risk-off; Nifty typically gaps down on spikes.",
    "india_vix": (
        "India VIX above ~20 marks a fear regime: widen stops, cut size, and expect "
        "option premiums to be rich (favours sellers, punishes naive buyers)."
    ),
    "gold": "Gold strength alongside equity weakness confirms a genuine risk-off regime.",
    "hangseng": "China/HK strength can pull FII allocation away from India, and vice versa.",
    "shanghai": "China stimulus news reallocates EM flows and moves Indian metals.",
    "nikkei": "Broad Asia risk appetite proxy for the Indian open.",
}


def fetch_macro_snapshot(config: dict[str, Any] | None = None) -> list[MacroIndicator]:
    """Latest level and 1-day change for each tracked macro instrument."""
    indicators: list[MacroIndicator] = []
    for key, symbol in MACRO_SYMBOLS.items():
        try:
            df = fetch_ohlcv(symbol, "1d", lookback_days=30, config=config)
            if df.empty or len(df) < 2:
                continue
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
            change = (last - prev) / prev * 100 if prev else None
            indicators.append(
                MacroIndicator(
                    name=key,
                    value=round(last, 2),
                    change_pct=round(change, 2) if change is not None else None,
                    direction="up" if change and change > 0 else "down" if change else "flat",
                    implication_for_india=TRANSMISSION.get(key),
                )
            )
        except Exception as exc:
            log.debug("macro fetch failed for %s: %s", key, exc)
    return indicators


def fetch_fii_dii_flows(config: dict[str, Any] | None = None) -> FlowSnapshot:
    """FII/DII cash-segment net flows in INR crore.

    Primary source is NSE's own FII/DII endpoint. It is unofficial and often
    blocked from datacentre IPs, so an empty snapshot is an expected outcome
    and is surfaced to the agent as a missing source.
    """
    if config is not None and not config.get("online_tools", True):
        return FlowSnapshot(interpretation="offline mode - flows not fetched")

    cache = get_cache(config)
    key = {"as_of": str(as_of_date(config))}
    hit = cache.get("fii_dii", key)
    if hit is not None:
        return hit

    snapshot = FlowSnapshot()
    payload = nse_api("fiidiiTradeReact")
    rows = payload if isinstance(payload, list) else []

    for row in rows:
        category = str(row.get("category", "")).upper()
        net = _to_float(row.get("netValue"))
        if "FII" in category or "FPI" in category:
            snapshot.fii_net_inr_cr = net
            snapshot.date = row.get("date")
        elif "DII" in category:
            snapshot.dii_net_inr_cr = net

    if snapshot.fii_net_inr_cr is None:
        snapshot = _fallback_flows(snapshot, config)

    snapshot.interpretation = interpret_flows(snapshot)
    if snapshot.fii_net_inr_cr is not None or snapshot.dii_net_inr_cr is not None:
        cache.set("fii_dii", key, snapshot)
    return snapshot


def _fallback_flows(snapshot: FlowSnapshot, config: dict[str, Any] | None) -> FlowSnapshot:
    """Secondary flow source when NSE refuses the request."""
    payload = get_json(
        "https://www.moneycontrol.com/mc/widget/fiidii/getFiiDiiData",
        params={"classic": "true"},
        timeout=12,
    )
    if not isinstance(payload, list) or not payload:
        return snapshot
    latest = payload[0]
    snapshot.date = latest.get("date") or snapshot.date
    snapshot.fii_net_inr_cr = _to_float(latest.get("fii_net") or latest.get("netfii"))
    snapshot.dii_net_inr_cr = _to_float(latest.get("dii_net") or latest.get("netdii"))
    recent = payload[:5]
    fii_5d = [_to_float(r.get("fii_net") or r.get("netfii")) for r in recent]
    dii_5d = [_to_float(r.get("dii_net") or r.get("netdii")) for r in recent]
    snapshot.fii_5d_net_inr_cr = _sum_or_none(fii_5d)
    snapshot.dii_5d_net_inr_cr = _sum_or_none(dii_5d)
    return snapshot


def interpret_flows(snapshot: FlowSnapshot) -> str:
    """Plain-English read of the flow tape.

    The regime that matters most in India is the FII-sell / DII-buy standoff:
    domestic SIP money absorbing foreign selling keeps the index up while
    breadth quietly deteriorates.
    """
    fii, dii = snapshot.fii_net_inr_cr, snapshot.dii_net_inr_cr
    if fii is None and dii is None:
        return "FII/DII flow data unavailable for this session."

    parts: list[str] = []
    if fii is not None:
        parts.append(f"FII net {'buy' if fii >= 0 else 'sell'} Rs {abs(fii):,.0f} cr")
    if dii is not None:
        parts.append(f"DII net {'buy' if dii >= 0 else 'sell'} Rs {abs(dii):,.0f} cr")
    line = "; ".join(parts) + "."

    if fii is not None and dii is not None:
        if fii < 0 < dii:
            line += (
                " Domestic institutions are absorbing foreign selling - index level holds up "
                "but breadth typically narrows; favour large caps over small caps."
            )
        elif fii > 0 and dii > 0:
            line += " Both sides buying - the strongest flow backdrop for a sustained up-move."
        elif fii < 0 and dii < 0:
            line += " Both sides selling - a genuine risk-off tape; cut gross exposure."
        elif fii > 0 > dii:
            line += " Foreign buying against domestic profit-booking - usually a large-cap led rally."
    return line


def fetch_india_indicators(config: dict[str, Any] | None = None) -> list[MacroIndicator]:
    """India-domestic gauges: VIX regime and index positioning."""
    out: list[MacroIndicator] = []
    try:
        vix = fetch_ohlcv("^INDIAVIX", "1d", lookback_days=90, config=config)
        if not vix.empty:
            level = float(vix["close"].iloc[-1])
            avg = float(vix["close"].tail(60).mean())
            if level < 13:
                regime = "complacent - cheap options, tight ranges, breakouts often fail"
            elif level < 18:
                regime = "normal"
            elif level < 25:
                regime = "elevated - widen stops and cut position size"
            else:
                regime = "fear regime - rich premiums, gap risk, avoid naked option buying"
            out.append(
                MacroIndicator(
                    name="India VIX",
                    value=round(level, 2),
                    change_pct=round((level - avg) / avg * 100, 2) if avg else None,
                    direction="above 60d avg" if level > avg else "below 60d avg",
                    implication_for_india=regime,
                )
            )
    except Exception as exc:
        log.debug("India VIX fetch failed: %s", exc)
    return out


# ---------------------------------------------------------------- calendar
# Recurring, date-known Indian macro events. The agent uses this to flag
# event risk inside the intended holding period.
RECURRING_EVENTS: tuple[tuple[str, str], ...] = (
    ("RBI Monetary Policy Committee", "Roughly every two months (Feb, Apr, Jun, Aug, Oct, Dec)"),
    ("Union Budget", "1 February"),
    ("CPI inflation print", "12th of each month"),
    ("WPI inflation print", "14th of each month"),
    ("IIP data", "12th of each month"),
    ("GDP release", "End of Feb / May / Aug / Nov"),
    ("Monsoon onset (IMD)", "Kerala onset around 1 June; progress through September"),
    ("GST collection data", "1st of each month"),
    ("F&O weekly expiry", "Thursday (NSE) / Tuesday (BSE)"),
    ("Quarterly results season", "Mid-Jan, mid-Apr, mid-Jul, mid-Oct onwards"),
    ("US FOMC", "8 times a year - sets the global risk tone for the Indian open"),
)


def upcoming_events(within_days: int = 21, as_of: date | None = None) -> list[str]:
    """Dated catalysts inside the horizon, plus the standing recurring list.

    ``as_of`` anchors the horizon to the session being decided: a run dated
    last March should be looking at that March's Budget, not next year's.
    """
    today = as_of or date.today()
    events: list[str] = []

    # Union Budget
    budget = date(today.year, 2, 1)
    if budget < today:
        budget = date(today.year + 1, 2, 1)
    if (budget - today).days <= within_days:
        events.append(f"Union Budget on {budget} - expect sector-rotation volatility")

    # Monthly data prints
    for label, day_of_month in (("CPI inflation", 12), ("IIP", 12), ("WPI inflation", 14)):
        try:
            nxt = today.replace(day=day_of_month)
        except ValueError:
            continue
        if nxt < today:
            nxt = (today.replace(day=1) + timedelta(days=32)).replace(day=day_of_month)
        if (nxt - today).days <= within_days:
            events.append(f"{label} release around {nxt}")

    events.extend(f"{name}: {when}" for name, when in RECURRING_EVENTS)
    return events


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("₹", "").strip())
    except (TypeError, ValueError):
        return None


def _sum_or_none(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean), 2) if clean else None
