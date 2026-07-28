"""Corporate actions: bonus, split, rights, dividend.

Why this module exists: Indian large caps do bonus issues and splits far more
often than developed-market peers, and an unadjusted price series turns a 1:1
bonus into a fake 50% crash. Every technical level the framework produces is
computed on adjusted data, and any action inside the lookback is surfaced
explicitly so an agent never reads a corporate action as a breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.http import nse_api
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import parse_date

log = get_logger(__name__)

ACTION_KEYWORDS = {
    "bonus": "BONUS",
    "split": "SPLIT",
    "face value": "SPLIT",
    "rights": "RIGHTS",
    "dividend": "DIVIDEND",
    "buy back": "BUYBACK",
    "buyback": "BUYBACK",
    "amalgamation": "MERGER",
    "merger": "MERGER",
    "demerger": "DEMERGER",
    "scheme of arrangement": "DEMERGER",
}


@dataclass
class CorporateAction:
    action_type: str
    ex_date: date | None
    detail: str
    ratio: str | None = None
    amount: float | None = None
    source: str = "yfinance"

    @property
    def is_price_adjusting(self) -> bool:
        """Actions that mechanically reset the quoted price."""
        return self.action_type in ("BONUS", "SPLIT", "DEMERGER", "RIGHTS")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        when = self.ex_date.isoformat() if self.ex_date else "date TBD"
        return f"{self.action_type} (ex {when}): {self.detail}"


def fetch_corporate_actions(
    ref: SymbolRef, lookback_days: int = 400, config: dict[str, Any] | None = None
) -> list[CorporateAction]:
    """Dividends and splits from yfinance, plus NSE's corporate-action feed."""
    actions: list[CorporateAction] = []
    cutoff = date.today() - timedelta(days=lookback_days)

    actions.extend(_from_yfinance(ref, cutoff, config))
    actions.extend(_from_nse(ref, cutoff, config))

    seen: set[tuple[str, str]] = set()
    unique: list[CorporateAction] = []
    for action in sorted(actions, key=lambda a: a.ex_date or date.min, reverse=True):
        key = (action.action_type, str(action.ex_date))
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def _from_yfinance(
    ref: SymbolRef, cutoff: date, config: dict[str, Any] | None
) -> list[CorporateAction]:
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return []

    cache = get_cache(config)
    key = {"symbol": ref.yf_symbol, "cutoff": str(cutoff)}
    hit = cache.get("corp_actions_yf", key)
    if hit is not None:
        return hit

    out: list[CorporateAction] = []
    try:
        frame = yf.Ticker(ref.yf_symbol).actions
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for stamp, row in frame.iterrows():
                when = stamp.date() if hasattr(stamp, "date") else None
                if when and when < cutoff:
                    continue
                dividend = float(row.get("Dividends", 0) or 0)
                split = float(row.get("Stock Splits", 0) or 0)
                if dividend:
                    out.append(
                        CorporateAction(
                            "DIVIDEND", when, f"Rs {dividend:.2f} per share", amount=dividend
                        )
                    )
                if split:
                    # yfinance folds bonus issues into the split ratio: a 1:1
                    # bonus arrives as a 2.0 split. Both adjust price the same way.
                    out.append(
                        CorporateAction(
                            "SPLIT", when, f"ratio {split:g}x (split or bonus)", ratio=f"{split:g}"
                        )
                    )
    except Exception as exc:
        log.debug("yfinance actions failed for %s: %s", ref.yf_symbol, exc)

    cache.set("corp_actions_yf", key, out)
    return out


def _from_nse(
    ref: SymbolRef, cutoff: date, config: dict[str, Any] | None
) -> list[CorporateAction]:
    """NSE corporate-actions feed. Unofficial; failure is expected and silent."""
    if config is not None and not config.get("online_tools", True):
        return []
    if ref.is_index:
        return []

    payload = nse_api("corporates-corporateActions", {"index": "equities", "symbol": ref.base_symbol})
    if not isinstance(payload, list):
        return []

    out: list[CorporateAction] = []
    for row in payload[:30]:
        purpose = str(row.get("subject") or row.get("purpose") or "")
        ex_raw = row.get("exDate") or row.get("ex_date")
        try:
            ex_date = parse_date(ex_raw) if ex_raw else None
        except ValueError:
            ex_date = None
        if ex_date and ex_date < cutoff:
            continue
        action_type = "OTHER"
        for keyword, label in ACTION_KEYWORDS.items():
            if keyword in purpose.lower():
                action_type = label
                break
        out.append(CorporateAction(action_type, ex_date, purpose.strip(), source="NSE"))
    return out


def upcoming_actions(actions: list[CorporateAction], within_days: int = 30) -> list[CorporateAction]:
    today = date.today()
    horizon = today + timedelta(days=within_days)
    return [a for a in actions if a.ex_date and today <= a.ex_date <= horizon]


def recent_price_adjusting(
    actions: list[CorporateAction], within_days: int = 90
) -> list[CorporateAction]:
    """Recent bonus/split/demerger events - the ones that break naive charts."""
    cutoff = date.today() - timedelta(days=within_days)
    return [
        a for a in actions if a.is_price_adjusting and a.ex_date and a.ex_date >= cutoff
    ]


def adjustment_warning(actions: list[CorporateAction]) -> str | None:
    """Warning text when the price series may be mid-adjustment."""
    recent = recent_price_adjusting(actions, within_days=120)
    if not recent:
        return None
    listed = "; ".join(str(a) for a in recent[:3])
    return (
        f"Price-adjusting corporate action(s) in the last 120 days: {listed}. "
        "Verify that historical levels, moving averages and any prior stop/target "
        "are quoted on the adjusted series before acting on them."
    )


def summarise(actions: list[CorporateAction], limit: int = 8) -> str:
    if not actions:
        return "No corporate actions found in the lookback window."
    return "\n".join(f"- {a}" for a in actions[:limit])
