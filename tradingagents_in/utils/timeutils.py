"""Asia/Kolkata time handling.

Every timestamp that crosses a module boundary in this project is tz-aware and
in IST. Naive datetimes are assumed to already be IST rather than UTC, because
every data source we touch reports Indian market time.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

IST_LABEL = "Asia/Kolkata"


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def to_ist(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def today_ist() -> date:
    return now_ist().date()


def combine_ist(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=IST)


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return to_ist(value).date()
    if isinstance(value, date):
        return value
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {value!r}")


def fmt_ist(value: datetime, with_tz: bool = True) -> str:
    ist = to_ist(value)
    stamp = ist.strftime("%Y-%m-%d %H:%M")
    return f"{stamp} IST" if with_tz else stamp


def days_between(start: date, end: date) -> int:
    return (end - start).days


def as_of_date(config: dict | None = None) -> date:
    """The date a run should treat as "now".

    ``None`` means live, so this is today. Anything else is a historical run
    (a backtest, or re-deciding a past session), and every data source must
    agree on which day that is - otherwise prices come from one date and the
    calendar from another, which is the subtlest way to leak the future into a
    decision.
    """
    raw = (config or {}).get("as_of")
    return parse_date(raw) if raw else today_ist()


def as_of_moment(config: dict | None = None) -> datetime | None:
    """The historical instant a run is pinned to, or ``None`` when live.

    Pinned to that session's close: a dated run decides on bars that are
    complete, never on a session still trading. The walk-forward harness then
    settles the decision at the *next* close, so the bar it is scored against
    is one the decision could not have seen.
    """
    raw = (config or {}).get("as_of")
    if not raw:
        return None
    # Imported here: calendar_in imports this module, so a top-level import
    # would be a cycle.
    from tradingagents_in.dataflows.calendar_in import MARKET_CLOSE  # noqa: PLC0415

    return combine_ist(parse_date(raw), MARKET_CLOSE)


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Last occurrence of `weekday` (Mon=0) in a month - used for monthly expiry."""
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    day = first_next - timedelta(days=1)
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day
