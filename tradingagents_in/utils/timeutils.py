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
