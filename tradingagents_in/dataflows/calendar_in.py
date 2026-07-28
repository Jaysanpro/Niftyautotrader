"""NSE/BSE trading calendar: session phases, holidays, F&O expiry.

Session timings (IST):
    09:00 - 09:15  pre-open call auction
    09:15 - 15:30  continuous (normal market)
    15:30 - 15:40  closing-price computation
    15:40 - 16:00  post-close session

Holidays are the published trading holidays for the equity segment. Muhurat
trading (a ~1 hour Diwali session) is deliberately *not* modelled as a normal
session - it is a ceremonial session and no strategy here should key off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from tradingagents_in.config.universe import WEEKLY_EXPIRY_WEEKDAY
from tradingagents_in.models.enums import MarketPhase
from tradingagents_in.utils.timeutils import (
    IST,
    combine_ist,
    now_ist,
    parse_date,
    to_ist,
)

PRE_OPEN_START = time(9, 0)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
CLOSING_SESSION_START = time(15, 40)
CLOSING_SESSION_END = time(16, 0)

# Published equity-segment trading holidays. Weekends are handled separately.
# Refresh annually from https://www.nseindia.com/resources/exchange-communication-holidays
TRADING_HOLIDAYS: dict[int, tuple[str, ...]] = {
    2024: (
        "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11",
        "2024-04-17", "2024-05-01", "2024-05-20", "2024-06-17", "2024-07-17",
        "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15", "2024-12-25",
    ),
    2025: (
        "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
        "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
        "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    ),
    2026: (
        "2026-01-01", "2026-01-26", "2026-03-04", "2026-03-19", "2026-03-26",
        "2026-04-01", "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-27",
        "2026-08-26", "2026-10-02", "2026-11-10", "2026-11-11", "2026-12-25",
    ),
}


def _holiday_set(year: int) -> set[date]:
    return {parse_date(d) for d in TRADING_HOLIDAYS.get(year, ())}


def is_holiday(day: date) -> bool:
    return day in _holiday_set(day.year)


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def is_trading_day(day: date) -> bool:
    return not is_weekend(day) and not is_holiday(day)


def has_calendar_coverage(day: date) -> bool:
    """False once we run past the last year of loaded holidays.

    Callers surface this so a stale calendar shows up as a data-quality note
    rather than a silently wrong expiry date.
    """
    return day.year in TRADING_HOLIDAYS


def previous_trading_day(day: date | None = None) -> date:
    cursor = (day or now_ist().date()) - timedelta(days=1)
    for _ in range(30):
        if is_trading_day(cursor):
            return cursor
        cursor -= timedelta(days=1)
    return cursor


def next_trading_day(day: date | None = None) -> date:
    cursor = (day or now_ist().date()) + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(cursor):
            return cursor
        cursor += timedelta(days=1)
    return cursor


def trading_days_between(start: date, end: date) -> int:
    if end < start:
        return -trading_days_between(end, start)
    count, cursor = 0, start
    while cursor < end:
        cursor += timedelta(days=1)
        if is_trading_day(cursor):
            count += 1
    return count


def market_phase(when: datetime | None = None) -> MarketPhase:
    moment = to_ist(when) if when else now_ist()
    day = moment.date()
    if is_weekend(day):
        return MarketPhase.CLOSED
    if is_holiday(day):
        return MarketPhase.HOLIDAY
    clock = moment.time()
    if PRE_OPEN_START <= clock < MARKET_OPEN:
        return MarketPhase.PRE_OPEN
    if MARKET_OPEN <= clock <= MARKET_CLOSE:
        return MarketPhase.REGULAR
    if CLOSING_SESSION_START <= clock <= CLOSING_SESSION_END:
        return MarketPhase.CLOSING
    return MarketPhase.CLOSED


def is_market_open(when: datetime | None = None) -> bool:
    return market_phase(when) == MarketPhase.REGULAR


def minutes_to_close(when: datetime | None = None) -> int | None:
    """Minutes left in the continuous session, or None outside it."""
    moment = to_ist(when) if when else now_ist()
    if market_phase(moment) != MarketPhase.REGULAR:
        return None
    close = combine_ist(moment.date(), MARKET_CLOSE)
    return max(0, int((close - moment).total_seconds() // 60))


def session_bounds(day: date) -> tuple[datetime, datetime]:
    return combine_ist(day, MARKET_OPEN), combine_ist(day, MARKET_CLOSE)


# ---------------------------------------------------------------------------
# F&O expiry
# ---------------------------------------------------------------------------
def _roll_back_for_holiday(day: date) -> date:
    """Expiry moves to the previous trading day when it lands on a holiday."""
    guard = 0
    while not is_trading_day(day) and guard < 10:
        day -= timedelta(days=1)
        guard += 1
    return day


def weekly_expiry(day: date | None = None, exchange: str = "NSE") -> date:
    """Next weekly expiry on or after `day` for the given exchange.

    SEBI permits one weekly expiry per exchange: NSE settled on Thursday,
    BSE on Tuesday. Holiday collisions roll the expiry back a day.
    """
    cursor = day or now_ist().date()
    target = WEEKLY_EXPIRY_WEEKDAY.get(exchange.upper(), 3)
    ahead = (target - cursor.weekday()) % 7
    candidate = _roll_back_for_holiday(cursor + timedelta(days=ahead))
    if candidate < cursor:
        candidate = _roll_back_for_holiday(cursor + timedelta(days=ahead + 7))
    return candidate


def monthly_expiry(year: int, month: int, exchange: str = "NSE") -> date:
    """Last weekly-expiry weekday of the month, rolled back for holidays."""
    target = WEEKLY_EXPIRY_WEEKDAY.get(exchange.upper(), 3)
    first_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    cursor = first_next - timedelta(days=1)
    while cursor.weekday() != target:
        cursor -= timedelta(days=1)
    return _roll_back_for_holiday(cursor)


def next_monthly_expiry(day: date | None = None, exchange: str = "NSE") -> date:
    cursor = day or now_ist().date()
    this_month = monthly_expiry(cursor.year, cursor.month, exchange)
    if this_month >= cursor:
        return this_month
    nxt = cursor.replace(day=1) + timedelta(days=32)
    return monthly_expiry(nxt.year, nxt.month, exchange)


def is_expiry_day(day: date | None = None, exchange: str = "NSE") -> bool:
    cursor = day or now_ist().date()
    return weekly_expiry(cursor, exchange) == cursor


def days_to_expiry(day: date | None = None, exchange: str = "NSE", monthly: bool = False) -> int:
    cursor = day or now_ist().date()
    expiry = next_monthly_expiry(cursor, exchange) if monthly else weekly_expiry(cursor, exchange)
    return trading_days_between(cursor, expiry)


@dataclass(frozen=True)
class SessionContext:
    """Everything time-related an agent needs, computed once per run."""

    now: datetime
    trade_date: date
    phase: MarketPhase
    is_trading_day: bool
    minutes_to_close: int | None
    weekly_expiry: date
    monthly_expiry: date
    is_expiry_day: bool
    days_to_weekly_expiry: int
    days_to_monthly_expiry: int
    last_trading_day: date
    calendar_covered: bool

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"- Now: {self.now:%Y-%m-%d %H:%M} IST ({self.phase.value})",
                f"- Trade date: {self.trade_date}"
                f"{'' if self.is_trading_day else ' (non-trading day)'}",
                f"- Last completed session: {self.last_trading_day}",
                f"- Minutes to close: {self.minutes_to_close if self.minutes_to_close is not None else 'n/a'}",
                f"- Weekly expiry: {self.weekly_expiry} "
                f"({self.days_to_weekly_expiry} sessions away)"
                f"{' — TODAY' if self.is_expiry_day else ''}",
                f"- Monthly expiry: {self.monthly_expiry} "
                f"({self.days_to_monthly_expiry} sessions away)",
            ]
            + ([] if self.calendar_covered else
               ["- WARNING: holiday calendar has no data for this year; expiry dates are approximate"])
        )


def session_context(when: datetime | None = None, exchange: str = "NSE") -> SessionContext:
    moment = to_ist(when) if when else now_ist()
    day = moment.date()
    phase = market_phase(moment)
    trading = is_trading_day(day)
    # "Last completed session" only counts today once today's close has passed;
    # during pre-open and the live session the newest complete daily bar is
    # still yesterday's.
    session_finished = trading and moment.time() >= MARKET_CLOSE
    last = day if session_finished else previous_trading_day(day)
    return SessionContext(
        now=moment,
        trade_date=day,
        phase=phase,
        is_trading_day=trading,
        minutes_to_close=minutes_to_close(moment),
        weekly_expiry=weekly_expiry(day, exchange),
        monthly_expiry=next_monthly_expiry(day, exchange),
        is_expiry_day=trading and is_expiry_day(day, exchange),
        days_to_weekly_expiry=days_to_expiry(day, exchange, monthly=False),
        days_to_monthly_expiry=days_to_expiry(day, exchange, monthly=True),
        last_trading_day=last,
        calendar_covered=has_calendar_coverage(day),
    )


__all__ = [
    "IST", "MARKET_CLOSE", "MARKET_OPEN", "MarketPhase", "SessionContext",
    "days_to_expiry", "is_expiry_day", "is_market_open", "is_trading_day",
    "market_phase", "minutes_to_close", "monthly_expiry", "next_monthly_expiry",
    "next_trading_day", "previous_trading_day", "session_bounds",
    "session_context", "trading_days_between", "weekly_expiry",
]
