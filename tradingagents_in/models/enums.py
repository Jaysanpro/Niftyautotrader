"""Enumerations shared by every agent contract."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):  # noqa: UP042 - see below
    """str-backed enum so values serialise cleanly into LLM JSON schemas.

    Deliberately ``str, Enum`` rather than :class:`enum.StrEnum`: the members
    must compare equal to plain strings for the JSON an LLM returns to validate
    without a custom coercion step, and this form behaves identically on 3.11
    and 3.12+.
    """

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return str(self.value)


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SHORT = "SHORT"


class Stance(StrEnum):
    BULLISH = "BULLISH"
    MILDLY_BULLISH = "MILDLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    MILDLY_BEARISH = "MILDLY_BEARISH"
    BEARISH = "BEARISH"


class Instrument(StrEnum):
    CASH = "CASH"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class LegSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class TimeHorizon(StrEnum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"  # a few days to a few weeks
    POSITIONAL = "POSITIONAL"  # weeks to months


class ProductType(StrEnum):
    """Broker product codes as Indian brokers label them."""

    CNC = "CNC"  # delivery / cash-and-carry
    MIS = "MIS"  # intraday, auto square-off
    NRML = "NRML"  # carry-forward F&O


class RiskVerdict(StrEnum):
    APPROVE = "APPROVE"
    APPROVE_WITH_CHANGES = "APPROVE_WITH_CHANGES"
    REJECT = "REJECT"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKER = "BLOCKER"


class AgentRole(StrEnum):
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    NEWS_MACRO_ANALYST = "news_macro_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    RESEARCH_MANAGER = "research_manager"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"


class MarketPhase(StrEnum):
    CLOSED = "CLOSED"
    PRE_OPEN = "PRE_OPEN"  # 09:00 - 09:15 IST
    REGULAR = "REGULAR"  # 09:15 - 15:30 IST
    CLOSING = "CLOSING"  # 15:40 - 16:00 IST post-close session
    HOLIDAY = "HOLIDAY"


class SurveillanceFlag(StrEnum):
    NONE = "NONE"
    ASM_SHORT_TERM = "ASM_SHORT_TERM"
    ASM_LONG_TERM = "ASM_LONG_TERM"
    GSM = "GSM"
    FNO_BAN = "FNO_BAN"
    CIRCUIT_LOCKED = "CIRCUIT_LOCKED"
