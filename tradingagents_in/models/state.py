"""The LangGraph state object threaded through every node.

Kept as a TypedDict (not a Pydantic model) because LangGraph merges partial
dict updates returned by nodes. The *values* are Pydantic models, so each node
still gets a validated contract.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from tradingagents_in.models.analyst import (
    FundamentalsReport,
    NewsMacroReport,
    SentimentReport,
    TechnicalReport,
)
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.models.research import DebateState, InvestmentPlan
from tradingagents_in.models.risk import (
    MarketMicrostructure,
    PortfolioDecision,
    PortfolioSnapshot,
    RiskAssessment,
)
from tradingagents_in.models.trade import TradeProposal


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer so parallel analyst nodes can each contribute to one dict."""
    return {**(left or {}), **(right or {})}


def append_list(left: list | None, right: list | None) -> list:
    return [*(left or []), *(right or [])]


class TradingState(TypedDict, total=False):
    # ------------------------------------------------------------- inputs
    symbol: str
    symbol_ref: SymbolRef
    trade_date: str
    config: dict[str, Any]
    run_id: str
    portfolio: PortfolioSnapshot

    # -------------------------------------------------- raw data bundles
    # Populated by the data-gathering node; analysts read from here so each
    # source is fetched exactly once per run.
    market_bundle: dict[str, Any]

    # ---------------------------------------------------- analyst layer
    fundamentals_report: FundamentalsReport
    sentiment_report: SentimentReport
    news_macro_report: NewsMacroReport
    technical_report: TechnicalReport

    # ---------------------------------------------------- research layer
    debate: DebateState
    investment_plan: InvestmentPlan

    # ------------------------------------------------------- trade layer
    trade_proposal: TradeProposal
    microstructure: MarketMicrostructure

    # -------------------------------------------------------- risk layer
    risk_assessment: RiskAssessment
    risk_round: int
    portfolio_decision: PortfolioDecision

    # ----------------------------------------------------------- meta
    memory_context: str
    errors: Annotated[list[str], append_list]
    telemetry: Annotated[dict[str, Any], merge_dicts]


ANALYST_STATE_KEYS: dict[str, str] = {
    "fundamentals": "fundamentals_report",
    "sentiment": "sentiment_report",
    "news_macro": "news_macro_report",
    "technical": "technical_report",
}
