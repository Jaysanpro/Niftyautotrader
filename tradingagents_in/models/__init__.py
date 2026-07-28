"""Pydantic contracts for every agent output in the framework."""

from tradingagents_in.models.analyst import (
    AnalystReport,
    FlowSnapshot,
    FundamentalsReport,
    LevelSet,
    MacroIndicator,
    NewsMacroReport,
    PeerMetric,
    QuarterlyResult,
    SentimentReport,
    SentimentSource,
    ShareholdingPattern,
    TechnicalReport,
    TimeframeSignal,
)
from tradingagents_in.models.base import (
    AgentOutput,
    DataQuality,
    Evidence,
    ITAModel,
    SymbolRef,
)
from tradingagents_in.models.enums import (
    Action,
    AgentRole,
    Exchange,
    Instrument,
    LegSide,
    MarketPhase,
    OptionType,
    ProductType,
    RiskVerdict,
    Severity,
    Stance,
    SurveillanceFlag,
    TimeHorizon,
)
from tradingagents_in.models.research import (
    DebatePoint,
    DebateState,
    DebateTurn,
    InvestmentPlan,
)
from tradingagents_in.models.risk import (
    FinalDecision,
    MarketMicrostructure,
    PortfolioDecision,
    PortfolioSnapshot,
    RiskAssessment,
    RuleBreach,
)
from tradingagents_in.models.state import TradingState
from tradingagents_in.models.trade import (
    CostEstimate,
    EntryZone,
    FuturesSpec,
    OptionLeg,
    Target,
    TradeProposal,
)

__all__ = [
    "Action", "AgentOutput", "AgentRole", "AnalystReport", "CostEstimate",
    "DataQuality", "DebatePoint", "DebateState", "DebateTurn", "EntryZone",
    "Evidence", "Exchange", "FinalDecision", "FlowSnapshot", "FundamentalsReport",
    "FuturesSpec", "ITAModel", "Instrument", "InvestmentPlan", "LegSide",
    "LevelSet", "MacroIndicator", "MarketMicrostructure", "MarketPhase",
    "NewsMacroReport", "OptionLeg", "OptionType", "PeerMetric",
    "PortfolioDecision", "PortfolioSnapshot", "ProductType", "QuarterlyResult",
    "RiskAssessment", "RiskVerdict", "RuleBreach", "SentimentReport",
    "SentimentSource", "Severity", "ShareholdingPattern", "Stance",
    "SurveillanceFlag", "SymbolRef", "Target", "TechnicalReport",
    "TimeHorizon", "TimeframeSignal", "TradeProposal", "TradingState",
]
