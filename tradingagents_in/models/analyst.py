"""Analyst-team output contracts."""

from __future__ import annotations

from pydantic import Field

from tradingagents_in.models.base import AgentOutput, ITAModel
from tradingagents_in.models.enums import Stance


class AnalystReport(AgentOutput):
    """Common shape for all four analysts."""

    stance: Stance = Stance.NEUTRAL
    key_points: list[str] = Field(default_factory=list, max_length=12)
    risks: list[str] = Field(default_factory=list, max_length=10)
    what_would_change_my_mind: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Concrete, observable events that would flip this stance",
    )

    def to_markdown(self) -> str:
        from tradingagents_in.utils.formatting import bullet_list

        return (
            f"**{self.role.value.replace('_', ' ').title()} - {self.symbol}**  \n"
            f"Stance: `{self.stance.value}` | Confidence: {self.effective_confidence():.0%}\n\n"
            f"{self.summary}\n\n"
            f"**Key points**\n{bullet_list(self.key_points)}\n\n"
            f"**Risks**\n{bullet_list(self.risks)}\n"
        )


# ---------------------------------------------------------------- fundamentals
class PeerMetric(ITAModel):
    symbol: str
    name: str | None = None
    pe: float | None = None
    pb: float | None = None
    roe_pct: float | None = None
    debt_to_equity: float | None = None
    market_cap_inr: float | None = None
    revenue_growth_yoy_pct: float | None = None


class QuarterlyResult(ITAModel):
    """One reported quarter under Ind-AS."""

    period: str = Field(..., description="e.g. 'Q2 FY26' or '2025-09-30'")
    revenue_inr: float | None = None
    ebitda_inr: float | None = None
    ebitda_margin_pct: float | None = None
    pat_inr: float | None = None
    eps: float | None = None
    revenue_yoy_pct: float | None = None
    pat_yoy_pct: float | None = None
    exceptional_items_inr: float | None = None
    is_consolidated: bool = True


class ShareholdingPattern(ITAModel):
    promoter_pct: float | None = None
    promoter_pledge_pct: float | None = None
    fii_pct: float | None = None
    dii_pct: float | None = None
    public_pct: float | None = None
    promoter_change_qoq_pct: float | None = None
    fii_change_qoq_pct: float | None = None


class FundamentalsReport(AnalystReport):
    """Ind-AS flavoured fundamental view."""

    market_cap_inr: float | None = None
    pe: float | None = None
    forward_pe: float | None = None
    pb: float | None = None
    ev_to_ebitda: float | None = None
    dividend_yield_pct: float | None = None
    roe_pct: float | None = None
    roce_pct: float | None = None
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    current_ratio: float | None = None
    operating_cash_flow_inr: float | None = None
    free_cash_flow_inr: float | None = None
    ocf_to_pat: float | None = Field(
        None, description="Cash conversion. Persistently <0.7 is an earnings-quality flag."
    )
    sales_growth_3y_cagr_pct: float | None = None
    profit_growth_3y_cagr_pct: float | None = None

    shareholding: ShareholdingPattern = Field(default_factory=ShareholdingPattern)
    recent_quarters: list[QuarterlyResult] = Field(default_factory=list, max_length=8)
    peers: list[PeerMetric] = Field(default_factory=list, max_length=8)

    valuation_verdict: str = Field(
        "", description="Cheap / fair / expensive versus own history and peers, with the number"
    )
    earnings_quality_notes: list[str] = Field(default_factory=list, max_length=6)
    governance_flags: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Pledge, related-party, auditor change, promoter selling, etc.",
    )


# ------------------------------------------------------------------ sentiment
class SentimentSource(ITAModel):
    source: str
    score: float = Field(0.0, ge=-1.0, le=1.0)
    volume: int = Field(0, ge=0, description="Number of items sampled")
    notes: str | None = None


class SentimentReport(AnalystReport):
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0)
    buzz_level: str = Field("normal", description="quiet | normal | elevated | frenzied")
    retail_vs_institutional: str = Field(
        "", description="Where the enthusiasm is concentrated and why that matters"
    )
    source_breakdown: list[SentimentSource] = Field(default_factory=list)
    notable_headlines: list[str] = Field(default_factory=list, max_length=10)
    contrarian_signal: str | None = Field(
        None, description="Set when crowding looks like a fade rather than a follow"
    )


# ---------------------------------------------------------------- news & macro
class MacroIndicator(ITAModel):
    name: str
    value: float | None = None
    change_pct: float | None = None
    direction: str | None = None
    implication_for_india: str | None = None


class FlowSnapshot(ITAModel):
    """FII/DII cash-segment flows in INR crore."""

    date: str | None = None
    fii_net_inr_cr: float | None = None
    dii_net_inr_cr: float | None = None
    fii_5d_net_inr_cr: float | None = None
    dii_5d_net_inr_cr: float | None = None
    fii_fno_index_futures_net_inr_cr: float | None = None
    interpretation: str | None = None


class NewsMacroReport(AnalystReport):
    macro_regime: str = Field("", description="risk-on / risk-off / range-bound, and the driver")
    rbi_policy_note: str = ""
    fiscal_policy_note: str = Field("", description="Union Budget, GST, PLI, capex cycle")
    monsoon_note: str = Field("", description="IMD rainfall vs LPA and rural-demand read-through")
    flows: FlowSnapshot = Field(default_factory=FlowSnapshot)
    global_cues: list[MacroIndicator] = Field(default_factory=list)
    india_indicators: list[MacroIndicator] = Field(default_factory=list)
    stock_specific_news: list[str] = Field(default_factory=list, max_length=12)
    upcoming_events: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Dated catalysts: results, policy, expiry, budget",
    )
    geopolitical_note: str = ""


# ------------------------------------------------------------------ technical
class LevelSet(ITAModel):
    supports: list[float] = Field(default_factory=list, max_length=5)
    resistances: list[float] = Field(default_factory=list, max_length=5)
    pivot: float | None = None
    r1: float | None = None
    r2: float | None = None
    s1: float | None = None
    s2: float | None = None


class TimeframeSignal(ITAModel):
    timeframe: str = Field(..., description="1d / 1h / 15m")
    trend: str = Field("sideways", description="uptrend | downtrend | sideways")
    close: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_percent_b: float | None = None
    supertrend: float | None = None
    supertrend_direction: str | None = Field(None, description="long | short")
    vwap: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    adx_14: float | None = None
    volume_ratio_20: float | None = Field(
        None, description="Latest volume divided by its 20-period average"
    )
    levels: LevelSet = Field(default_factory=LevelSet)
    signal: str = Field("neutral", description="bullish | bearish | neutral for this timeframe")
    notes: list[str] = Field(default_factory=list, max_length=6)


class TechnicalReport(AnalystReport):
    last_price: float | None = None
    timeframes: list[TimeframeSignal] = Field(default_factory=list)
    primary_timeframe: str = "1d"
    multi_timeframe_alignment: str = Field(
        "", description="Do the timeframes agree? Disagreement is itself a signal."
    )
    patterns: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Chart/candle patterns incl. India-specific ones (gap-up open, expiry pin, circuit move)",
    )
    relative_strength_vs_nifty_pct: float | None = None
    delivery_pct: float | None = Field(
        None, description="Delivery volume as % of traded volume - conviction proxy in cash segment"
    )
    suggested_stop_loss: float | None = None
    suggested_targets: list[float] = Field(default_factory=list, max_length=3)
    volatility_regime: str = Field("normal", description="low | normal | high | extreme")
