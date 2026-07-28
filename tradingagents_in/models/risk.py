"""Risk-management and portfolio-manager contracts."""

from __future__ import annotations

from pydantic import Field

from tradingagents_in.models.base import AgentOutput, ITAModel
from tradingagents_in.models.enums import (
    Action,
    RiskVerdict,
    Severity,
    SurveillanceFlag,
    TimeHorizon,
)
from tradingagents_in.models.trade import TradeProposal


class RuleBreach(ITAModel):
    """A named, machine-checked rule violation. Deterministic, not LLM opinion."""

    rule: str
    severity: Severity = Severity.WARNING
    detail: str
    observed: str | None = None
    limit: str | None = None
    remedy: str | None = None


class MarketMicrostructure(ITAModel):
    """India-specific tradability checks the risk manager must see."""

    circuit_band_pct: float | None = Field(
        None, description="Applicable price band: 2 / 5 / 10 / 20 %, or None for F&O stocks"
    )
    upper_circuit: float | None = None
    lower_circuit: float | None = None
    is_circuit_locked: bool = False
    surveillance: SurveillanceFlag = SurveillanceFlag.NONE
    in_fno_ban: bool = False
    avg_daily_turnover_inr: float | None = None
    avg_daily_volume: float | None = None
    impact_cost_estimate_pct: float | None = None
    is_expiry_day: bool = False
    days_to_expiry: int | None = None
    market_phase: str | None = None
    index_circuit_note: str | None = None


class PortfolioSnapshot(ITAModel):
    """Existing book state used for concentration and heat checks."""

    capital_inr: float = 0.0
    deployed_inr: float = 0.0
    cash_inr: float = 0.0
    open_positions: int = 0
    portfolio_heat_pct: float = Field(0.0, description="Sum of open risk as % of capital")
    current_drawdown_pct: float = 0.0
    sector_exposure_pct: dict[str, float] = Field(default_factory=dict)
    symbol_exposure_pct: dict[str, float] = Field(default_factory=dict)
    overnight_fno_exposure_pct: float = 0.0


class RiskAssessment(AgentOutput):
    """Risk manager's verdict on a specific proposal."""

    verdict: RiskVerdict = RiskVerdict.APPROVE
    breaches: list[RuleBreach] = Field(default_factory=list)
    microstructure: MarketMicrostructure = Field(default_factory=MarketMicrostructure)

    approved_position_size_pct: float = Field(0.0, ge=0.0, le=100.0)
    approved_quantity: int | None = None
    approved_lots: int | None = None
    max_loss_inr: float | None = None
    max_loss_pct_of_capital: float | None = None
    adjusted_stop_loss: float | None = None

    sizing_rationale: str = ""
    risk_notes: list[str] = Field(default_factory=list, max_length=10)
    sebi_compliance_notes: list[str] = Field(default_factory=list, max_length=8)
    required_changes: list[str] = Field(
        default_factory=list, max_length=6, description="Must-fix items for APPROVE_WITH_CHANGES"
    )
    stress_test: dict[str, float] = Field(
        default_factory=dict,
        description="Scenario -> P&L in INR, e.g. {'gap_down_5pct': -18500}",
    )

    @property
    def blocking(self) -> list[RuleBreach]:
        return [b for b in self.breaches if b.severity == Severity.BLOCKER]


class PortfolioDecision(AgentOutput):
    """Final call. This is what the framework returns to the caller."""

    approved: bool = False
    final_action: Action = Action.HOLD
    final_position_size_pct: float = Field(0.0, ge=0.0, le=100.0)
    capital_deployed_inr: float = 0.0
    capital_at_risk_inr: float = 0.0
    time_horizon: TimeHorizon = TimeHorizon.SWING

    portfolio_reasoning: str = Field(
        "", description="Why this fits (or does not fit) the book as a whole"
    )
    risk_notes: list[str] = Field(default_factory=list, max_length=10)
    conditions: list[str] = Field(
        default_factory=list, max_length=8, description="Preconditions that must hold at execution"
    )
    review_triggers: list[str] = Field(
        default_factory=list, max_length=6, description="Events that force a re-review"
    )
    rejection_reasons: list[str] = Field(default_factory=list, max_length=6)
    dissent: str | None = Field(
        None, description="Where the PM overrode the trader or risk manager, and why"
    )


class FinalDecision(ITAModel):
    """Full run record: every agent artefact plus the adjudicated outcome."""

    symbol: str
    yf_symbol: str
    exchange: str = "NSE"
    company_name: str | None = None
    sector: str | None = None
    as_of: str = ""
    run_id: str = ""
    config_digest: dict = Field(default_factory=dict)

    # Headline outcome
    final_action: Action = Action.HOLD
    approved: bool = False
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    position_size_pct: float = 0.0
    capital_at_risk_inr: float = 0.0
    risk_notes: list[str] = Field(default_factory=list)

    # Artefacts
    analyst_reports: dict = Field(default_factory=dict)
    debate_transcript: str = ""
    investment_plan: dict = Field(default_factory=dict)
    trade_proposal: dict = Field(default_factory=dict)
    risk_assessment: dict = Field(default_factory=dict)
    portfolio_decision: dict = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float | None = None

    # Convenience accessors used by the trade proposal in the CLI/reporting layer
    proposal: TradeProposal | None = None

    def headline(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return (
            f"{self.symbol}: {self.final_action.value} [{status}] "
            f"size={self.position_size_pct:.2f}% conf={self.confidence:.0%}"
        )
