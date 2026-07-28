"""Trader-agent contracts: the actual proposed trade."""

from __future__ import annotations

from pydantic import Field, model_validator

from tradingagents_in.models.base import AgentOutput, ITAModel
from tradingagents_in.models.enums import (
    Action,
    Instrument,
    LegSide,
    OptionType,
    ProductType,
    TimeHorizon,
)


class EntryZone(ITAModel):
    """Traders in India habitually work a band, not a single print."""

    low: float
    high: float
    reference_price: float | None = None
    trigger: str | None = Field(
        None, description="Condition to act on, e.g. 'close above 2,940 on daily'"
    )

    @model_validator(mode="after")
    def _ordered(self) -> EntryZone:
        if self.low > self.high:
            self.low, self.high = self.high, self.low
        return self

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.low:,.2f} - {self.high:,.2f}"


class Target(ITAModel):
    price: float
    rationale: str | None = None
    book_pct: float = Field(
        50.0, ge=0.0, le=100.0, description="Portion of the position to exit here"
    )


class OptionLeg(ITAModel):
    side: LegSide
    option_type: OptionType
    strike: float
    expiry: str = Field(..., description="Expiry date, YYYY-MM-DD")
    lots: int = Field(1, ge=1)
    lot_size: int | None = None
    estimated_premium: float | None = Field(None, description="Per-share premium in INR")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.side.value} {self.lots}x {self.strike:g} {self.option_type.value} "
            f"({self.expiry})"
        )


class FuturesSpec(ITAModel):
    expiry: str
    lots: int = Field(1, ge=1)
    lot_size: int | None = None
    contract_label: str | None = Field(None, description="e.g. RELIANCE 25SEP FUT")


class CostEstimate(ITAModel):
    """Round-trip friction, because Indian charges materially change small-R trades."""

    brokerage_inr: float = 0.0
    stt_inr: float = 0.0
    exchange_txn_inr: float = 0.0
    sebi_charges_inr: float = 0.0
    stamp_duty_inr: float = 0.0
    gst_inr: float = 0.0
    total_inr: float = 0.0
    total_pct_of_turnover: float = 0.0


class TradeProposal(AgentOutput):
    """The trader's concrete, executable idea."""

    action: Action = Action.HOLD
    instrument: Instrument = Instrument.CASH
    product_type: ProductType = ProductType.CNC
    time_horizon: TimeHorizon = TimeHorizon.SWING

    entry_zone: EntryZone | None = None
    stop_loss: float | None = None
    targets: list[Target] = Field(default_factory=list, max_length=3)

    position_size_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Capital deployed as % of total capital"
    )
    quantity: int | None = Field(None, description="Shares (cash) - lots are on the leg specs")
    capital_deployed_inr: float | None = None
    risk_amount_inr: float | None = Field(
        None, description="Loss if the stop is hit, before slippage"
    )

    futures: FuturesSpec | None = None
    option_legs: list[OptionLeg] = Field(default_factory=list, max_length=4)
    option_strategy: str | None = Field(
        None, description="e.g. 'bull call spread', 'long CE', 'short strangle'"
    )

    reward_risk: float | None = None
    expected_holding_days: int | None = None
    invalidation: str = Field("", description="What makes this idea wrong, stated before entry")
    execution_notes: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Liquidity, slippage, circuit, expiry-day and square-off mechanics",
    )
    costs: CostEstimate | None = None
    alternatives_considered: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _coherent(self) -> TradeProposal:
        # A HOLD is not a position. Strip anything that implies one.
        if self.action == Action.HOLD:
            self.position_size_pct = 0.0
            self.quantity = None
            self.capital_deployed_inr = 0.0
            self.risk_amount_inr = 0.0
        # Cash-segment shorts are an intraday product in India; NRML/CNC is invalid.
        if self.action == Action.SHORT and self.instrument == Instrument.CASH:
            self.product_type = ProductType.MIS
            self.time_horizon = TimeHorizon.INTRADAY
        if self.instrument == Instrument.OPTIONS and not self.option_legs:
            self.instrument = Instrument.CASH
        return self

    @property
    def is_actionable(self) -> bool:
        return self.action != Action.HOLD and self.position_size_pct > 0

    def to_markdown(self) -> str:
        from tradingagents_in.utils.formatting import bullet_list, inr_plain

        if not self.is_actionable:
            return f"**TRADE: HOLD {self.symbol}** — {self.summary}"

        lines = [
            f"**TRADE: {self.action.value} {self.symbol}** via {self.instrument.value}"
            f" / {self.product_type.value} / {self.time_horizon.value}",
            "",
            f"- Entry: {self.entry_zone}",
        ]
        if self.stop_loss is not None:
            lines.append(f"- Stop-loss: {self.stop_loss:,.2f}")
        if self.targets:
            tgt = ", ".join(f"{t.price:,.2f} ({t.book_pct:.0f}%)" for t in self.targets)
            lines.append(f"- Targets: {tgt}")
        lines.append(
            f"- Size: {self.position_size_pct:.2f}% of capital"
            f" ({inr_plain(self.capital_deployed_inr)})"
        )
        lines.append(f"- Risk at stop: {inr_plain(self.risk_amount_inr)}")
        if self.reward_risk is not None:
            lines.append(f"- Reward:Risk = {self.reward_risk:.2f}")
        if self.futures:
            lines.append(f"- Futures: {self.futures.contract_label or self.futures.expiry}"
                         f" x {self.futures.lots} lot(s)")
        if self.option_legs:
            lines.append(f"- Option strategy: {self.option_strategy or 'custom'}")
            lines.extend(f"  - {leg}" for leg in self.option_legs)
        if self.costs:
            lines.append(f"- Est. round-trip cost: {inr_plain(self.costs.total_inr)}")

        lines += [
            "",
            f"_Invalidation:_ {self.invalidation}",
            "",
            "**Execution notes**",
            bullet_list(self.execution_notes),
        ]
        return "\n".join(lines)
