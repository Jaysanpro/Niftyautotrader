"""Bull/bear debate and research-manager contracts."""

from __future__ import annotations

from pydantic import Field

from tradingagents_in.models.base import AgentOutput, ITAModel
from tradingagents_in.models.enums import Stance, TimeHorizon


class DebatePoint(ITAModel):
    """A single argument. Every point must be falsifiable, not a slogan."""

    claim: str
    support: str = Field(..., description="The data or reasoning behind the claim")
    strength: float = Field(0.5, ge=0.0, le=1.0)
    india_specific: bool = Field(
        False, description="True if this hinges on SEBI/circuit/expiry/GST/monsoon mechanics"
    )
    source_agent: str | None = None


class DebateTurn(AgentOutput):
    """One researcher's turn in one round."""

    round_number: int = Field(1, ge=1)
    side: str = Field(..., description="bull | bear")
    thesis: str
    points: list[DebatePoint] = Field(default_factory=list, max_length=8)
    rebuttals: list[DebatePoint] = Field(
        default_factory=list,
        max_length=6,
        description="Direct responses to the opponent's previous turn",
    )
    conceded: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Opponent points accepted as valid. An empty list every round is a red flag.",
    )
    india_risk_focus: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="SEBI regulation, circuit filters, delivery vs intraday, F&O expiry, GST, ASM/GSM",
    )

    def to_markdown(self) -> str:
        from tradingagents_in.utils.formatting import bullet_list

        pts = bullet_list([f"{p.claim} — {p.support}" for p in self.points])
        rebs = bullet_list([f"{p.claim} — {p.support}" for p in self.rebuttals])
        return (
            f"**{self.side.upper()} · round {self.round_number}** "
            f"(confidence {self.effective_confidence():.0%})\n\n"
            f"_Thesis:_ {self.thesis}\n\n"
            f"**Points**\n{pts}\n\n"
            f"**Rebuttals**\n{rebs}\n\n"
            f"**Conceded**\n{bullet_list(self.conceded)}\n"
        )


class DebateState(ITAModel):
    """Accumulated debate transcript passed through the graph."""

    rounds_completed: int = 0
    max_rounds: int = 2
    turns: list[DebateTurn] = Field(default_factory=list)

    @property
    def bull_turns(self) -> list[DebateTurn]:
        return [t for t in self.turns if t.side == "bull"]

    @property
    def bear_turns(self) -> list[DebateTurn]:
        return [t for t in self.turns if t.side == "bear"]

    def last(self, side: str) -> DebateTurn | None:
        for turn in reversed(self.turns):
            if turn.side == side:
                return turn
        return None

    def transcript(self) -> str:
        if not self.turns:
            return "(no debate yet)"
        return "\n\n---\n\n".join(turn.to_markdown() for turn in self.turns)


class InvestmentPlan(AgentOutput):
    """Research manager's adjudication of the debate."""

    stance: Stance = Stance.NEUTRAL
    conviction: float = Field(0.5, ge=0.0, le=1.0)
    winning_side: str = Field("neither", description="bull | bear | neither")
    verdict: str = Field(..., description="What the debate actually established")
    decisive_arguments: list[str] = Field(default_factory=list, max_length=6)
    dismissed_arguments: list[str] = Field(
        default_factory=list, max_length=6, description="Arguments rejected, with the reason"
    )
    preferred_horizon: TimeHorizon = TimeHorizon.SWING
    key_monitorables: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Dated or observable things that would revise this view",
    )
    unresolved_questions: list[str] = Field(default_factory=list, max_length=5)
    lessons_applied: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Prior decisions from trading memory that informed this call",
    )
