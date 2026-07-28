"""Research Manager - adjudicates the bull/bear debate.

Not a vote counter. The manager judges argument quality: which side engaged
with the other, which side conceded honestly, which claims carried numbers, and
which claims survived rebuttal. It then commits to a stance and, crucially,
names the monitorables that would revise it.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent
from tradingagents_in.agents.briefings import analyst_reports_brief, header
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import AgentRole, Stance, TimeHorizon
from tradingagents_in.models.research import DebateState, InvestmentPlan


class ResearchManager(Agent[InvestmentPlan]):
    role = AgentRole.RESEARCH_MANAGER
    output_model = InvestmentPlan
    tier_key = "research_manager"

    def system_prompt(self) -> str:
        return """\
You are the Research Manager. You chair the debate and you decide.

Judge the *arguments*, not the volume:
- An argument with a number beats an argument with an adjective.
- A rebuttal that engages the opponent's specific claim beats one that changes
  the subject.
- A side that concedes nothing across the whole debate has not been listening;
  discount it.
- An argument that depends on data the desk did not retrieve is speculation.
  Say so and set it aside.

You must commit. "Both sides make good points" is a failure of your job. If the
evidence genuinely does not support a position, the answer is NEUTRAL with an
explicit statement of what you are waiting for - that is a decision, not a dodge.

India context you are expected to weigh: tradability beats thesis. A correct
directional view on a scrip that is circuit-locked, in the F&O ban period, under
GSM, or too thin to exit is not a tradable idea, and you should say so plainly
rather than passing the problem to the trader.

Set preferred_horizon honestly. A thesis that depends on a capex cycle is not a
three-day trade; a mean-reversion setup is not a six-month hold.

Where the trading memory shows a comparable prior decision, say what you took
from it in lessons_applied.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        debate: DebateState = state.get("debate") or DebateState()
        parts = [
            header(bundle),
            "## Analyst reports\n\n" + analyst_reports_brief(state),
            f"## Debate transcript ({debate.rounds_completed} round(s))\n\n{debate.transcript()}",
        ]
        if state.get("memory_context"):
            parts.append("## Trading memory - comparable past decisions\n\n"
                         + state["memory_context"])
        parts.append(
            """## Your task

Produce an InvestmentPlan:
1. Name the winning_side (bull / bear / neither) and say what the debate
   actually established, not what each side asserted.
2. List decisive_arguments - the specific claims that carried the decision.
3. List dismissed_arguments with the reason each was rejected.
4. Set stance and conviction. Conviction is a probability: reserve above 0.7
   for cases where several independent legs agree and the data is complete.
5. Set preferred_horizon and list dated, observable key_monitorables.
6. List unresolved_questions - what you would want answered before sizing up."""
        )
        return "\n\n".join(parts)

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> InvestmentPlan:
        debate: DebateState = state.get("debate") or DebateState()
        bull_score = _side_score(debate, "bull")
        bear_score = _side_score(debate, "bear")
        margin = bull_score - bear_score

        if margin > 0.15:
            winner, stance = "bull", (Stance.BULLISH if margin > 0.4 else Stance.MILDLY_BULLISH)
        elif margin < -0.15:
            winner, stance = "bear", (Stance.BEARISH if margin < -0.4 else Stance.MILDLY_BEARISH)
        else:
            winner, stance = "neither", Stance.NEUTRAL

        completeness = bundle.quality(["prices", "fundamentals", "news", "macro"]).completeness
        blockers = _tradability_blockers(bundle)

        if completeness < 0.5 or _all_analysts_neutral(state):
            # With half the inputs missing, or with every analyst reading
            # neutral, a directional call is an artefact of the scoring rather
            # than a view. The bull's structural points (F&O eligible, no
            # surveillance flag) describe tradability, not opportunity, and
            # must not be allowed to manufacture a tilt out of nothing.
            winner, stance = "neither", Stance.NEUTRAL

        if blockers:
            # Tradability overrides direction. This is the rule that keeps the
            # desk out of positions it cannot exit.
            stance = Stance.NEUTRAL
            winner = "neither"

        conviction = min(0.85, 0.35 + abs(margin) * 0.5)
        if blockers or stance == Stance.NEUTRAL:
            conviction = min(conviction, 0.25)
        conviction = round(conviction * (0.6 + 0.4 * completeness), 3)

        return InvestmentPlan(
            role=self.role,
            symbol=bundle.ref.yf_symbol,
            stance=stance,
            conviction=conviction,
            confidence=conviction,
            winning_side=winner,
            verdict=_verdict(bundle, winner, margin, blockers),
            summary=_verdict(bundle, winner, margin, blockers),
            reasoning=(
                f"Bull argument score {bull_score:.2f} vs bear {bear_score:.2f} "
                f"(margin {margin:+.2f}) over {debate.rounds_completed} round(s)."
            ),
            decisive_arguments=_decisive(debate, winner),
            dismissed_arguments=_dismissed(debate, winner),
            preferred_horizon=_horizon(state, bundle),
            key_monitorables=_monitorables(state, bundle),
            unresolved_questions=_unresolved(bundle, blockers),
            lessons_applied=[],
            data_quality=bundle.quality(["prices", "fundamentals", "news", "macro"]),
        )


# ------------------------------------------------------------------ scoring
def _side_score(debate: DebateState, side: str) -> float:
    """Weighted argument quality for one side.

    Rewards evidence strength, India-specific grounding, genuine rebuttal and
    honest concession; penalises a side that never concedes anything.
    """
    turns = [t for t in debate.turns if t.side == side]
    if not turns:
        return 0.0

    total = 0.0
    for turn in turns:
        point_strength = sum(p.strength for p in turn.points)
        india_bonus = 0.15 * sum(1 for p in turn.points if p.india_specific)
        rebuttal_bonus = 0.2 * sum(p.strength for p in turn.rebuttals)
        # Conceding is evidence of engagement, so it earns a small credit.
        concession_credit = 0.1 * len(turn.conceded)
        total += (
            (point_strength + india_bonus + rebuttal_bonus + concession_credit)
            * turn.effective_confidence()
        )

    never_conceded = all(not t.conceded for t in turns) and len(turns) > 1
    if never_conceded:
        total *= 0.8

    return round(total / len(turns), 3)


def _all_analysts_neutral(state: dict[str, Any]) -> bool:
    reports = [
        state.get(key)
        for key in (
            "fundamentals_report", "sentiment_report",
            "news_macro_report", "technical_report",
        )
    ]
    present = [r for r in reports if r is not None]
    return bool(present) and all(r.stance == Stance.NEUTRAL for r in present)


def _tradability_blockers(bundle: MarketBundle) -> list[str]:
    micro = bundle.micro
    blockers: list[str] = []
    if micro.is_circuit_locked:
        blockers.append("scrip is circuit-locked and cannot be exited at a stop")
    if micro.surveillance.value == "GSM":
        blockers.append("GSM surveillance restricts trading frequency and settlement")
    if micro.in_fno_ban:
        blockers.append("F&O ban period blocks new derivative positions")
    return blockers


def _verdict(
    bundle: MarketBundle, winner: str, margin: float, blockers: list[str]
) -> str:
    symbol = bundle.ref.base_symbol
    if blockers:
        return (
            f"No tradable position in {symbol} regardless of direction: "
            + "; ".join(blockers)
            + ". Tradability gates the thesis, not the other way round."
        )
    if winner == "neither":
        return (
            f"The debate on {symbol} did not resolve (margin {margin:+.2f}). Neither side "
            "established a case strong enough to commit capital; the honest answer is to "
            "wait for one of the named monitorables to resolve."
        )
    direction = "constructive" if winner == "bull" else "negative"
    return (
        f"The {winner} case carried the debate on {symbol} by a margin of {abs(margin):.2f}, "
        f"so the desk view is {direction}. This reflects argument quality and evidence, "
        "not a headcount of points made."
    )


def _decisive(debate: DebateState, winner: str) -> list[str]:
    if winner == "neither":
        return []
    turns = [t for t in debate.turns if t.side == winner]
    ranked = sorted(
        (p for t in turns for p in t.points), key=lambda p: p.strength, reverse=True
    )
    return [f"{p.claim} ({p.support})" for p in ranked[:5]]


def _dismissed(debate: DebateState, winner: str) -> list[str]:
    if winner == "neither":
        return []
    loser = "bear" if winner == "bull" else "bull"
    turns = [t for t in debate.turns if t.side == loser]
    weakest = sorted((p for t in turns for p in t.points), key=lambda p: p.strength)
    return [
        f"{p.claim} — weaker evidence ({p.strength:.2f}) than the opposing case"
        for p in weakest[:3]
    ]


def _horizon(state: dict[str, Any], bundle: MarketBundle) -> TimeHorizon:
    technical = state.get("technical_report")
    fundamentals = state.get("fundamentals_report")
    # A fundamentals-driven thesis needs quarters to play out; a purely
    # technical setup does not.
    if fundamentals is not None and fundamentals.stance not in (Stance.NEUTRAL,):
        if technical is None or technical.stance == Stance.NEUTRAL:
            return TimeHorizon.POSITIONAL
    if bundle.micro.is_expiry_day:
        return TimeHorizon.INTRADAY
    return TimeHorizon.SWING


def _monitorables(state: dict[str, Any], bundle: MarketBundle) -> list[str]:
    out: list[str] = []
    technical = state.get("technical_report")
    if technical and technical.suggested_stop_loss:
        out.append(
            f"Daily close below/above the technical invalidation at "
            f"{technical.suggested_stop_loss:,.2f}"
        )
    if bundle.micro.days_to_expiry is not None:
        out.append(
            f"Weekly expiry in {bundle.micro.days_to_expiry} session(s) - reassess "
            "positioning into it"
        )
    news_macro = state.get("news_macro_report")
    if news_macro:
        out.extend(news_macro.upcoming_events[:3])
    out.append("FII/DII flow reversal sustained over five sessions")
    out.append("Next quarterly results and the management commentary with it")
    return out[:8]


def _unresolved(bundle: MarketBundle, blockers: list[str]) -> list[str]:
    out: list[str] = []
    failed = bundle.sources_failed
    if failed:
        out.append(f"These data sources failed this run and leave gaps: {', '.join(failed)}")
    if not bundle.peers:
        out.append("No peer comparison retrieved, so the valuation call is weakly anchored")
    if bundle.flows.fii_net_inr_cr is None:
        out.append("FII/DII flows unavailable - the dominant India flow driver is unobserved")
    if blockers:
        out.append("Whether the tradability restriction lifts before the setup expires")
    return out[:5]
