"""Bull and Bear researchers.

They see identical analyst reports and identical raw data. The only difference
is the mandate. Each round they must engage with the opponent's *previous*
turn - a turn that ignores the other side and simply restates its own thesis is
worthless, and the research manager is instructed to discount it.

Both sides must ground at least part of the case in India-specific mechanics:
SEBI rules, circuit filters, delivery vs intraday, F&O expiry dynamics, GST and
duty changes, promoter and pledge behaviour, ASM/GSM surveillance.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent
from tradingagents_in.agents.briefings import (
    analyst_reports_brief,
    header,
    microstructure_brief,
)
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import AgentRole, Stance
from tradingagents_in.models.research import DebatePoint, DebateState, DebateTurn

INDIA_RISK_CHECKLIST = """\
India-specific angles you are expected to consider (use the ones that actually
apply, ignore the rest - do not recite the list):
- SEBI regulation: position limits, disclosure requirements, surveillance actions,
  insider-trading windows around results.
- Circuit filters: a banded scrip that keeps hitting a limit cannot be exited at
  your stop. Index-level breakers (10/15/20%) halt everything.
- ASM / GSM surveillance: higher margins, trade-for-trade settlement, reduced
  liquidity. GSM stages can cut trading to once a week.
- F&O ban period: when a scrip crosses 95% of market-wide position limit, only
  position-reducing orders are allowed and new positions attract penalties.
- Expiry dynamics: weekly (Thursday NSE, Tuesday BSE) and monthly expiry pin
  prices toward high-OI strikes and distort the last hour.
- Delivery vs intraday: MIS positions are auto-squared-off by the broker; CNC
  needs full capital and T+1 settlement. A cash-segment short must be closed
  the same day.
- Taxation and charges: STT, stamp duty, GST on charges, and the difference
  between short-term and long-term capital gains treatment.
- Promoter behaviour: pledge, stake sale, preferential allotment, QIP dilution.
- Flow regime: FII selling absorbed by DII SIP inflows narrows breadth even
  while the index holds.
"""


class _Researcher(Agent[DebateTurn]):
    output_model = DebateTurn
    side: str

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        debate: DebateState = state.get("debate") or DebateState()
        round_number = debate.rounds_completed + 1
        opponent = "bear" if self.side == "bull" else "bull"
        last_opposing = debate.last(opponent)
        own_last = debate.last(self.side)

        parts = [
            header(bundle),
            "## Analyst reports\n\n" + analyst_reports_brief(state),
            microstructure_brief(bundle),
        ]

        if state.get("memory_context"):
            parts.append("## Lessons from previous decisions\n\n" + state["memory_context"])

        if last_opposing is not None:
            parts.append(
                f"## The {opponent} case you must answer (round "
                f"{last_opposing.round_number})\n\n{last_opposing.to_markdown()}"
            )
        if own_last is not None:
            parts.append(
                f"## Your own previous turn - do not simply repeat it\n\n{own_last.to_markdown()}"
            )

        if round_number == 1:
            task = f"""## Your task (round {round_number})

Open the {self.side} case. Build it from the analyst reports, cite specific
numbers, and include at least two India-specific arguments. State your thesis in
one sentence before the supporting points."""
        else:
            task = f"""## Your task (round {round_number})

This is a rebuttal round. You must:
1. Answer the {opponent}'s strongest specific point directly - by name, with a
   number. Vague dismissal counts as conceding.
2. Concede at least one point that genuinely holds. Conceding nothing across a
   whole debate is a tell that you are not reading the other side, and the
   research manager will discount your case for it.
3. Add at least one argument you have not made before. Do not restate round 1.
4. Keep at least one India-specific mechanic in play."""

        parts.append(task)
        return "\n\n".join(parts)

    def _round_number(self, state: dict[str, Any]) -> int:
        debate: DebateState = state.get("debate") or DebateState()
        return debate.rounds_completed + 1


class BullResearcher(_Researcher):
    role = AgentRole.BULL_RESEARCHER
    tier_key = "bull_researcher"
    side = "bull"

    def system_prompt(self) -> str:
        return f"""\
You are the Bull Researcher.

Your mandate is to build the strongest *honest* case for owning this stock. You
are an advocate, not a fantasist: an argument that cannot survive the bear's
scrutiny costs the desk money and costs you credibility.

Build the case on:
- Growth runway and the specific driver behind it (capacity, order book, pricing,
  market-share shift, a policy tailwind like PLI or a capex cycle).
- Valuation support: why the multiple is defensible against peers and history.
- Balance sheet and cash generation strength.
- Technical structure and the actual entry level, not just "momentum is strong".
- Flow and positioning: FII/DII behaviour, index inclusion, delivery percentage.

You must also name the India-specific mechanics that *help* this trade: F&O
eligibility giving you a hedge, high delivery percentage signalling genuine
accumulation, a liquid option chain letting you define risk.

{INDIA_RISK_CHECKLIST}

Never claim data you were not given. If the fundamental leg is missing, build
the case on what exists and say the fundamental support is unverified.
"""

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> DebateTurn:
        return _build_turn(self, bundle, state, side="bull")


class BearResearcher(_Researcher):
    role = AgentRole.BEAR_RESEARCHER
    tier_key = "bear_researcher"
    side = "bear"

    def system_prompt(self) -> str:
        return f"""\
You are the Bear Researcher.

Your mandate is to find the ways this trade loses money. Not to be reflexively
negative - to be specific about the mechanism of loss. "It could fall" is not an
argument; "at 2.4x D/E with interest coverage of 2.1x, a 100bp rate move takes
out a fifth of PAT" is.

Attack on:
- Earnings quality: cash conversion, receivables, other income, exceptionals.
- Valuation risk: what is priced in, and what happens on a single miss.
- Leverage, refinancing and covenant risk.
- Competitive and regulatory threat, including GST/duty changes and price controls.
- Technical damage: broken structure, distribution volume, negative divergence.
- Positioning: crowded retail longs, promoter selling, pledge, FII exit.

You must also name the India-specific mechanics that *hurt* this trade
concretely: a banded scrip you cannot exit at your stop, ASM/GSM margins, an
F&O ban blocking your hedge, expiry pinning, or a stock where intraday and
delivery volumes tell different stories.

{INDIA_RISK_CHECKLIST}

Never claim data you were not given. Concede real strengths - a bear case that
denies obvious quality is not credible.
"""

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> DebateTurn:
        return _build_turn(self, bundle, state, side="bear")


# --------------------------------------------------------------------------
# Deterministic debate construction (offline mode / LLM fallback)
# --------------------------------------------------------------------------
def _reports(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "fundamentals": state.get("fundamentals_report"),
        "sentiment": state.get("sentiment_report"),
        "news_macro": state.get("news_macro_report"),
        "technical": state.get("technical_report"),
    }


_BULLISH = (Stance.BULLISH, Stance.MILDLY_BULLISH)
_BEARISH = (Stance.BEARISH, Stance.MILDLY_BEARISH)


def _build_turn(
    agent: _Researcher, bundle: MarketBundle, state: dict[str, Any], side: str
) -> DebateTurn:
    reports = _reports(state)
    round_number = agent._round_number(state)
    supporting = _BULLISH if side == "bull" else _BEARISH
    opposing = _BEARISH if side == "bull" else _BULLISH

    points: list[DebatePoint] = []
    conceded: list[str] = []

    for name, report in reports.items():
        if report is None:
            continue
        aligned = report.stance in supporting
        # Draw arguments from the analysts that agree with this side, and
        # concede the ones that do not. That is what an honest debater does.
        # An aligned report can still contain individually contradictory
        # observations - a bullish technical read whose hourly frame is
        # bearish - and quoting one of those as support is how a debate starts
        # looking dishonest.
        source_points = [t for t in report.key_points if _supports(t, side)] if aligned else []
        for text in source_points[:2]:
            points.append(
                DebatePoint(
                    claim=text,
                    support=f"{name} analyst, stance {report.stance.value}, "
                            f"confidence {report.effective_confidence():.0%}",
                    strength=round(report.effective_confidence(), 2),
                    india_specific=False,
                    source_agent=name,
                )
            )

        # The bear must have a case even when no analyst is bearish. Every
        # report names its own risks; those are the bear's raw material, and a
        # debate with an empty bear side is not a debate.
        if side == "bear" and not aligned:
            for text in report.risks[:2]:
                points.append(
                    DebatePoint(
                        claim=text,
                        support=f"Risk named by the {name} analyst itself, against its own "
                                f"{report.stance.value} stance",
                        strength=round(0.6 * report.effective_confidence(), 2),
                        india_specific=_is_india_specific(text),
                        source_agent=name,
                    )
                )

        if report.stance in opposing:
            conceded.append(
                f"The {name} analyst reads {report.stance.value} "
                f"({report.summary[:140]})"
            )

    points.extend(_india_points(bundle, side))
    india_focus = [p.claim for p in points if p.india_specific][:5]

    if side == "bull":
        risks_source = reports.get("technical")
        rebuttal_seed = (risks_source.risks if risks_source else [])[:2]
    else:
        rebuttal_seed = []
        fundamentals = reports.get("fundamentals")
        if fundamentals:
            rebuttal_seed = fundamentals.governance_flags[:2] or fundamentals.risks[:2]

    rebuttals = [
        DebatePoint(
            claim=text,
            support="Raised directly against the opposing case from analyst findings",
            strength=0.5,
            india_specific=_is_india_specific(text),
        )
        for text in rebuttal_seed
    ]

    thesis = _thesis(bundle, reports, side, points)
    net = _net_stance_score(reports)
    confidence = 0.5 + (0.25 * net if side == "bull" else -0.25 * net)

    return DebateTurn(
        role=agent.role,
        symbol=bundle.ref.yf_symbol,
        side=side,
        round_number=round_number,
        thesis=thesis,
        summary=thesis,
        reasoning=(
            f"Deterministic {side} construction from analyst stances "
            f"(net stance score {net:+.2f})."
        ),
        confidence=round(max(0.1, min(0.9, confidence)), 3),
        points=points[:8],
        rebuttals=rebuttals[:4],
        conceded=conceded[:4],
        india_risk_focus=india_focus,
        data_quality=bundle.quality(["prices", "news", "macro"]),
    )


_BEARISH_TOKENS = (
    "bearish", "downtrend", "negative", "declining", "weak", "deteriorat",
    "below", "breakdown", "oversold", "contract",
)
_BULLISH_TOKENS = (
    "bullish", "uptrend", "positive", "improving", "strong", "expand",
    "above", "breakout", "overbought",
)
_INDIA_TOKENS = (
    "circuit", "sebi", "asm", "gsm", "expiry", "gst", "promoter", "pledge",
    "fii", "dii", "stt", "rbi", "monsoon", "delivery", "intraday", "f&o",
)


def _supports(text: str, side: str) -> bool:
    """Does this observation actually argue for `side`?

    Neutral or unclassifiable text passes: the test is for *contradiction*, not
    for enthusiasm, and discarding everything without a polarity word would
    throw away most of the substance.
    """
    lowered = text.lower()
    against = _BEARISH_TOKENS if side == "bull" else _BULLISH_TOKENS
    for token in against:
        if token in lowered:
            return False
    return True


def _is_india_specific(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _INDIA_TOKENS)


def _india_points(bundle: MarketBundle, side: str) -> list[DebatePoint]:
    """India-mechanics arguments that follow from the microstructure data."""
    micro = bundle.micro
    out: list[DebatePoint] = []

    if side == "bear":
        if micro.is_circuit_locked:
            out.append(
                DebatePoint(
                    claim="The scrip is circuit-locked, so a stop-loss cannot be filled",
                    support="Price is pinned at the exchange band; there is no liquidity "
                            "on the other side to exit into.",
                    strength=0.95,
                    india_specific=True,
                )
            )
        if micro.circuit_band_pct and micro.circuit_band_pct <= 5:
            out.append(
                DebatePoint(
                    claim=f"A {micro.circuit_band_pct:.0f}% price band caps both upside and exit",
                    support="Narrow-band scrips gap between limits; the modelled stop is "
                            "not the achievable stop.",
                    strength=0.7,
                    india_specific=True,
                )
            )
        if micro.in_fno_ban:
            out.append(
                DebatePoint(
                    claim="Scrip is in the F&O ban period",
                    support="Only position-reducing derivative orders are permitted, so the "
                            "position cannot be hedged with futures or options.",
                    strength=0.85,
                    india_specific=True,
                )
            )
        if micro.surveillance.value.startswith(("ASM", "GSM")):
            out.append(
                DebatePoint(
                    claim=f"Under {micro.surveillance.value} surveillance",
                    support="Higher margins and possible trade-for-trade settlement reduce "
                            "liquidity and raise the true cost of the position.",
                    strength=0.8,
                    india_specific=True,
                )
            )
        if micro.is_expiry_day:
            out.append(
                DebatePoint(
                    claim="Entering on expiry day invites pinning and unwind noise",
                    support="Prices gravitate toward high open-interest strikes into the "
                            "close, so intraday levels are unreliable.",
                    strength=0.6,
                    india_specific=True,
                )
            )
        turnover = micro.avg_daily_turnover_inr
        if turnover is not None and turnover < 5_00_00_000:
            out.append(
                DebatePoint(
                    claim=f"Thin liquidity: only Rs {turnover / 1e7:.1f} cr traded per day",
                    support="Impact cost on any meaningful size will exceed the modelled "
                            "entry, and exiting in a falling tape will be worse.",
                    strength=0.75,
                    india_specific=True,
                )
            )
    else:
        if bundle.ref.fno_eligible:
            out.append(
                DebatePoint(
                    claim="F&O eligibility allows the position to be hedged or expressed "
                          "with defined risk",
                    support=f"Lot size {bundle.ref.lot_size}; futures and options are "
                            "available for hedging or a debit spread.",
                    strength=0.6,
                    india_specific=True,
                )
            )
        if not micro.in_fno_ban and micro.surveillance.value == "NONE":
            out.append(
                DebatePoint(
                    claim="No surveillance flag and not in the F&O ban list",
                    support="The scrip is freely tradable with normal margins and no "
                            "settlement restriction.",
                    strength=0.5,
                    india_specific=True,
                )
            )
        turnover = micro.avg_daily_turnover_inr
        if turnover is not None and turnover > 50_00_00_000:
            out.append(
                DebatePoint(
                    claim=f"Deep liquidity at Rs {turnover / 1e7:.0f} cr traded per day",
                    support="Institutional-scale positions can be built and exited with "
                            "minimal impact cost.",
                    strength=0.6,
                    india_specific=True,
                )
            )
        if bundle.flows.dii_net_inr_cr and bundle.flows.dii_net_inr_cr > 0:
            out.append(
                DebatePoint(
                    claim="Domestic institutions are net buyers",
                    support=bundle.flows.interpretation or "DII net buying in the cash segment.",
                    strength=0.5,
                    india_specific=True,
                )
            )
    return out


def _net_stance_score(reports: dict[str, Any]) -> float:
    weights = {
        Stance.BULLISH: 1.0, Stance.MILDLY_BULLISH: 0.5, Stance.NEUTRAL: 0.0,
        Stance.MILDLY_BEARISH: -0.5, Stance.BEARISH: -1.0,
    }
    values = [
        weights[r.stance] * r.effective_confidence()
        for r in reports.values()
        if r is not None
    ]
    return round(sum(values) / len(values), 3) if values else 0.0


def _thesis(
    bundle: MarketBundle,
    reports: dict[str, Any],
    side: str,
    points: list[DebatePoint] | None = None,
) -> str:
    supporting = [
        name for name, r in reports.items()
        if r is not None and r.stance in (_BULLISH if side == "bull" else _BEARISH)
    ]
    symbol = bundle.ref.base_symbol
    if not supporting:
        substantive = [p for p in (points or []) if p.source_agent]
        if substantive:
            # No analyst carries this side outright, but the case is built from
            # the risks those same analysts named against their own stance -
            # which is a stronger position than it sounds.
            sources = sorted({p.source_agent for p in substantive if p.source_agent})
            return (
                f"No analyst leg reads {side} on {symbol} outright, so the case rests on the "
                f"risks the {', '.join(sources)} analyst(s) named against their own stance: "
                f"{substantive[0].claim}"
            )
        return (
            f"The {side} case on {symbol} is weak on this data: no analyst leg supports it. "
            f"Presented for completeness, not conviction."
        )
    legs = ", ".join(supporting)
    direction = "upside" if side == "bull" else "downside"
    return (
        f"The {side} case on {symbol} rests on the {legs} leg(s), which point to {direction} "
        f"at Rs {bundle.last_price:,.2f}." if bundle.last_price else
        f"The {side} case on {symbol} rests on the {legs} leg(s)."
    )
