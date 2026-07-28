"""The Risk Manager.

Sits between the trader and the book. It runs the deterministic rulebook first,
then asks the model to explain and, where warranted, to be *more* conservative.

The asymmetry is deliberate and enforced in code, not in the prompt: a BLOCKER
found by :mod:`tradingagents_in.risk.rules` forces REJECT and the approved size
to zero after the LLM has spoken. A language model may add caution; it may not
remove it. Anything else would put a text generator in charge of whether the
desk can exit a position.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent
from tradingagents_in.agents.briefings import corporate_actions_brief, header, microstructure_brief
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import (
    Action,
    AgentRole,
    Instrument,
    RiskVerdict,
    Severity,
)
from tradingagents_in.models.risk import PortfolioSnapshot, RiskAssessment, RuleBreach
from tradingagents_in.models.trade import TradeProposal
from tradingagents_in.risk import rules as rulebook
from tradingagents_in.risk.costs import breakeven_move_pct
from tradingagents_in.risk.sizing import risk_budget_inr
from tradingagents_in.utils.formatting import inr_plain


class RiskManager(Agent[RiskAssessment]):
    role = AgentRole.RISK_MANAGER
    output_model = RiskAssessment
    tier_key = "risk_manager"

    def system_prompt(self) -> str:
        return """\
You are the Risk Manager. Your mandate is capital preservation, and you are the
last check before the portfolio manager sees the idea.

A deterministic rulebook has already been run against this proposal. Its output
is authoritative:

- Any breach marked BLOCKER means REJECT. You cannot argue it away, and a
  response that approves through a blocker will be overridden.
- Breaches marked WARNING mean APPROVE_WITH_CHANGES. Say precisely what must
  change - a number, not a sentiment.
- No breaches means APPROVE, but you may still cut the size if the data is thin
  or the setup is fragile. Being more conservative than the rulebook is always
  allowed; being less conservative never is.

Judge these things the rulebook cannot:

1. **Is the stop real?** A stop is a price at which you will actually be filled.
   In a banded scrip, a thin counter, or an expiry-day tape, it may not be.
2. **Is the loss the modelled loss?** Indian single stocks gap on results,
   block deals, regulatory action and promoter news. Size for the gap, not for
   the stop.
3. **Correlation.** Three private banks is one position in three tickers. Say
   so if the book is already exposed to the same factor.
4. **Costs versus edge.** If the breakeven move is a meaningful fraction of the
   first target, the edge is mostly friction.

Write sizing_rationale as arithmetic a reader can check: risk budget, stop
distance, resulting quantity, and which constraint bound the size. Populate
stress_test with rupee P&L for each scenario. Keep risk_notes concrete and put
standing regulatory context in sebi_compliance_notes.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        proposal: TradeProposal | None = state.get("trade_proposal")
        portfolio: PortfolioSnapshot = state.get("portfolio") or PortfolioSnapshot()
        breaches = _breaches(state, bundle, self.config)
        cfg = self.config

        parts = [
            header(bundle),
            microstructure_brief(bundle),
            corporate_actions_brief(bundle),
            "## The proposal under review\n\n"
            + (proposal.to_markdown() if proposal else "(no proposal)"),
            "## Deterministic rulebook output\n\n" + rulebook.summarise(breaches),
            "## Mandate limits\n\n"
            f"- Capital {inr_plain(cfg.get('capital_inr'))} ({cfg.get('profile')})\n"
            f"- Risk per trade {cfg.get('risk_per_trade_pct')}% = "
            f"{inr_plain(risk_budget_inr(float(cfg.get('capital_inr', 0)), cfg))}\n"
            f"- Max single name {cfg.get('max_position_pct')}% | Max sector "
            f"{cfg.get('max_sector_pct')}% | Max heat {cfg.get('max_portfolio_heat_pct')}%\n"
            f"- Max drawdown {cfg.get('max_drawdown_pct')}% | Min net R:R "
            f"{cfg.get('min_reward_risk')}",
            "## Current book\n\n"
            f"- Deployed {inr_plain(portfolio.deployed_inr)} of "
            f"{inr_plain(portfolio.capital_inr or cfg.get('capital_inr'))}, "
            f"cash {inr_plain(portfolio.cash_inr)}\n"
            f"- Open positions: {portfolio.open_positions} | Heat: "
            f"{portfolio.portfolio_heat_pct:.2f}% | Drawdown: "
            f"{portfolio.current_drawdown_pct:.2f}%\n"
            f"- Sector exposure: {portfolio.sector_exposure_pct or 'flat'}\n"
            f"- Overnight F&O exposure: {portfolio.overnight_fno_exposure_pct:.2f}%",
            """## Your task

Return a RiskAssessment. Set verdict per the rules above, restate every breach
in `breaches`, set approved_position_size_pct (0 on REJECT), give
max_loss_inr and max_loss_pct_of_capital, fill stress_test with rupee outcomes,
and list required_changes as specific numeric edits when approving with
changes.""",
        ]
        return "\n\n".join(parts)

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> RiskAssessment:
        return assess(state, bundle, self.config, role=self.role)

    # The rulebook has the final word, whatever the model returned.
    def run(self, bundle: MarketBundle, state: dict[str, Any] | None = None) -> RiskAssessment:
        state = state or {}
        result = super().run(bundle, state)
        return enforce_rulebook(result, state, bundle, self.config)


# ---------------------------------------------------------------------------
def _breaches(
    state: dict[str, Any], bundle: MarketBundle, config: dict[str, Any]
) -> list[RuleBreach]:
    proposal: TradeProposal | None = state.get("trade_proposal")
    portfolio: PortfolioSnapshot = state.get("portfolio") or PortfolioSnapshot(
        capital_inr=float(config.get("capital_inr", 0)),
        cash_inr=float(config.get("capital_inr", 0)),
    )
    if proposal is None:
        return []
    return rulebook.check_all(proposal, bundle, portfolio, config)


def assess(
    state: dict[str, Any],
    bundle: MarketBundle,
    config: dict[str, Any],
    role: AgentRole = AgentRole.RISK_MANAGER,
) -> RiskAssessment:
    """The deterministic assessment: rulebook, sizing arithmetic, stress test."""
    proposal: TradeProposal | None = state.get("trade_proposal")
    symbol = bundle.ref.yf_symbol
    capital = float(config.get("capital_inr", 1_000_000))

    if proposal is None or proposal.action == Action.HOLD:
        state_breaches = rulebook.check_portfolio_state(
            state.get("portfolio") or PortfolioSnapshot(), config
        )
        return RiskAssessment(
            role=role,
            symbol=symbol,
            verdict=RiskVerdict.APPROVE,
            breaches=state_breaches,
            microstructure=bundle.micro,
            approved_position_size_pct=0.0,
            max_loss_inr=0.0,
            max_loss_pct_of_capital=0.0,
            confidence=0.6,
            summary="No position proposed, so there is no new risk to approve. "
                    "Zero exposure is a valid outcome.",
            sizing_rationale="Not applicable: the trader proposed HOLD.",
            risk_notes=[b.detail for b in state_breaches],
            sebi_compliance_notes=[],
            data_quality=bundle.quality(["prices", "quote", "microstructure"]),
        )

    breaches = _breaches(state, bundle, config)
    blockers = [b for b in breaches if b.severity == Severity.BLOCKER]
    warnings = [b for b in breaches if b.severity == Severity.WARNING]

    if blockers:
        verdict = RiskVerdict.REJECT
        approved_pct = 0.0
    elif warnings:
        verdict = RiskVerdict.APPROVE_WITH_CHANGES
        approved_pct = _trimmed_size(proposal, state, config, bundle.ref.sector)
    else:
        verdict = RiskVerdict.APPROVE
        approved_pct = proposal.position_size_pct

    scale = (approved_pct / proposal.position_size_pct) if proposal.position_size_pct else 0.0
    approved_qty = int((proposal.quantity or 0) * scale)
    approved_lots = None
    lot = bundle.ref.lot_size
    if proposal.instrument in (Instrument.FUTURES, Instrument.OPTIONS) and lot:
        approved_lots = approved_qty // lot
        approved_qty = approved_lots * lot
        if approved_lots == 0 and verdict == RiskVerdict.APPROVE_WITH_CHANGES:
            # Lots are indivisible: trimming below one lot means no trade.
            verdict = RiskVerdict.REJECT
            approved_pct = 0.0

    max_loss = round((proposal.risk_amount_inr or 0.0) * scale, 2)
    stress = rulebook.stress_test(proposal, bundle, config)
    entry = proposal.entry_zone.mid if proposal.entry_zone else bundle.last_price

    return RiskAssessment(
        role=role,
        symbol=symbol,
        verdict=verdict,
        breaches=breaches,
        microstructure=bundle.micro,
        approved_position_size_pct=round(approved_pct, 3),
        approved_quantity=approved_qty or None,
        approved_lots=approved_lots,
        max_loss_inr=max_loss,
        max_loss_pct_of_capital=round(max_loss / capital * 100, 3) if capital else None,
        adjusted_stop_loss=proposal.stop_loss,
        confidence=_confidence(verdict, breaches, bundle),
        summary=_summary(verdict, proposal, blockers, warnings, bundle),
        reasoning=rulebook.summarise(breaches),
        sizing_rationale=_sizing_rationale(proposal, config, approved_pct, approved_qty, entry),
        risk_notes=_risk_notes(proposal, bundle, config, entry, approved_qty),
        sebi_compliance_notes=rulebook.sebi_compliance_notes(proposal, bundle, config),
        required_changes=[
            f"{b.rule}: {b.remedy}" for b in warnings if b.remedy
        ][:6],
        stress_test=stress,
        data_quality=bundle.quality(["prices", "quote", "microstructure"]),
    )


def enforce_rulebook(
    assessment: RiskAssessment,
    state: dict[str, Any],
    bundle: MarketBundle,
    config: dict[str, Any],
) -> RiskAssessment:
    """Clamp an LLM assessment back inside the deterministic rulebook.

    The model may tighten anything. It may not approve through a blocker, size
    above what the trader proposed, or exceed the mandate's single-name cap.
    """
    breaches = _breaches(state, bundle, config)
    assessment.breaches = breaches  # never trust a model's list of violations
    assessment.microstructure = bundle.micro
    blockers = [b for b in breaches if b.severity == Severity.BLOCKER]

    if blockers:
        assessment.verdict = RiskVerdict.REJECT
        assessment.approved_position_size_pct = 0.0
        assessment.approved_quantity = None
        assessment.approved_lots = None
        assessment.max_loss_inr = 0.0
        assessment.max_loss_pct_of_capital = 0.0
        reasons = "; ".join(f"{b.rule} - {b.detail}" for b in blockers[:3])
        if "REJECT" not in assessment.summary.upper():
            assessment.summary = f"REJECTED on hard risk limits: {reasons}"
        return assessment

    proposal: TradeProposal | None = state.get("trade_proposal")
    ceiling = min(
        float(config.get("max_position_pct", 10.0)),
        proposal.position_size_pct if proposal else 0.0,
    )
    if assessment.approved_position_size_pct > ceiling:
        assessment.risk_notes.append(
            f"Approved size clamped from {assessment.approved_position_size_pct:.2f}% to "
            f"{ceiling:.2f}%: the risk manager cannot approve more than the trader "
            "requested or more than the mandate allows."
        )
        assessment.approved_position_size_pct = round(ceiling, 3)

    capital = float(config.get("capital_inr", 1_000_000))
    budget = risk_budget_inr(capital, config)
    if (assessment.max_loss_inr or 0) > budget * 1.05:
        assessment.risk_notes.append(
            f"Stated max loss {inr_plain(assessment.max_loss_inr)} exceeds the per-trade "
            f"budget {inr_plain(budget)}; capped to the budget."
        )
        assessment.max_loss_inr = round(budget, 2)
        assessment.max_loss_pct_of_capital = round(budget / capital * 100, 3)

    if [b for b in breaches if b.severity == Severity.WARNING] and (
        assessment.verdict == RiskVerdict.APPROVE
    ):
        assessment.verdict = RiskVerdict.APPROVE_WITH_CHANGES
    return assessment


# ----------------------------------------------------------------- helpers
def _trimmed_size(
    proposal: TradeProposal,
    state: dict[str, Any],
    config: dict[str, Any],
    sector: str | None = None,
) -> float:
    """Largest size that clears every soft limit."""
    portfolio: PortfolioSnapshot = state.get("portfolio") or PortfolioSnapshot()
    capital = float(config.get("capital_inr", 1_000_000))

    ceiling = min(proposal.position_size_pct, float(config.get("max_position_pct", 10.0)))

    if sector:
        used = portfolio.sector_exposure_pct.get(sector, 0.0)
        ceiling = min(ceiling, max(0.0, float(config.get("max_sector_pct", 25.0)) - used))

    heat_left = max(
        0.0, float(config.get("max_portfolio_heat_pct", 6.0)) - portfolio.portfolio_heat_pct
    )
    risk_pct = (proposal.risk_amount_inr or 0.0) / capital * 100
    if risk_pct > 0 and heat_left < risk_pct:
        ceiling = min(ceiling, proposal.position_size_pct * heat_left / risk_pct)

    return round(max(0.0, ceiling), 3)


def _confidence(
    verdict: RiskVerdict, breaches: list[RuleBreach], bundle: MarketBundle
) -> float:
    base = {
        RiskVerdict.APPROVE: 0.8,
        RiskVerdict.APPROVE_WITH_CHANGES: 0.65,
        RiskVerdict.REJECT: 0.85,  # rejecting on a hard rule is a high-confidence act
    }[verdict]
    completeness = bundle.quality(["prices", "quote", "microstructure"]).completeness
    return round(max(0.2, min(0.95, base * (0.7 + 0.3 * completeness))), 3)


def _summary(
    verdict: RiskVerdict,
    proposal: TradeProposal,
    blockers: list[RuleBreach],
    warnings: list[RuleBreach],
    bundle: MarketBundle,
) -> str:
    symbol = bundle.ref.base_symbol
    if verdict == RiskVerdict.REJECT:
        return (
            f"REJECT {proposal.action.value} {symbol}. "
            + "; ".join(f"{b.rule}: {b.detail}" for b in blockers[:2])
        )
    if verdict == RiskVerdict.APPROVE_WITH_CHANGES:
        return (
            f"APPROVE {proposal.action.value} {symbol} with changes. "
            f"{len(warnings)} soft limit(s) engaged: "
            + "; ".join(b.rule for b in warnings[:3])
            + ". Size trimmed accordingly."
        )
    return (
        f"APPROVE {proposal.action.value} {symbol} as proposed. No mandate limit is "
        f"breached and the stop is fillable inside the price band."
    )


def _sizing_rationale(
    proposal: TradeProposal,
    config: dict[str, Any],
    approved_pct: float,
    approved_qty: int,
    entry: float | None,
) -> str:
    capital = float(config.get("capital_inr", 1_000_000))
    budget = risk_budget_inr(capital, config)
    if not entry or not proposal.stop_loss:
        return "Cannot show sizing arithmetic without an entry and a stop."
    distance = abs(entry - proposal.stop_loss)
    return (
        f"Risk budget {config.get('risk_per_trade_pct')}% of {inr_plain(capital)} = "
        f"{inr_plain(budget)}. Stop sits {distance:,.2f} from the {entry:,.2f} entry, so the "
        f"unconstrained size is {int(budget // distance) if distance else 0} units. "
        f"After the mandate caps, approved size is {approved_qty} units "
        f"({approved_pct:.2f}% of capital), risking "
        f"{inr_plain(approved_qty * distance)} at the stop."
    )


def _risk_notes(
    proposal: TradeProposal,
    bundle: MarketBundle,
    config: dict[str, Any],
    entry: float | None,
    quantity: int,
) -> list[str]:
    notes: list[str] = []
    micro = bundle.micro

    if entry and quantity and proposal.costs:
        breakeven = breakeven_move_pct(proposal.costs, entry, quantity)
        if breakeven:
            notes.append(
                f"Round-trip charges are {breakeven:.3f}% of turnover: price must move that "
                "far just to break even before the trade earns anything."
            )

    if micro.circuit_band_pct and entry and proposal.stop_loss:
        gap_loss = entry * micro.circuit_band_pct / 100 * quantity
        notes.append(
            f"A limit-down session would cost roughly {inr_plain(gap_loss)} with no chance to "
            f"act on the {proposal.stop_loss:,.2f} stop. That, not the modelled stop, is the "
            "single-session worst case."
        )

    if proposal.instrument == Instrument.OPTIONS:
        notes.append(
            "Option risk is the net debit and it can be lost in full while the directional "
            "call is still correct - time, not just direction, has to cooperate."
        )
    if proposal.instrument == Instrument.FUTURES:
        notes.append(
            "Futures mark to market daily: an adverse move triggers a margin call before the "
            "stop is reached, and an unmet call is squared off by the broker."
        )
    if micro.days_to_expiry is not None and micro.days_to_expiry <= 2:
        notes.append(
            f"{micro.days_to_expiry} session(s) to weekly expiry - expect pinning and thinner "
            "depth away from high-OI strikes."
        )
    if bundle.sources_failed:
        notes.append(
            f"These sources failed this run: {', '.join(bundle.sources_failed)}. The risk view "
            "is correspondingly incomplete."
        )
    return notes[:10]
