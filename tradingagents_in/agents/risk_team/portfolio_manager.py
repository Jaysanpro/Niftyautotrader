"""The Portfolio Manager - final approve or reject.

The risk manager judges one trade in isolation. The PM judges it against the
book: what else is on, what the same shock does to all of it at once, and
whether this idea earns the risk it consumes relative to what is already held.

It is also the last place a correct-but-unowned view gets stopped. A trade the
desk cannot monitor, cannot exit, or cannot afford to be wrong about twice is
rejected here even when every individual limit passes.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent
from tradingagents_in.agents.briefings import header
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import (
    Action,
    AgentRole,
    Instrument,
    RiskVerdict,
    Severity,
    TimeHorizon,
)
from tradingagents_in.models.research import InvestmentPlan
from tradingagents_in.models.risk import PortfolioDecision, PortfolioSnapshot, RiskAssessment
from tradingagents_in.models.trade import TradeProposal
from tradingagents_in.utils.formatting import inr_plain


class PortfolioManager(Agent[PortfolioDecision]):
    role = AgentRole.PORTFOLIO_MANAGER
    output_model = PortfolioDecision
    tier_key = "portfolio_manager"

    def system_prompt(self) -> str:
        return """\
You are the Portfolio Manager. You own the book, and this is the final decision.

You are not re-running the analysis. Research settled the view, the trader
specified the order, the risk manager sized it against hard limits. Your
question is narrower and harder: **does this position belong in this book right
now?**

Approve or reject on portfolio grounds:

- **Marginal contribution.** Does this add a return stream the book does not
  already have, or is it a fourth expression of the same factor? Three IT
  exporters is one bet on the rupee and on US discretionary spend.
- **Simultaneous loss.** If the market gaps 3% against the book, what does the
  whole thing lose - not this position alone? Correlation goes to one in an
  Indian risk-off session, and FII selling hits every large-cap at once.
- **Capacity to manage.** Positions the desk cannot watch are positions the
  desk does not really own. An intraday F&O trade needs a screen; a positional
  cash holding does not.
- **Opportunity cost.** Capital committed here is capital unavailable for a
  better setup this week. Say so if the idea is merely acceptable.

You may override the risk manager only downward - reject what it approved, or
cut the size further. You cannot approve what it rejected, and you cannot size
above what it allowed. If you override anything, record it in `dissent` with
the reason.

State conditions that must hold at execution (a price level, a volume
confirmation, a flow print) and review_triggers that force a fresh look. If you
reject, rejection_reasons must be specific and portfolio-level, not a repeat of
the risk manager's rule list.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        proposal: TradeProposal | None = state.get("trade_proposal")
        risk: RiskAssessment | None = state.get("risk_assessment")
        plan: InvestmentPlan | None = state.get("investment_plan")
        portfolio: PortfolioSnapshot = state.get("portfolio") or PortfolioSnapshot()
        cfg = self.config

        parts = [
            header(bundle),
            "## Research view\n\n"
            + (
                f"- Stance {plan.stance.value}, conviction {plan.conviction:.0%}, "
                f"horizon {plan.preferred_horizon.value}\n- {plan.verdict}"
                if plan else "(none)"
            ),
            "## Trade proposal\n\n" + (proposal.to_markdown() if proposal else "(none)"),
            "## Risk assessment\n\n"
            + (
                f"- Verdict: **{risk.verdict.value}**\n"
                f"- Approved size: {risk.approved_position_size_pct:.2f}% "
                f"({risk.approved_quantity or 0} units)\n"
                f"- Max loss: {inr_plain(risk.max_loss_inr)} "
                f"({risk.max_loss_pct_of_capital or 0:.2f}% of capital)\n"
                f"- Sizing: {risk.sizing_rationale}\n"
                f"- Breaches:\n" + "\n".join(
                    f"  - [{b.severity.value}] {b.rule}: {b.detail}" for b in risk.breaches
                ) + "\n"
                f"- Stress test (INR): {risk.stress_test}\n"
                f"- Risk notes:\n" + "\n".join(f"  - {n}" for n in risk.risk_notes)
                if risk else "(none)"
            ),
            "## The book as it stands\n\n"
            f"- Capital {inr_plain(portfolio.capital_inr or cfg.get('capital_inr'))}, "
            f"deployed {inr_plain(portfolio.deployed_inr)}, cash "
            f"{inr_plain(portfolio.cash_inr)}\n"
            f"- {portfolio.open_positions} open position(s), heat "
            f"{portfolio.portfolio_heat_pct:.2f}% of a "
            f"{cfg.get('max_portfolio_heat_pct')}% budget\n"
            f"- Drawdown {portfolio.current_drawdown_pct:.2f}% against a "
            f"{cfg.get('max_drawdown_pct')}% limit\n"
            f"- Sector exposure: {portfolio.sector_exposure_pct or 'flat'}\n"
            f"- Symbol exposure: {portfolio.symbol_exposure_pct or 'flat'}\n"
            f"- This idea's sector: {bundle.ref.sector or 'unclassified'}",
        ]

        if state.get("memory_context"):
            parts.append("## Trading memory\n\n" + state["memory_context"])

        parts.append(
            """## Your task

Return a PortfolioDecision: approved, final_action, final_position_size_pct
(never above the risk manager's approved size), capital_deployed_inr,
capital_at_risk_inr, time_horizon, portfolio_reasoning, conditions,
review_triggers, and rejection_reasons if rejecting. Set confidence as your
probability that this is the right call for the book, not for the stock."""
        )
        return "\n\n".join(parts)

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> PortfolioDecision:
        return decide(state, bundle, self.config, role=self.role)

    def run(self, bundle: MarketBundle, state: dict[str, Any] | None = None) -> PortfolioDecision:
        state = state or {}
        result = super().run(bundle, state)
        return enforce_ceiling(result, state, self.config)


# ---------------------------------------------------------------------------
def decide(
    state: dict[str, Any],
    bundle: MarketBundle,
    config: dict[str, Any],
    role: AgentRole = AgentRole.PORTFOLIO_MANAGER,
) -> PortfolioDecision:
    """Deterministic final decision from the risk assessment and the book."""
    proposal: TradeProposal | None = state.get("trade_proposal")
    risk: RiskAssessment | None = state.get("risk_assessment")
    plan: InvestmentPlan | None = state.get("investment_plan")
    portfolio: PortfolioSnapshot = state.get("portfolio") or PortfolioSnapshot()
    symbol = bundle.ref.yf_symbol
    capital = float(config.get("capital_inr", 1_000_000))

    if proposal is None or proposal.action == Action.HOLD:
        reason = proposal.summary if proposal else "No proposal was produced."
        return PortfolioDecision(
            role=role,
            symbol=symbol,
            approved=False,
            final_action=Action.HOLD,
            confidence=0.6,
            summary=f"HOLD {bundle.ref.base_symbol}. {reason}",
            portfolio_reasoning=(
                "No position is taken, so the book is unchanged. Staying flat when the "
                "setup is absent is the decision, not the absence of one."
            ),
            risk_notes=(risk.risk_notes[:5] if risk else []),
            review_triggers=(plan.key_monitorables[:5] if plan else []),
            rejection_reasons=[],
            data_quality=bundle.quality(["prices", "quote", "microstructure"]),
        )

    if risk is None or risk.verdict == RiskVerdict.REJECT:
        blockers = [b for b in (risk.breaches if risk else []) if b.severity == Severity.BLOCKER]
        return PortfolioDecision(
            role=role,
            symbol=symbol,
            approved=False,
            final_action=Action.HOLD,
            confidence=0.8,
            summary=(
                f"REJECTED: {proposal.action.value} {bundle.ref.base_symbol} does not clear "
                "risk. No capital committed."
            ),
            portfolio_reasoning=(
                "A directional view the desk cannot express within its risk limits is not a "
                "position. The view is recorded and will be revisited when the blocking "
                "condition clears."
            ),
            risk_notes=(risk.risk_notes[:5] if risk else []),
            rejection_reasons=[f"{b.rule}: {b.detail}" for b in blockers[:5]] or
                              ["Risk manager rejected the proposal."],
            review_triggers=(plan.key_monitorables[:5] if plan else []),
            data_quality=bundle.quality(["prices", "quote", "microstructure"]),
        )

    size_pct = risk.approved_position_size_pct
    portfolio_reasons = _portfolio_objections(proposal, bundle, portfolio, risk, config)

    if portfolio_reasons or size_pct <= 0:
        return PortfolioDecision(
            role=role,
            symbol=symbol,
            approved=False,
            final_action=Action.HOLD,
            confidence=0.7,
            summary=(
                f"REJECTED on portfolio grounds: {bundle.ref.base_symbol} clears the "
                "single-trade rules but not the book-level ones."
            ),
            portfolio_reasoning="; ".join(portfolio_reasons)
            or "Approved size rounded to zero, so there is no position to take.",
            risk_notes=risk.risk_notes[:5],
            rejection_reasons=portfolio_reasons[:5] or ["Approved size is zero."],
            review_triggers=(plan.key_monitorables[:5] if plan else []),
            dissent=(
                "Risk manager approved; the portfolio manager declines on concentration "
                "or capacity grounds."
                if risk.verdict != RiskVerdict.REJECT else None
            ),
            data_quality=bundle.quality(["prices", "quote", "microstructure"]),
        )

    deployed = round((proposal.capital_deployed_inr or 0.0)
                     * (size_pct / proposal.position_size_pct if proposal.position_size_pct else 0), 2)
    at_risk = risk.max_loss_inr or 0.0

    return PortfolioDecision(
        role=role,
        symbol=symbol,
        approved=True,
        final_action=proposal.action,
        final_position_size_pct=round(size_pct, 3),
        capital_deployed_inr=deployed,
        capital_at_risk_inr=at_risk,
        time_horizon=proposal.time_horizon,
        confidence=_confidence(plan, risk, bundle),
        summary=(
            f"APPROVED: {proposal.action.value} {bundle.ref.base_symbol} at "
            f"{size_pct:.2f}% of capital ({inr_plain(deployed)}), risking "
            f"{inr_plain(at_risk)} ({at_risk / capital * 100:.2f}% of the book)."
        ),
        portfolio_reasoning=_reasoning(proposal, bundle, portfolio, risk, config),
        risk_notes=risk.risk_notes[:6],
        conditions=_conditions(proposal, bundle, risk),
        review_triggers=_review_triggers(plan, proposal, bundle),
        rejection_reasons=[],
        dissent=(
            f"Size cut from the trader's {proposal.position_size_pct:.2f}% to "
            f"{size_pct:.2f}% by the risk manager."
            if size_pct < proposal.position_size_pct - 1e-6 else None
        ),
        data_quality=bundle.quality(["prices", "quote", "microstructure"]),
    )


def enforce_ceiling(
    decision: PortfolioDecision, state: dict[str, Any], config: dict[str, Any]
) -> PortfolioDecision:
    """The PM may only be more conservative than the risk manager."""
    risk: RiskAssessment | None = state.get("risk_assessment")
    if risk is None:
        return decision

    if risk.verdict == RiskVerdict.REJECT and decision.approved:
        decision.approved = False
        decision.final_action = Action.HOLD
        decision.final_position_size_pct = 0.0
        decision.capital_deployed_inr = 0.0
        decision.capital_at_risk_inr = 0.0
        decision.rejection_reasons = [
            f"{b.rule}: {b.detail}" for b in risk.breaches if b.severity == Severity.BLOCKER
        ] or ["Risk manager rejected the proposal."]
        decision.dissent = (
            "Portfolio manager attempted to approve a risk-rejected trade; overridden. "
            "A hard risk limit is not negotiable at the portfolio level."
        )
        return decision

    if decision.final_position_size_pct > risk.approved_position_size_pct + 1e-6:
        decision.risk_notes.append(
            f"Size clamped from {decision.final_position_size_pct:.2f}% to the risk "
            f"manager's approved {risk.approved_position_size_pct:.2f}%."
        )
        decision.final_position_size_pct = risk.approved_position_size_pct

    if not decision.approved:
        decision.final_position_size_pct = 0.0
        decision.capital_deployed_inr = 0.0
        decision.capital_at_risk_inr = 0.0
        decision.final_action = Action.HOLD
    return decision


# ----------------------------------------------------------------- helpers
def _portfolio_objections(
    proposal: TradeProposal,
    bundle: MarketBundle,
    portfolio: PortfolioSnapshot,
    risk: RiskAssessment,
    config: dict[str, Any],
) -> list[str]:
    """Book-level reasons to decline a trade that passed the single-trade rules."""
    out: list[str] = []
    sector = bundle.ref.sector or "Unclassified"
    size = risk.approved_position_size_pct

    sector_after = portfolio.sector_exposure_pct.get(sector, 0.0) + size
    if sector_after > float(config.get("max_sector_pct", 25.0)):
        out.append(
            f"{sector} exposure would reach {sector_after:.1f}% against a "
            f"{config.get('max_sector_pct')}% cap. Indian sector moves are near-perfectly "
            "correlated on a policy or rate print, so this is one position, not several."
        )

    capital = float(config.get("capital_inr", 1_000_000))
    heat_after = portfolio.portfolio_heat_pct + (risk.max_loss_inr or 0) / capital * 100
    if heat_after > float(config.get("max_portfolio_heat_pct", 6.0)):
        out.append(
            f"Aggregate open risk would reach {heat_after:.2f}% against a "
            f"{config.get('max_portfolio_heat_pct')}% budget. In a gap-down the stops do not "
            "fire independently - they all fire."
        )

    if portfolio.current_drawdown_pct >= float(config.get("max_drawdown_pct", 15.0)) * 0.66:
        out.append(
            f"The book is {portfolio.current_drawdown_pct:.1f}% into a "
            f"{config.get('max_drawdown_pct')}% drawdown limit. Adding risk while drawn down "
            "is how a recoverable quarter becomes a mandate breach."
        )

    if portfolio.open_positions >= int(config.get("max_open_positions", 12)):
        out.append(
            f"{portfolio.open_positions} positions are already open, at the monitoring limit. "
            "A position nobody is watching is not risk-managed."
        )

    if (
        proposal.instrument in (Instrument.FUTURES, Instrument.OPTIONS)
        and proposal.time_horizon == TimeHorizon.INTRADAY
        and portfolio.open_positions > 5
    ):
        out.append(
            "An intraday derivative position needs continuous attention, and the desk is "
            "already carrying more names than it can watch tick by tick."
        )
    return out


def _reasoning(
    proposal: TradeProposal,
    bundle: MarketBundle,
    portfolio: PortfolioSnapshot,
    risk: RiskAssessment,
    config: dict[str, Any],
) -> str:
    sector = bundle.ref.sector or "unclassified"
    capital = float(config.get("capital_inr", 1_000_000))
    heat_after = portfolio.portfolio_heat_pct + (risk.max_loss_inr or 0) / capital * 100
    sector_after = portfolio.sector_exposure_pct.get(sector, 0.0) + risk.approved_position_size_pct

    return (
        f"Takes {sector} exposure to {sector_after:.1f}% (cap {config.get('max_sector_pct')}%) "
        f"and total open risk to {heat_after:.2f}% (budget "
        f"{config.get('max_portfolio_heat_pct')}%), leaving room for "
        f"{max(0, int((float(config.get('max_portfolio_heat_pct', 6)) - heat_after) / max(float(config.get('risk_per_trade_pct', 1)), 0.01)))} "
        f"further ideas at full size. The position is {proposal.time_horizon.value.lower()} "
        f"in {proposal.instrument.value.lower()}, which the desk can monitor alongside the "
        f"{portfolio.open_positions} name(s) already open. Worst modelled single-session "
        f"outcome is {inr_plain(min(risk.stress_test.values()) if risk.stress_test else 0)}."
    )


def _conditions(
    proposal: TradeProposal, bundle: MarketBundle, risk: RiskAssessment
) -> list[str]:
    out = []
    if proposal.entry_zone:
        out.append(
            f"Execute only inside {proposal.entry_zone} - chasing above the band changes the "
            "reward-risk the approval was based on."
        )
        if proposal.entry_zone.trigger:
            out.append(proposal.entry_zone.trigger)
    out.append(
        f"Stop-loss order at {proposal.stop_loss:,.2f} goes in with the entry, not after it."
    )
    if risk.approved_quantity:
        out.append(f"Maximum {risk.approved_quantity} units; do not average down.")
    if bundle.micro.is_expiry_day:
        out.append("Do not initiate in the final hour of an expiry session.")
    out.extend(risk.required_changes[:2])
    return out[:8]


def _review_triggers(
    plan: InvestmentPlan | None, proposal: TradeProposal, bundle: MarketBundle
) -> list[str]:
    out: list[str] = []
    if plan:
        out.extend(plan.key_monitorables[:3])
    out.append(
        f"Close beyond {proposal.stop_loss:,.2f} on the {proposal.time_horizon.value.lower()} "
        "timeframe"
    )
    if bundle.ref.fno_eligible:
        out.append("Entry of the scrip into the F&O ban list or an ASM/GSM stage")
    out.append("Quarterly results or a material exchange filing")
    return out[:6]


def _confidence(
    plan: InvestmentPlan | None, risk: RiskAssessment, bundle: MarketBundle
) -> float:
    conviction = plan.conviction if plan else 0.5
    penalty = 0.1 if risk.verdict == RiskVerdict.APPROVE_WITH_CHANGES else 0.0
    completeness = bundle.quality(["prices", "fundamentals", "news", "macro"]).completeness
    value = (0.6 * conviction + 0.4 * risk.confidence) * (0.7 + 0.3 * completeness) - penalty
    return round(max(0.1, min(0.92, value)), 3)
