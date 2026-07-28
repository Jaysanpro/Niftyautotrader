"""Turn a finished graph state into a :class:`FinalDecision` and a report.

The report is the deliverable a human reads. It is written so that someone who
disagrees with the conclusion can find the exact step where the reasoning
diverged from theirs - the analyst stance, the debate turn, the rule that
blocked it - rather than being handed a verdict to take on trust.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.models.enums import Action
from tradingagents_in.models.research import DebateState
from tradingagents_in.models.risk import FinalDecision, PortfolioDecision, RiskAssessment
from tradingagents_in.models.state import ANALYST_STATE_KEYS
from tradingagents_in.models.trade import TradeProposal
from tradingagents_in.utils.formatting import bullet_list, inr_plain
from tradingagents_in.utils.timeutils import fmt_ist, now_ist

CONFIG_DIGEST_KEYS = (
    "llm_provider", "deep_think_model", "quick_think_model", "profile",
    "capital_inr", "risk_per_trade_pct", "max_position_pct", "max_debate_rounds",
    "max_risk_rounds", "selected_analysts", "online_tools", "data_provider",
)


def build_final_decision(
    state: dict[str, Any],
    config: dict[str, Any],
    elapsed_seconds: float | None = None,
) -> FinalDecision:
    """Collapse the graph state into the object callers actually consume."""
    ref = state.get("symbol_ref")
    proposal: TradeProposal | None = state.get("trade_proposal")
    risk: RiskAssessment | None = state.get("risk_assessment")
    decision: PortfolioDecision | None = state.get("portfolio_decision")
    debate: DebateState | None = state.get("debate")
    plan = state.get("investment_plan")

    analyst_reports = {
        name: state[key].model_dump(mode="json")
        for name, key in ANALYST_STATE_KEYS.items()
        if state.get(key) is not None
    }

    final = FinalDecision(
        symbol=ref.base_symbol if ref else state.get("symbol", "?"),
        yf_symbol=ref.yf_symbol if ref else state.get("symbol", "?"),
        exchange=ref.exchange.value if ref else "NSE",
        company_name=ref.name if ref else None,
        sector=ref.sector if ref else None,
        as_of=fmt_ist(now_ist()),
        run_id=state.get("run_id", ""),
        config_digest={k: config.get(k) for k in CONFIG_DIGEST_KEYS},
        final_action=decision.final_action if decision else Action.HOLD,
        approved=bool(decision.approved) if decision else False,
        confidence=decision.confidence if decision else 0.0,
        position_size_pct=decision.final_position_size_pct if decision else 0.0,
        capital_at_risk_inr=decision.capital_at_risk_inr if decision else 0.0,
        risk_notes=(decision.risk_notes if decision else []) or (risk.risk_notes if risk else []),
        analyst_reports=analyst_reports,
        debate_transcript=debate.transcript() if debate else "",
        investment_plan=plan.model_dump(mode="json") if plan else {},
        trade_proposal=proposal.model_dump(mode="json") if proposal else {},
        risk_assessment=risk.model_dump(mode="json") if risk else {},
        portfolio_decision=decision.model_dump(mode="json") if decision else {},
        errors=list(state.get("errors", [])),
        elapsed_seconds=round(elapsed_seconds, 2) if elapsed_seconds else None,
        proposal=proposal,
    )
    return final


def render_report(state: dict[str, Any], final: FinalDecision, config: dict[str, Any]) -> str:
    """Full markdown report: headline, every agent artefact, then the audit trail."""
    proposal: TradeProposal | None = state.get("trade_proposal")
    risk: RiskAssessment | None = state.get("risk_assessment")
    decision: PortfolioDecision | None = state.get("portfolio_decision")
    plan = state.get("investment_plan")
    debate: DebateState | None = state.get("debate")

    status = "APPROVED" if final.approved else "NOT TAKEN"
    parts: list[str] = [
        f"# {final.company_name or final.symbol} ({final.yf_symbol}) — "
        f"{final.final_action.value} [{status}]",
        "",
        f"_{final.as_of} · run `{final.run_id}` · "
        f"{config.get('llm_provider')} · {final.elapsed_seconds or 0:.1f}s_",
        "",
        "## Decision",
        "",
        f"- **Action**: {final.final_action.value}",
        f"- **Approved**: {'yes' if final.approved else 'no'}",
        f"- **Confidence**: {final.confidence:.0%}",
        f"- **Size**: {final.position_size_pct:.2f}% of capital "
        f"({inr_plain(decision.capital_deployed_inr if decision else 0)})",
        f"- **Capital at risk**: {inr_plain(final.capital_at_risk_inr)}",
        f"- **Sector**: {final.sector or 'unclassified'} · **Exchange**: {final.exchange}",
        "",
    ]

    if decision is not None:
        parts += ["### Portfolio reasoning", "", decision.portfolio_reasoning, ""]
        if decision.conditions:
            parts += ["**Conditions at execution**", bullet_list(decision.conditions), ""]
        if decision.rejection_reasons:
            parts += ["**Why this was not taken**", bullet_list(decision.rejection_reasons), ""]
        if decision.review_triggers:
            parts += ["**Review triggers**", bullet_list(decision.review_triggers), ""]
        if decision.dissent:
            parts += [f"> **Dissent.** {decision.dissent}", ""]

    if proposal is not None:
        parts += ["## Trade proposal", "", proposal.to_markdown(), ""]

    if risk is not None:
        parts += [
            "## Risk assessment",
            "",
            f"**Verdict: {risk.verdict.value}** — approved "
            f"{risk.approved_position_size_pct:.2f}% "
            f"({risk.approved_quantity or 0} units"
            + (f", {risk.approved_lots} lot(s)" if risk.approved_lots else "")
            + ")",
            "",
            risk.sizing_rationale,
            "",
        ]
        if risk.breaches:
            from tradingagents_in.risk.rules import summarise as summarise_breaches

            parts += ["**Rulebook**", summarise_breaches(risk.breaches), ""]
        if risk.stress_test:
            parts += [
                "**Stress test (INR)**",
                bullet_list(
                    [f"{k.replace('_', ' ')}: {inr_plain(v)}" for k, v in risk.stress_test.items()]
                ),
                "",
            ]
        if risk.risk_notes:
            parts += ["**Risk notes**", bullet_list(risk.risk_notes), ""]
        if risk.sebi_compliance_notes:
            parts += ["**Regulatory context**", bullet_list(risk.sebi_compliance_notes), ""]

    if plan is not None:
        parts += [
            "## Research view",
            "",
            f"**{plan.stance.value}** · conviction {plan.conviction:.0%} · debate won by "
            f"{plan.winning_side} · horizon {plan.preferred_horizon.value}",
            "",
            plan.verdict,
            "",
        ]
        if plan.decisive_arguments:
            parts += ["**Decisive arguments**", bullet_list(plan.decisive_arguments), ""]
        if plan.dismissed_arguments:
            parts += ["**Dismissed**", bullet_list(plan.dismissed_arguments), ""]
        if plan.unresolved_questions:
            parts += ["**Still unresolved**", bullet_list(plan.unresolved_questions), ""]

    if debate is not None and debate.turns:
        parts += [
            f"## Debate transcript ({debate.rounds_completed} round(s))",
            "",
            debate.transcript(),
            "",
        ]

    parts += ["## Analyst reports", ""]
    for key in ANALYST_STATE_KEYS.values():
        report = state.get(key)
        if report is None:
            continue
        parts += [report.to_markdown(), ""]

    parts += ["## Run audit", ""]
    bundle = (state.get("market_bundle") or {}).get("bundle")
    if bundle is not None:
        parts += [
            f"- Sources retrieved: {', '.join(bundle.sources_used) or 'none'}",
            f"- Sources failed: {', '.join(bundle.sources_failed) or 'none'}",
            f"- Data gather time: {bundle.elapsed_seconds}s",
        ]
    telemetry = state.get("telemetry", {})
    timings = {k: v for k, v in telemetry.items() if isinstance(v, dict) and "ms" in v}
    if timings:
        parts.append(
            "- Node timings (ms): "
            + ", ".join(f"{k} {v['ms']}" for k, v in sorted(timings.items()))
        )
    if final.errors:
        parts += ["", "**Errors during the run**", bullet_list(final.errors)]

    parts += [
        "",
        "---",
        "",
        "_Research output from an automated multi-agent system. Not investment advice. "
        "Verify every number against the exchange or a licensed source before acting, and "
        "consult a SEBI-registered investment adviser._",
    ]
    return "\n".join(parts)


def render_summary(final: FinalDecision) -> str:
    """One-screen summary for a terminal or a batch run."""
    proposal = final.proposal
    lines = [final.headline()]
    if proposal and proposal.is_actionable:
        lines += [
            f"  entry {proposal.entry_zone} | stop {proposal.stop_loss:,.2f}"
            + (f" | target {proposal.targets[0].price:,.2f}" if proposal.targets else ""),
            f"  {proposal.instrument.value}/{proposal.product_type.value}/"
            f"{proposal.time_horizon.value} | risk {inr_plain(final.capital_at_risk_inr)}",
        ]
    reason = (final.portfolio_decision or {}).get("portfolio_reasoning", "")
    if reason:
        lines.append(f"  {reason[:200]}")
    return "\n".join(lines)
