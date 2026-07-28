#!/usr/bin/env python3
"""The pure Python API: custom config, an existing book, and closing the loop.

Shows the three things a real integration needs beyond `analyse()`:

1. passing the book you actually hold, so concentration limits mean something;
2. reading the structured Pydantic output instead of the printed report;
3. recording the outcome later, which is what turns the decision log into
   something the next run can learn from.
"""

from __future__ import annotations

from tradingagents_in import IndianTradingAgents
from tradingagents_in.models.risk import PortfolioSnapshot

# 1. Configuration. Pass only the keys you want to change: they are layered
#    over DEFAULT_CONFIG and over anything in .env. Spreading `**DEFAULT_CONFIG`
#    into this dict would make *every* key an explicit override and silently
#    disable your .env, which is rarely what you want.
config = {
    "llm_provider": "offline",        # swap for "anthropic", "openai", ...
    "capital_inr": 2_500_000,
    "profile": "institutional",       # tighter single-name and heat limits
    "max_debate_rounds": 3,           # deeper research
    "selected_analysts": ["fundamentals", "technical", "news_macro"],
    "min_reward_risk": 2.0,
    # Nested dicts merge rather than replace, so this keeps the other tiers.
    "agent_model_tier": {"trader": "deep"},
}

# 2. The book you are actually running. Without this every trade is judged
#    against an empty portfolio, which is how books end up 60% in one sector.
book = PortfolioSnapshot(
    capital_inr=config["capital_inr"],
    deployed_inr=800_000,
    cash_inr=1_700_000,
    open_positions=3,
    portfolio_heat_pct=1.8,
    current_drawdown_pct=2.4,
    sector_exposure_pct={"Information Technology": 14.0, "Banking": 18.0},
    symbol_exposure_pct={"INFY": 6.0, "HDFCBANK": 9.0},
)

app = IndianTradingAgents(config=config, portfolio=book)
# save=True writes the report to results/ and appends to trading_memory.md.
# record_outcome() below can only find runs that were saved.
decision = app.analyse("TCS.NS", save=True)

# 3. Structured output - every field is a validated Pydantic model.
print(decision.headline())
print(f"approved={decision.approved} confidence={decision.confidence:.0%}")

plan = decision.investment_plan
print(f"research: {plan.get('stance')} / conviction {plan.get('conviction')}")

risk = decision.risk_assessment
print(f"risk verdict: {risk.get('verdict')}")
for breach in risk.get("breaches", []):
    print(f"  [{breach['severity']}] {breach['rule']}: {breach['detail'][:80]}")

proposal = decision.proposal
if proposal and proposal.is_actionable:
    print(f"order: {proposal.action.value} {proposal.quantity} @ {proposal.entry_zone}, "
          f"stop {proposal.stop_loss}")

# 4. Later, once you know how it went. This is the only way the framework
#    accumulates anything resembling experience.
recorded = app.record_outcome(
    decision.run_id,
    outcome="win",
    realised_pnl_inr=18_500,
    notes="First target hit on day 4; trailed the rest out on the supertrend flip.",
    lesson="The 1.5R first target paid for the trade before the expiry-week chop.",
)
print(f"outcome recorded: {recorded}")
print(app.stats())
