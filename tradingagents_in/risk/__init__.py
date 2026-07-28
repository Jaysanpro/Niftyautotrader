"""Deterministic risk engine: costs, sizing and the SEBI/exchange rulebook.

Nothing in this package calls an LLM. The risk-manager agent consults it and
may be *more* conservative than it says; it can never overturn a BLOCKER.
"""

from tradingagents_in.risk.costs import (
    DEFAULT_RATES,
    breakeven_move_pct,
    estimate_costs,
    net_reward_risk,
)
from tradingagents_in.risk.rules import (
    check_all,
    check_coherence,
    check_portfolio_risk,
    check_portfolio_state,
    check_position_risk,
    check_tradability,
    sebi_compliance_notes,
    stress_test,
    summarise,
)
from tradingagents_in.risk.sizing import (
    SizingResult,
    risk_budget_inr,
    size_cash_position,
    size_for_proposal,
    size_futures_position,
    size_option_position,
)

__all__ = [
    "DEFAULT_RATES",
    "SizingResult",
    "breakeven_move_pct",
    "check_all",
    "check_coherence",
    "check_portfolio_risk",
    "check_portfolio_state",
    "check_position_risk",
    "check_tradability",
    "estimate_costs",
    "net_reward_risk",
    "risk_budget_inr",
    "sebi_compliance_notes",
    "size_cash_position",
    "size_for_proposal",
    "size_futures_position",
    "size_option_position",
    "stress_test",
    "summarise",
]
