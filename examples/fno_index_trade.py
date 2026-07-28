#!/usr/bin/env python3
"""An index derivatives run: Nifty / Bank Nifty with defined-risk structures.

    python examples/fno_index_trade.py BANKNIFTY

Indices have no cash instrument a retail account can hold, so the trader picks
between futures and an option spread. Watch the sizing output: one Bank Nifty
lot is often larger than a retail risk budget allows, and the honest answer in
that case is zero lots.
"""

from __future__ import annotations

import sys

from tradingagents_in import IndianTradingAgents
from tradingagents_in.dataflows.calendar_in import session_context
from tradingagents_in.utils.formatting import inr_plain


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    ctx = session_context()
    print(f"Session: {ctx.phase.value} | weekly expiry {ctx.weekly_expiry} "
          f"({ctx.days_to_weekly_expiry} sessions away)\n")

    app = IndianTradingAgents(
        config={
            "capital_inr": 5_000_000,      # index lots need a real book behind them
            "risk_per_trade_pct": 1.5,
            "allow_fno": True,
            "allow_overnight_fno": True,
            "allow_expiry_day_entry": False,
            "llm_provider": "offline",
        }
    )
    decision = app.analyse(symbol)
    proposal = decision.proposal

    print(decision.headline())
    if proposal and proposal.is_actionable:
        print(f"\nInstrument: {proposal.instrument.value} / {proposal.product_type.value}")
        if proposal.option_legs:
            print(f"Strategy:   {proposal.option_strategy}")
            for leg in proposal.option_legs:
                print(f"  {leg}  @ ~Rs {leg.estimated_premium}")
        if proposal.futures:
            print(f"Contract:   {proposal.futures.contract_label} x {proposal.futures.lots} lot(s)")
        print(f"Deployed:   {inr_plain(proposal.capital_deployed_inr)}")
        print(f"Max loss:   {inr_plain(decision.capital_at_risk_inr)}")
    else:
        print(f"\nNo position: {(decision.portfolio_decision or {}).get('portfolio_reasoning', '')}")

    for note in decision.risk_notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
