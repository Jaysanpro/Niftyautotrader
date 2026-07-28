#!/usr/bin/env python3
"""Screen an index constituent list and rank the approved ideas.

    python examples/screen_index.py NIFTY_IT

Positions accumulate across the batch, so the sector cap and the portfolio heat
budget bind the way they would on a real book: the first names through the door
consume the risk budget, and later ones get trimmed or declined.
"""

from __future__ import annotations

import sys

from tradingagents_in import IndianTradingAgents
from tradingagents_in.config.universe import INDEX_CONSTITUENTS
from tradingagents_in.utils.formatting import inr_plain


def main() -> int:
    key = (sys.argv[1] if len(sys.argv) > 1 else "NIFTY_IT")
    key = key.upper().replace(" ", "").replace("_", "")
    members = INDEX_CONSTITUENTS.get(key)
    if not members:
        print(f"Unknown index {key!r}. Known: {', '.join(sorted(INDEX_CONSTITUENTS))}")
        return 1

    symbols = [f"{s}.NS" for s in members]
    app = IndianTradingAgents(
        config={"capital_inr": 2_000_000, "max_debate_rounds": 1, "llm_provider": "offline"}
    )
    results = app.analyse_many(symbols)

    approved = sorted(
        (d for d in results.values() if d.approved),
        key=lambda d: d.confidence,
        reverse=True,
    )
    print(f"\n{len(approved)} of {len(results)} names cleared risk\n")
    for decision in approved:
        print(
            f"  {decision.symbol:<12} {decision.final_action.value:<5} "
            f"{decision.position_size_pct:5.2f}%  risk {inr_plain(decision.capital_at_risk_inr, 0)}"
            f"  conf {decision.confidence:.0%}"
        )

    declined = [d for d in results.values() if not d.approved]
    if declined:
        print("\nDeclined, with the binding reason:\n")
        for decision in declined:
            reasons = (decision.portfolio_decision or {}).get("rejection_reasons") or [
                (decision.portfolio_decision or {}).get("portfolio_reasoning", "")
            ]
            print(f"  {decision.symbol:<12} {reasons[0][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
