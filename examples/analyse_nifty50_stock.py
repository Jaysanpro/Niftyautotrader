#!/usr/bin/env python3
"""Analyse a single Nifty 50 stock and print the full report.

    export ANTHROPIC_API_KEY=...
    python examples/analyse_nifty50_stock.py RELIANCE.NS

With no API key set it falls back to the deterministic offline engine, which
still produces a complete, sized, risk-checked decision.
"""

from __future__ import annotations

import os
import sys

from tradingagents_in import IndianTradingAgents

NIFTY50_SAMPLE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"

    has_key = any(
        os.getenv(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    )
    config = {
        "capital_inr": 500_000,
        "profile": "retail",
        "risk_per_trade_pct": 1.0,
        "max_debate_rounds": 2,
    }
    if not has_key:
        print("No LLM key found - running the deterministic offline engine.\n")
        config["llm_provider"] = "offline"

    app = IndianTradingAgents(config=config)
    decision = app.analyse(symbol)

    print(app.report(decision))
    print("\n" + "=" * 70)
    print(decision.headline())
    return 0 if decision.approved else 1


if __name__ == "__main__":
    sys.exit(main())
