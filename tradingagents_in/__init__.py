"""IndianTradingAgents (TradingAgents-IN).

A multi-agent LLM research framework for Indian equity markets (NSE + BSE).

Quick start::

    from tradingagents_in import IndianTradingAgents, DEFAULT_CONFIG

    app = IndianTradingAgents(config={**DEFAULT_CONFIG, "capital_inr": 500_000})
    decision = app.analyse("RELIANCE.NS")
    print(decision.final_action, decision.confidence)
"""

from __future__ import annotations

__version__ = "0.1.0"

from tradingagents_in.config.default_config import DEFAULT_CONFIG, build_config

__all__ = [
    "DEFAULT_CONFIG",
    "build_config",
    "IndianTradingAgents",
    "__version__",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    # The graph pulls in langgraph/langchain. Importing lazily keeps
    # `import tradingagents_in` cheap and usable without those extras.
    if name == "IndianTradingAgents":
        from tradingagents_in.graph.runner import IndianTradingAgents

        return IndianTradingAgents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
