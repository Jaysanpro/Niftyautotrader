"""LangGraph assembly.

Topology::

    START
      -> gather            (one fetch pass; every agent reads the same snapshot)
      -> memory            (recall comparable past decisions)
      -> [fundamentals | sentiment | news_macro | technical]   (parallel)
      -> analysts_join     (barrier)
      -> bull -> bear -+-> (another round?) -> bull
                       +-> research_manager
      -> trader
      -> risk_manager -+-> (changes requested, rounds left?) -> trader
                       +-> portfolio_manager
      -> END

The analysts fan out because they share no state and each one is a network-bound
call; running them sequentially triples wall-clock for no benefit. Everything
after the join is sequential, because each stage genuinely depends on the last.

The debate loop is a real cycle in the graph, not an unrolled chain, so
``max_debate_rounds`` is a runtime knob rather than a graph rebuild.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.graph import nodes
from tradingagents_in.models.state import TradingState
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)


class LangGraphUnavailable(RuntimeError):
    """Raised when langgraph is not installed."""


def build_graph(config: dict[str, Any], checkpointer: Any | None = None) -> Any:
    """Compile the trading graph. Raises :class:`LangGraphUnavailable` if absent."""
    try:
        from langgraph.graph import END, START, StateGraph  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised by the fallback path
        raise LangGraphUnavailable(
            "langgraph is not installed. Install with `pip install 'tradingagents-in[graph]'`, "
            "or use the built-in sequential executor (the runner falls back automatically)."
        ) from exc

    selected = [a for a in config.get("selected_analysts", []) if a in nodes.ANALYSTS]
    if not selected:
        raise ValueError("no valid analysts selected")

    graph = StateGraph(TradingState)

    graph.add_node("gather", nodes.make_gather_node(config))
    graph.add_node("memory", nodes.make_memory_node(config))
    for name in selected:
        graph.add_node(f"analyst_{name}", nodes.make_analyst_node(name, config))
    graph.add_node("analysts_join", nodes.analysts_join)
    graph.add_node("bull_researcher", nodes.make_debater_node("bull", config))
    graph.add_node("bear_researcher", nodes.make_debater_node("bear", config))
    graph.add_node("research_manager", nodes.make_research_manager_node(config))
    graph.add_node("trader", nodes.make_trader_node(config))
    graph.add_node("risk_manager", nodes.make_risk_manager_node(config))
    graph.add_node("portfolio_manager", nodes.make_portfolio_manager_node(config))

    graph.add_edge(START, "gather")
    graph.add_edge("gather", "memory")
    for name in selected:
        graph.add_edge("memory", f"analyst_{name}")
        graph.add_edge(f"analyst_{name}", "analysts_join")

    graph.add_edge("analysts_join", "bull_researcher")
    graph.add_edge("bull_researcher", "bear_researcher")
    graph.add_conditional_edges(
        "bear_researcher",
        nodes.should_continue_debate,
        {"bull_researcher": "bull_researcher", "research_manager": "research_manager"},
    )
    graph.add_edge("research_manager", "trader")
    graph.add_edge("trader", "risk_manager")
    graph.add_conditional_edges(
        "risk_manager",
        nodes.make_risk_router(config),
        {"trader": "trader", "portfolio_manager": "portfolio_manager"},
    )
    graph.add_edge("portfolio_manager", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    log.debug("graph compiled with analysts: %s", selected)
    return compiled


def recursion_limit(config: dict[str, Any]) -> int:
    """Headroom for the debate and risk cycles, plus the fixed stages."""
    debate_steps = 2 * int(config.get("max_debate_rounds", 2))
    risk_steps = 2 * int(config.get("max_risk_rounds", 1))
    fixed = 6 + len(config.get("selected_analysts", []))
    return max(25, (debate_steps + risk_steps + fixed) * 2)


def mermaid() -> str:
    """The topology as a Mermaid diagram, for docs and for `ita graph`."""
    return """\
flowchart TD
    START([Run: symbol + config]) --> G[gather<br/>one fetch pass]
    G --> M[memory recall<br/>trading_memory.md]

    M --> A1[Fundamentals Analyst<br/>Ind-AS, promoter, ROCE]
    M --> A2[Sentiment Analyst<br/>ET, MC, Mint, X, Reddit]
    M --> A3[News &amp; Macro Analyst<br/>RBI, Budget, FII/DII, crude]
    M --> A4[Technical Analyst<br/>RSI, MACD, Supertrend, VWAP]

    A1 --> J{{analysts join}}
    A2 --> J
    A3 --> J
    A4 --> J

    J --> BULL[Bull Researcher]
    BULL --> BEAR[Bear Researcher]
    BEAR -->|rounds remaining| BULL
    BEAR -->|debate closed| RM[Research Manager<br/>adjudicates]

    RM --> T[Trader<br/>entry / stop / targets / size]
    T --> RISK[Risk Manager<br/>+ deterministic rulebook]
    RISK -->|changes required| T
    RISK -->|verdict final| PM[Portfolio Manager<br/>book-level approve or reject]
    PM --> OUT([FinalDecision + memory write])
"""
