"""Sequential executor: the same pipeline without LangGraph.

LangGraph is an optional extra. When it is not installed - a locked-down
environment, a slim container, a dependency conflict - the desk should still
run. This module walks the identical node functions in the identical order,
honouring the same debate and risk cycles.

What it does not provide: parallel analyst execution, checkpointing, and
streaming. Those are the reasons to install the extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tradingagents_in.graph import nodes
from tradingagents_in.models.state import ANALYST_STATE_KEYS, append_list, merge_dicts
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)


def _apply(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge a node's partial update using the state's declared reducers."""
    for key, value in update.items():
        if key == "errors":
            state[key] = append_list(state.get(key), value)
        elif key == "telemetry":
            state[key] = merge_dicts(state.get(key), value)
        else:
            state[key] = value
    return state


class SequentialPipeline:
    """Drop-in replacement for a compiled LangGraph app."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.selected = [a for a in config.get("selected_analysts", []) if a in nodes.ANALYSTS]
        if not self.selected:
            raise ValueError("no valid analysts selected")

        self.gather = nodes.make_gather_node(config)
        self.memory = nodes.make_memory_node(config)
        self.analysts: dict[str, Callable] = {
            name: nodes.make_analyst_node(name, config) for name in self.selected
        }
        self.bull = nodes.make_debater_node("bull", config)
        self.bear = nodes.make_debater_node("bear", config)
        self.research_manager = nodes.make_research_manager_node(config)
        self.trader = nodes.make_trader_node(config)
        self.risk_manager = nodes.make_risk_manager_node(config)
        self.portfolio_manager = nodes.make_portfolio_manager_node(config)
        self.risk_router = nodes.make_risk_router(config)

    def invoke(self, state: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        state = dict(state)
        state.setdefault("errors", [])
        state.setdefault("telemetry", {})

        _apply(state, self.gather(state))
        _apply(state, self.memory(state))

        for name in self.selected:
            _apply(state, self.analysts[name](state))
        _apply(state, nodes.analysts_join(state))

        max_rounds = int(self.config.get("max_debate_rounds", 2))
        for _ in range(max_rounds):
            _apply(state, self.bull(state))
            _apply(state, self.bear(state))
            if nodes.should_continue_debate(state) == "research_manager":
                break

        _apply(state, self.research_manager(state))

        # Trader -> risk manager, with a bounded revision loop.
        max_risk_rounds = int(self.config.get("max_risk_rounds", 1))
        for _ in range(max_risk_rounds + 1):
            _apply(state, self.trader(state))
            _apply(state, self.risk_manager(state))
            if self.risk_router(state) == "portfolio_manager":
                break

        _apply(state, self.portfolio_manager(state))
        return state

    def stream(self, state: dict[str, Any], *_args: Any, **_kwargs: Any):
        """Minimal stream shim so callers can use one code path."""
        result = self.invoke(state)
        for key in ("gather", "memory", *ANALYST_STATE_KEYS.values()):
            if key in result:
                yield {key: result[key]}
        yield result
