"""Graph nodes.

Every node has the same shape: ``(state) -> partial state``. That keeps them
runnable both inside LangGraph and, when LangGraph is not installed, inside the
sequential fallback executor in :mod:`tradingagents_in.graph.pipeline` - which
matters because a research desk should not lose its whole pipeline to a missing
optional dependency.

Nodes never raise. A failed node records the error in ``state["errors"]`` and
returns whatever it could compute, because a partial answer with an explicit
gap is more useful than a traceback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from tradingagents_in.agents.analysts import ANALYSTS
from tradingagents_in.agents.researchers import BearResearcher, BullResearcher, ResearchManager
from tradingagents_in.agents.risk_team import PortfolioManager, RiskManager
from tradingagents_in.agents.trader import Trader
from tradingagents_in.dataflows.toolkit import MarketBundle, Toolkit
from tradingagents_in.memory import TradingMemory
from tradingagents_in.models.enums import RiskVerdict
from tradingagents_in.models.research import DebateState
from tradingagents_in.models.risk import PortfolioSnapshot
from tradingagents_in.models.state import ANALYST_STATE_KEYS
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def _bundle(state: dict[str, Any]) -> MarketBundle:
    bundle = (state.get("market_bundle") or {}).get("bundle")
    if bundle is None:
        raise RuntimeError("market bundle missing: the gather node did not run")
    return bundle


def _timed(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run a node body, capturing latency and converting failure into state."""
    started = time.time()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - a node must never break the run
        log.exception("node %s failed", name)
        return {
            "errors": [f"{name}: {type(exc).__name__}: {exc}"],
            "telemetry": {name: {"ok": False, "ms": int((time.time() - started) * 1000)}},
        }
    result.setdefault("telemetry", {})[name] = {
        "ok": True,
        "ms": int((time.time() - started) * 1000),
    }
    return result


# ---------------------------------------------------------------- setup
def make_gather_node(config: dict[str, Any]) -> NodeFn:
    """Resolve the symbol and fetch every data source exactly once."""
    toolkit = Toolkit(config)

    def gather(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            symbol = state["symbol"]
            # Read per run, not from the node's build-time config: one graph
            # serves every date in a walk-forward sweep.
            bundle = toolkit.gather(symbol, as_of=state.get("config", {}).get("as_of"))
            return {
                "symbol_ref": bundle.ref,
                "market_bundle": {"bundle": bundle},
                "microstructure": bundle.micro,
                "trade_date": str(bundle.session.trade_date),
                "debate": DebateState(max_rounds=int(config.get("max_debate_rounds", 2))),
                "risk_round": 0,
                "portfolio": state.get("portfolio") or _default_portfolio(config),
            }

        return _timed("gather", body)

    return gather


def make_memory_node(config: dict[str, Any]) -> NodeFn:
    """Load comparable past decisions into the prompt context."""
    memory = TradingMemory(config)

    def recall(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            ref = state.get("symbol_ref")
            context = memory.context(
                symbol=ref.yf_symbol if ref else state["symbol"],
                sector=ref.sector if ref else None,
            )
            return {"memory_context": context}

        return _timed("memory", body)

    return recall


# -------------------------------------------------------------- analysts
def make_analyst_node(name: str, config: dict[str, Any]) -> NodeFn:
    agent = ANALYSTS[name](config)
    state_key = ANALYST_STATE_KEYS[name]

    def analyst(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            report = agent.run(_bundle(state), state)
            log.info("%s: %s (%.0f%%)", name, report.stance.value, report.confidence * 100)
            return {state_key: report}

        return _timed(f"analyst.{name}", body)

    return analyst


def analysts_join(state: dict[str, Any]) -> dict[str, Any]:
    """Barrier node: exists so the debate starts only once all analysts land."""
    ran = [key for key in ANALYST_STATE_KEYS.values() if state.get(key) is not None]
    return {"telemetry": {"analysts_completed": len(ran)}}


# ------------------------------------------------------------- research
def make_debater_node(side: str, config: dict[str, Any]) -> NodeFn:
    agent = (BullResearcher if side == "bull" else BearResearcher)(config)

    def debater(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            debate: DebateState = state.get("debate") or DebateState(
                max_rounds=int(config.get("max_debate_rounds", 2))
            )
            turn = agent.run(_bundle(state), state)
            turns = [*debate.turns, turn]
            # The round closes when the bear has answered the bull.
            rounds = debate.rounds_completed + (1 if side == "bear" else 0)
            return {
                "debate": DebateState(
                    rounds_completed=rounds, max_rounds=debate.max_rounds, turns=turns
                )
            }

        return _timed(f"researcher.{side}", body)

    return debater


def make_research_manager_node(config: dict[str, Any]) -> NodeFn:
    agent = ResearchManager(config)

    def manager(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            plan = agent.run(_bundle(state), state)
            log.info(
                "research manager: %s (conviction %.0f%%, winner %s)",
                plan.stance.value, plan.conviction * 100, plan.winning_side,
            )
            return {"investment_plan": plan}

        return _timed("research_manager", body)

    return manager


# ---------------------------------------------------------------- trade
def make_trader_node(config: dict[str, Any]) -> NodeFn:
    agent = Trader(config)

    def trader(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            proposal = agent.run(_bundle(state), state)
            log.info(
                "trader: %s %s %.2f%%",
                proposal.action.value, proposal.instrument.value, proposal.position_size_pct,
            )
            return {"trade_proposal": proposal}

        return _timed("trader", body)

    return trader


def make_risk_manager_node(config: dict[str, Any]) -> NodeFn:
    agent = RiskManager(config)

    def risk(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            assessment = agent.run(_bundle(state), state)
            log.info(
                "risk manager: %s, approved %.2f%%",
                assessment.verdict.value, assessment.approved_position_size_pct,
            )
            return {
                "risk_assessment": assessment,
                "risk_round": int(state.get("risk_round", 0)) + 1,
            }

        return _timed("risk_manager", body)

    return risk


def make_portfolio_manager_node(config: dict[str, Any]) -> NodeFn:
    agent = PortfolioManager(config)

    def portfolio(state: dict[str, Any]) -> dict[str, Any]:
        def body() -> dict[str, Any]:
            decision = agent.run(_bundle(state), state)
            log.info(
                "portfolio manager: %s %s at %.2f%%",
                "APPROVED" if decision.approved else "REJECTED",
                decision.final_action.value, decision.final_position_size_pct,
            )
            return {"portfolio_decision": decision}

        return _timed("portfolio_manager", body)

    return portfolio


# ------------------------------------------------------------ conditions
def should_continue_debate(state: dict[str, Any]) -> str:
    """Another round, or hand the transcript to the research manager?"""
    debate: DebateState | None = state.get("debate")
    if debate is None:
        return "research_manager"
    if debate.rounds_completed >= debate.max_rounds:
        return "research_manager"
    # A debate where neither side has conceded anything by the second round is
    # two monologues; nothing further is learned by continuing it.
    if debate.rounds_completed >= 2 and not any(t.conceded for t in debate.turns):
        log.info("ending debate early: neither side engaged with the other")
        return "research_manager"
    return "bull_researcher"


def make_risk_router(config: dict[str, Any]) -> Callable[[dict[str, Any]], str]:
    """Send a change-requested proposal back to the trader, once."""
    max_rounds = int(config.get("max_risk_rounds", 1))

    def route(state: dict[str, Any]) -> str:
        assessment = state.get("risk_assessment")
        rounds = int(state.get("risk_round", 0))
        if (
            assessment is not None
            and assessment.verdict == RiskVerdict.APPROVE_WITH_CHANGES
            and rounds < max_rounds
        ):
            return "trader"
        return "portfolio_manager"

    return route


def _default_portfolio(config: dict[str, Any]) -> PortfolioSnapshot:
    """A flat book with the mandate's capital, for a standalone run."""
    capital = float(config.get("capital_inr", 1_000_000))
    return PortfolioSnapshot(capital_inr=capital, cash_inr=capital)
