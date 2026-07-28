"""``IndianTradingAgents`` - the public Python API.

    from tradingagents_in import IndianTradingAgents

    app = IndianTradingAgents(config={"capital_inr": 500_000})
    decision = app.analyse("RELIANCE.NS")
    print(decision.headline())
    print(app.report(decision))

One instance can analyse many symbols; the graph is compiled once and the
per-run state is created fresh each time, so a batch run does not leak state
between names.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.pipeline import SequentialPipeline
from tradingagents_in.graph.reporting import (
    build_final_decision,
    render_report,
    render_summary,
)
from tradingagents_in.graph.setup import LangGraphUnavailable, build_graph, recursion_limit
from tradingagents_in.memory import TradingMemory, build_record
from tradingagents_in.models.risk import FinalDecision, PortfolioSnapshot
from tradingagents_in.utils.logging import get_logger, setup_logging
from tradingagents_in.utils.timeutils import now_ist

log = get_logger(__name__)


class IndianTradingAgents:
    """The framework entry point."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        portfolio: PortfolioSnapshot | None = None,
        use_langgraph: bool = True,
    ):
        self.config = build_config(config)
        setup_logging(self.config.get("log_level", "INFO"))
        self.portfolio = portfolio
        self.memory = TradingMemory(self.config)
        self._checkpointer = _make_checkpointer(self.config)

        self.backend = "sequential"
        self.app: Any
        if use_langgraph:
            try:
                self.app = build_graph(self.config, self._checkpointer)
                self.backend = "langgraph"
            except LangGraphUnavailable as exc:
                log.warning("%s Falling back to the sequential executor.", exc)
                self.app = SequentialPipeline(self.config)
        else:
            self.app = SequentialPipeline(self.config)

        log.info(
            "IndianTradingAgents ready (%s backend, %s)",
            self.backend, _describe_llm(self.config),
        )

    # --------------------------------------------------------------- API
    def analyse(
        self,
        symbol: str,
        *,
        portfolio: PortfolioSnapshot | None = None,
        run_id: str | None = None,
        save: bool = True,
    ) -> FinalDecision:
        """Run the full agent pipeline on one symbol."""
        started = time.time()
        run_id = run_id or f"{symbol.replace('.', '_').upper()}-{uuid.uuid4().hex[:8]}"

        initial: dict[str, Any] = {
            "symbol": symbol,
            "run_id": run_id,
            "config": self.config,
            "portfolio": portfolio or self.portfolio or _flat_book(self.config),
            "errors": [],
            "telemetry": {},
        }

        invoke_kwargs: dict[str, Any] = {}
        if self.backend == "langgraph":
            invoke_kwargs["config"] = {
                "recursion_limit": recursion_limit(self.config),
                "configurable": {"thread_id": run_id},
            }

        log.info("analysing %s (run %s)", symbol, run_id)
        state = self.app.invoke(initial, **invoke_kwargs)

        final = build_final_decision(state, self.config, time.time() - started)
        self._last_state = state

        if save:
            self._persist(state, final)
        return final

    # American spelling, because half the users will type it.
    analyze = analyse

    def analyse_many(
        self,
        symbols: list[str],
        *,
        save: bool = True,
        accumulate: bool = True,
    ) -> dict[str, FinalDecision]:
        """Sequentially analyse a list of symbols.

        Deliberately serial: the data sources are free public endpoints and
        hammering them in parallel is the fastest way to get rate-limited into
        a run with no data at all.

        With ``accumulate=True`` (the default) each approved position is added
        to a running book, so the sector caps and the portfolio heat budget bind
        across the batch. Without it, ten names in one sector would each be
        approved at the single-name limit and the desk would end the run 100%
        concentrated - every trade individually compliant, the portfolio
        indefensible.
        """
        out: dict[str, FinalDecision] = {}
        book = _copy_book(self.portfolio or _flat_book(self.config))

        for symbol in symbols:
            try:
                decision = self.analyse(symbol, portfolio=book, save=save)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop a batch
                log.error("analysis failed for %s: %s", symbol, exc)
                continue
            out[symbol] = decision
            if accumulate and decision.approved:
                book = _apply_to_book(book, decision, self.config)
        return out

    def report(self, decision: FinalDecision) -> str:
        """Full markdown report for a decision produced by this instance."""
        state = getattr(self, "_last_state", {})
        if state.get("run_id") != decision.run_id:
            state = {}
        return render_report(state, decision, self.config)

    def summary(self, decision: FinalDecision) -> str:
        return render_summary(decision)

    def record_outcome(
        self,
        run_id: str,
        outcome: str,
        realised_pnl_inr: float | None = None,
        notes: str | None = None,
        lesson: str | None = None,
    ) -> bool:
        """Close the loop on a past decision so future runs can learn from it."""
        return self.memory.record_outcome(run_id, outcome, realised_pnl_inr, notes, lesson)

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()

    # ---------------------------------------------------------- internals
    def _persist(self, state: dict[str, Any], final: FinalDecision) -> None:
        try:
            self.memory.append(build_record(final, self.config))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write trading memory: %s", exc)

        results_dir = Path(self.config.get("results_dir", "results"))
        try:
            target = results_dir / final.symbol / now_ist().strftime("%Y-%m-%d")
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{final.run_id}.md").write_text(
                render_report(state, final, self.config), encoding="utf-8"
            )
            (target / f"{final.run_id}.json").write_text(
                final.model_dump_json(indent=2, exclude={"proposal"}), encoding="utf-8"
            )
            log.info("results written to %s", target)
        except OSError as exc:
            log.warning("could not write results: %s", exc)


def _flat_book(config: dict[str, Any]) -> PortfolioSnapshot:
    capital = float(config.get("capital_inr", 1_000_000))
    return PortfolioSnapshot(capital_inr=capital, cash_inr=capital)


def _copy_book(book: PortfolioSnapshot) -> PortfolioSnapshot:
    return book.model_copy(deep=True)


def _apply_to_book(
    book: PortfolioSnapshot, decision: FinalDecision, config: dict[str, Any]
) -> PortfolioSnapshot:
    """Fold an approved decision into the running book for the next symbol."""
    updated = _copy_book(book)
    capital = float(config.get("capital_inr", updated.capital_inr or 1_000_000))
    deployed = float((decision.portfolio_decision or {}).get("capital_deployed_inr", 0.0))

    updated.deployed_inr += deployed
    updated.cash_inr = max(0.0, updated.cash_inr - deployed)
    updated.open_positions += 1
    updated.portfolio_heat_pct += decision.capital_at_risk_inr / capital * 100

    sector = decision.sector or "Unclassified"
    updated.sector_exposure_pct[sector] = (
        updated.sector_exposure_pct.get(sector, 0.0) + decision.position_size_pct
    )
    updated.symbol_exposure_pct[decision.symbol] = (
        updated.symbol_exposure_pct.get(decision.symbol, 0.0) + decision.position_size_pct
    )

    proposal = decision.trade_proposal or {}
    if proposal.get("instrument") in ("FUTURES", "OPTIONS") and (
        proposal.get("time_horizon") != "INTRADAY"
    ):
        updated.overnight_fno_exposure_pct += decision.position_size_pct
    return updated


def _make_checkpointer(config: dict[str, Any]) -> Any | None:
    """SQLite checkpointer for resumable runs, when enabled and installed."""
    if not config.get("enable_checkpoints"):
        return None
    try:
        import sqlite3  # noqa: PLC0415

        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415
    except ImportError:
        log.warning(
            "checkpointing requested but langgraph-checkpoint-sqlite is not installed; "
            "continuing without it"
        )
        return None

    path = Path(config.get("checkpoint_db", "checkpoints/graph.sqlite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    log.info("checkpointing to %s", path)
    return SqliteSaver(conn)


def _describe_llm(config: dict[str, Any]) -> str:
    from tradingagents_in.llm.provider import describe  # noqa: PLC0415

    try:
        return describe(config)
    except Exception:  # noqa: BLE001 - description must never break startup
        return str(config.get("llm_provider"))
