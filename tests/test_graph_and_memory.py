"""End-to-end graph runs, the memory log, and configuration resolution."""

from __future__ import annotations

import json

import pytest

from tradingagents_in.config.default_config import DEFAULT_CONFIG, build_config
from tradingagents_in.graph.pipeline import SequentialPipeline
from tradingagents_in.graph.reporting import build_final_decision, render_report
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.graph.setup import mermaid, recursion_limit
from tradingagents_in.memory import MemoryRecord, TradingMemory
from tradingagents_in.models.enums import Action, RiskVerdict
from tradingagents_in.models.risk import PortfolioSnapshot


class TestPipeline:
    def test_sequential_run_populates_every_stage(self, config):
        pipeline = SequentialPipeline(config)
        state = pipeline.invoke({"symbol": "RELIANCE.NS", "run_id": "test-1"})

        for key in (
            "symbol_ref", "fundamentals_report", "sentiment_report",
            "news_macro_report", "technical_report", "debate", "investment_plan",
            "trade_proposal", "risk_assessment", "portfolio_decision",
        ):
            assert state.get(key) is not None, f"{key} missing from the final state"
        assert not state["errors"]

    def test_debate_runs_the_configured_number_of_rounds(self, config_factory):
        config = config_factory(max_debate_rounds=2)
        state = SequentialPipeline(config).invoke({"symbol": "TCS.NS", "run_id": "test-2"})
        debate = state["debate"]
        assert debate.rounds_completed >= 1
        assert len(debate.turns) == debate.rounds_completed * 2
        assert {t.side for t in debate.turns} == {"bull", "bear"}

    def test_node_failure_is_captured_not_raised(self, config, monkeypatch):
        from tradingagents_in.agents.analysts import technical

        def boom(self, bundle, state):
            raise RuntimeError("simulated indicator failure")

        monkeypatch.setattr(technical.TechnicalAnalyst, "deterministic_draft", boom)

        state = SequentialPipeline(config).invoke({"symbol": "INFY.NS", "run_id": "test-3"})
        assert any("simulated indicator failure" in e for e in state["errors"])
        # The run still reaches a decision without the technical leg.
        assert state.get("portfolio_decision") is not None

    def test_recursion_limit_leaves_headroom_for_the_cycles(self):
        config = build_config({"max_debate_rounds": 5, "max_risk_rounds": 3})
        assert recursion_limit(config) > 5 * 2 + 3 * 2 + 6


class TestRunner:
    def test_analyse_returns_a_complete_decision(self, analysed):
        _, decision = analysed
        assert decision.run_id
        assert decision.symbol == "RELIANCE"
        assert decision.yf_symbol == "RELIANCE.NS"
        assert decision.exchange == "NSE"
        assert isinstance(decision.final_action, Action)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.analyst_reports and len(decision.analyst_reports) == 4
        assert decision.debate_transcript
        assert decision.headline()

    def test_decision_is_json_serialisable(self, analysed):
        _, decision = analysed
        payload = json.loads(decision.model_dump_json(exclude={"proposal"}))
        assert payload["run_id"] == decision.run_id
        assert "config_digest" in payload

    def test_report_contains_every_section(self, analysed):
        app, decision = analysed
        report = app.report(decision)
        for heading in (
            "## Decision", "## Risk assessment", "## Research view",
            "## Debate transcript", "## Analyst reports", "## Run audit",
        ):
            assert heading in report
        assert "not investment advice" in report.lower()

    def test_approved_decision_never_exceeds_the_risk_budget(self, analysed, config):
        _, decision = analysed
        if decision.approved:
            budget = config["capital_inr"] * config["risk_per_trade_pct"] / 100
            assert decision.capital_at_risk_inr <= budget * 1.01
            assert decision.position_size_pct <= config["max_position_pct"] + 1e-6

    def test_existing_book_constrains_the_decision(self, config):
        app = IndianTradingAgents(config=config, use_langgraph=False)
        full_book = PortfolioSnapshot(
            capital_inr=config["capital_inr"],
            cash_inr=0.0,
            portfolio_heat_pct=6.0,
            open_positions=12,
            current_drawdown_pct=16.0,
        )
        decision = app.analyse("RELIANCE.NS", portfolio=full_book, save=False)
        assert not decision.approved
        assert decision.final_action == Action.HOLD

    def test_langgraph_and_sequential_agree_on_the_action(self, config):
        seq = IndianTradingAgents(config=config, use_langgraph=False).analyse(
            "RELIANCE.NS", save=False
        )
        graph = IndianTradingAgents(config=config, use_langgraph=True).analyse(
            "RELIANCE.NS", save=False
        )
        # Same data, same deterministic agents: the backend must not change the call.
        assert seq.final_action == graph.final_action
        assert seq.approved == graph.approved

    def test_batch_survives_a_bad_symbol(self, config):
        app = IndianTradingAgents(config=config, use_langgraph=False)
        results = app.analyse_many(["RELIANCE.NS", "!!!invalid!!!"], save=False)
        assert "RELIANCE.NS" in results

    def test_batch_accumulates_the_book_across_symbols(self, config):
        """The sector cap must bind across a batch, not just within one trade."""
        app = IndianTradingAgents(config=config, use_langgraph=False)
        it_names = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]

        accumulated = app.analyse_many(it_names, save=False, accumulate=True)
        independent = app.analyse_many(it_names, save=False, accumulate=False)

        it_exposure = sum(
            d.position_size_pct for d in accumulated.values() if d.approved
        )
        assert it_exposure <= config["max_sector_pct"] + 1e-6

        # Judged in isolation every name clears the single-name limit, and the
        # book ends up concentrated. That is the failure mode being prevented.
        assert sum(d.position_size_pct for d in independent.values() if d.approved) > it_exposure

    def test_batch_respects_the_portfolio_heat_budget(self, config):
        app = IndianTradingAgents(config=config, use_langgraph=False)
        results = app.analyse_many(
            ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS",
             "SUNPHARMA.NS", "MARUTI.NS", "TATASTEEL.NS"],
            save=False,
        )
        total_heat = sum(
            d.capital_at_risk_inr for d in results.values() if d.approved
        ) / config["capital_inr"] * 100
        assert total_heat <= config["max_portfolio_heat_pct"] + 1e-6

    def test_results_are_written_to_disk(self, config_factory, tmp_path):
        config = config_factory(
            results_dir=str(tmp_path / "results"),
            memory_file=str(tmp_path / "trading_memory.md"),
            enable_memory=True,
        )
        app = IndianTradingAgents(config=config, use_langgraph=False)
        decision = app.analyse("RELIANCE.NS", save=True)

        written = list((tmp_path / "results").rglob(f"{decision.run_id}.*"))
        assert {p.suffix for p in written} == {".md", ".json"}
        assert (tmp_path / "trading_memory.md").exists()


class TestMemory:
    def _record(self, run_id="RUN-1", symbol="RELIANCE.NS", **kwargs) -> MemoryRecord:
        defaults = dict(
            run_id=run_id, timestamp="2026-07-28 10:30", symbol=symbol,
            sector="Energy", action="BUY", approved=True, confidence=0.7,
            position_size_pct=4.0, capital_at_risk_inr=9_800.0,
            thesis="Momentum with fundamental support",
        )
        defaults.update(kwargs)
        return MemoryRecord(**defaults)

    def test_append_writes_both_formats(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        memory.append(self._record())

        assert memory.md_path.exists() and memory.jsonl_path.exists()
        assert "RELIANCE.NS" in memory.md_path.read_text()
        assert len(memory.load()) == 1

    def test_recall_prefers_the_same_symbol_then_the_sector(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        memory.append(self._record("R-1", "TCS.NS", sector="IT", timestamp="2026-07-01 10:00"))
        memory.append(self._record("R-2", "INFY.NS", sector="IT", timestamp="2026-07-02 10:00"))
        memory.append(self._record("R-3", "TCS.NS", sector="IT", timestamp="2026-07-20 10:00"))

        recalled = memory.recall("TCS.NS", sector="IT", limit=3)
        assert [r.run_id for r in recalled[:2]] == ["R-3", "R-1"]  # same symbol, newest first

    def test_outcome_updates_the_record_and_the_log(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        memory.append(self._record("R-9"))

        assert memory.record_outcome("R-9", "win", 18_500, lesson="Trail on the supertrend")
        updated = next(r for r in memory.load() if r.run_id == "R-9")
        assert updated.outcome == "win" and updated.realised_pnl_inr == 18_500
        assert "Trail on the supertrend" in memory.md_path.read_text()

    def test_unknown_run_id_is_reported_not_swallowed(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        assert memory.record_outcome("nope", "win") is False

    def test_stats_track_the_hit_rate(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        memory.append(self._record("A"))
        memory.append(self._record("B"))
        memory.record_outcome("A", "win", 10_000)
        memory.record_outcome("B", "loss", -5_000)

        stats = memory.stats()
        assert stats["closed"] == 2 and stats["wins"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["net_pnl_inr"] == 5_000

    def test_corrupt_line_does_not_break_recall(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        memory.append(self._record("A"))
        with memory.jsonl_path.open("a") as fh:
            fh.write("{ this is not json\n")
        assert len(memory.load()) == 1

    def test_context_is_empty_when_there_is_no_history(self, tmp_path):
        memory = TradingMemory({"memory_file": str(tmp_path / "mem.md")})
        assert memory.context("RELIANCE.NS") == ""


class TestConfig:
    def test_defaults_are_valid(self):
        config = build_config({})
        assert config["timezone"] == "Asia/Kolkata"
        assert config["currency"] == "INR"
        assert config["capital_inr"] > 0

    def test_overrides_win_over_defaults(self):
        config = build_config({"capital_inr": 250_000, "max_debate_rounds": 4})
        assert config["capital_inr"] == 250_000
        assert config["max_debate_rounds"] == 4

    def test_nested_dicts_merge_rather_than_replace(self):
        config = build_config({"costs": {"brokerage_pct_intraday": 0.05}})
        assert config["costs"]["brokerage_pct_intraday"] == 0.05
        # The untouched keys must survive the merge.
        assert "stt_pct_delivery_buy" in config["costs"]

    def test_institutional_profile_tightens_limits(self):
        retail = build_config({"profile": "retail"})
        inst = build_config({"profile": "institutional"})
        assert inst["risk_per_trade_pct"] < retail["risk_per_trade_pct"]
        assert inst["max_position_pct"] < retail["max_position_pct"]
        assert inst["max_open_positions"] > retail["max_open_positions"]

    def test_explicit_override_beats_the_profile(self):
        config = build_config({"profile": "institutional", "max_position_pct": 8.0})
        assert config["max_position_pct"] == 8.0

    @pytest.mark.parametrize(
        "bad",
        [
            {"capital_inr": 0},
            {"risk_per_trade_pct": 25},
            {"max_position_pct": 0},
            {"max_debate_rounds": 0},
            {"profile": "hedge-fund"},
            {"selected_analysts": ["astrology"]},
        ],
    )
    def test_invalid_configuration_is_rejected(self, bad):
        with pytest.raises(ValueError):
            build_config(bad)

    def test_env_vars_are_applied(self, monkeypatch):
        monkeypatch.setenv("ITA_CAPITAL_INR", "750000")
        monkeypatch.setenv("ITA_ALLOW_FNO", "false")
        config = build_config({})
        assert config["capital_inr"] == 750_000
        assert config["allow_fno"] is False

    def test_default_config_is_not_mutated_by_a_build(self):
        before = DEFAULT_CONFIG["capital_inr"]
        build_config({"capital_inr": 1})
        assert DEFAULT_CONFIG["capital_inr"] == before


class TestReporting:
    def test_mermaid_diagram_names_every_stage(self):
        diagram = mermaid()
        for node in ("Fundamentals Analyst", "Bull Researcher", "Bear Researcher",
                     "Research Manager", "Trader", "Risk Manager", "Portfolio Manager"):
            assert node in diagram

    def test_final_decision_survives_an_empty_state(self, config):
        final = build_final_decision({"symbol": "RELIANCE.NS"}, config)
        assert final.final_action == Action.HOLD
        assert not final.approved
        assert render_report({}, final, config)

    def test_risk_verdict_is_recorded_in_the_decision(self, analysed):
        _, decision = analysed
        assert decision.risk_assessment["verdict"] in {v.value for v in RiskVerdict}
