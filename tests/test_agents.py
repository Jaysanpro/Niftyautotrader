"""Agent behaviour, exercised through the deterministic (offline) engine.

Every agent must produce a genuinely usable output with no LLM at all - that is
what makes the LLM an improvement step rather than a single point of failure.
These tests assert on that baseline.
"""

from __future__ import annotations

from tradingagents_in.agents.analysts import ANALYSTS
from tradingagents_in.agents.researchers import BearResearcher, BullResearcher, ResearchManager
from tradingagents_in.agents.risk_team import PortfolioManager, RiskManager
from tradingagents_in.agents.trader import Trader
from tradingagents_in.models.enums import (
    Action,
    Instrument,
    RiskVerdict,
    Stance,
    SurveillanceFlag,
)
from tradingagents_in.models.research import DebateState
from tradingagents_in.models.risk import PortfolioSnapshot


def run_analysts(bundle, config) -> dict:
    state: dict = {}
    for name, cls in ANALYSTS.items():
        report = cls(config).run(bundle, state)
        state[f"{name}_report"] = report
    return state


def run_debate(bundle, config, state: dict, rounds: int = 1) -> dict:
    """One or more real bull/bear rounds, then the manager's adjudication.

    The research manager scores *arguments*, so a state with an empty
    transcript legitimately produces NEUTRAL. Tests that need a directional
    view have to hold the debate.
    """
    debate = DebateState(max_rounds=rounds)
    for _ in range(rounds):
        for side, agent in (("bull", BullResearcher), ("bear", BearResearcher)):
            state["debate"] = debate
            turn = agent(config).run(bundle, state)
            debate = DebateState(
                rounds_completed=debate.rounds_completed + (1 if side == "bear" else 0),
                max_rounds=rounds,
                turns=[*debate.turns, turn],
            )
    state["debate"] = debate
    state["investment_plan"] = ResearchManager(config).run(bundle, state)
    return state


def bullish_state(bundle, config) -> dict:
    """Analyst reports forced bullish, then a genuine debate over them."""
    state = run_analysts(bundle, config)
    for report in state.values():
        report.stance = Stance.BULLISH
    return run_debate(bundle, config, state)


class TestAnalysts:
    def test_all_four_produce_a_usable_report(self, bundle, config):
        state = run_analysts(bundle, config)
        assert len(state) == 4
        for report in state.values():
            assert report.summary
            assert isinstance(report.stance, Stance)
            assert 0.0 <= report.confidence <= 1.0
            assert report.to_markdown()

    def test_confidence_is_haircut_by_missing_data(self, bundle, config):
        report = ANALYSTS["technical"](config).run(bundle, {})
        full = report.effective_confidence()

        report.data_quality.completeness = 0.2
        assert report.effective_confidence() < full

    def test_technical_report_carries_levels_and_timeframes(self, bundle, config):
        report = ANALYSTS["technical"](config).run(bundle, {})
        assert report.timeframes
        assert report.last_price
        # A technical read without an invalidation level is not actionable.
        assert report.suggested_stop_loss is not None

    def test_index_skips_company_fundamentals_gracefully(self, index_bundle, config):
        report = ANALYSTS["fundamentals"](config).run(index_bundle, {})
        assert report.summary
        assert report.data_quality.completeness <= 1.0


class TestDebate:
    def test_both_sides_argue_from_the_same_briefing(self, bundle, config):
        state = run_analysts(bundle, config)
        bull = BullResearcher(config).run(bundle, state)
        bear = BearResearcher(config).run(bundle, state)

        assert bull.side == "bull" and bear.side == "bear"
        assert bull.thesis and bear.thesis
        assert bull.points or bull.conceded
        assert bear.points or bear.conceded

    def test_india_specific_arguments_are_produced(self, bundle, config):
        state = run_analysts(bundle, config)
        bundle.micro.in_fno_ban = True
        bundle.micro.surveillance = SurveillanceFlag.ASM_SHORT_TERM
        bear = BearResearcher(config).run(bundle, state)
        assert any(p.india_specific for p in bear.points)
        assert bear.india_risk_focus

    def test_manager_stays_neutral_when_every_analyst_is_neutral(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.NEUTRAL

        run_debate(bundle, config, state)
        plan = state["investment_plan"]
        # Structural tradability points must not manufacture a directional tilt.
        assert plan.stance == Stance.NEUTRAL
        assert plan.winning_side == "neither"

    def test_tradability_blocker_overrides_direction(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.BULLISH
        bundle.micro.is_circuit_locked = True

        run_debate(bundle, config, state)
        plan = state["investment_plan"]
        assert plan.stance == Stance.NEUTRAL
        assert "circuit" in plan.verdict.lower()


class TestTrader:
    def test_bullish_view_produces_a_complete_order(self, bundle, config):
        state = bullish_state(bundle, config)
        proposal = Trader(config).run(bundle, state)

        assert proposal.action == Action.BUY
        assert proposal.entry_zone and proposal.entry_zone.trigger
        assert proposal.stop_loss and proposal.stop_loss < proposal.entry_zone.mid
        assert proposal.targets and proposal.targets[0].price > proposal.entry_zone.mid
        assert proposal.quantity and proposal.quantity > 0
        assert proposal.invalidation
        assert proposal.execution_notes

    def test_size_matches_the_risk_budget(self, bundle, config):
        state = bullish_state(bundle, config)
        proposal = Trader(config).run(bundle, state)
        budget = config["capital_inr"] * config["risk_per_trade_pct"] / 100
        assert proposal.risk_amount_inr <= budget * 1.01

    def test_neutral_view_produces_hold_with_a_reason(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.NEUTRAL
        run_debate(bundle, config, state)

        proposal = Trader(config).run(bundle, state)
        assert proposal.action == Action.HOLD
        assert not proposal.is_actionable
        assert proposal.summary

    def test_untradable_scrip_produces_hold(self, bundle, config):
        state = bullish_state(bundle, config)
        bundle.micro.is_circuit_locked = True
        proposal = Trader(config).run(bundle, state)
        assert proposal.action == Action.HOLD
        assert "circuit" in proposal.summary.lower()

    def test_index_uses_a_defined_risk_option_structure(self, index_bundle, config_factory):
        config = config_factory(capital_inr=10_000_000)
        state = bullish_state(index_bundle, config)
        proposal = Trader(config).run(index_bundle, state)

        if proposal.action != Action.HOLD:
            assert proposal.instrument == Instrument.OPTIONS
            assert len(proposal.option_legs) == 2
            # A debit spread: one bought leg, one sold leg, same expiry.
            sides = {leg.side.value for leg in proposal.option_legs}
            assert sides == {"BUY", "SELL"}
            assert len({leg.expiry for leg in proposal.option_legs}) == 1

    def test_cash_short_is_forced_intraday(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.BEARISH
        run_debate(bundle, config, state)

        proposal = Trader(config).run(bundle, state)
        if proposal.action == Action.SHORT and proposal.instrument == Instrument.CASH:
            assert proposal.product_type.value == "MIS"
            assert proposal.time_horizon.value == "INTRADAY"


class TestRiskAndPortfolio:
    def _proposed(self, bundle, config) -> dict:
        state = bullish_state(bundle, config)
        state["portfolio"] = PortfolioSnapshot(
            capital_inr=config["capital_inr"], cash_inr=config["capital_inr"]
        )
        state["trade_proposal"] = Trader(config).run(bundle, state)
        return state

    def test_clean_proposal_is_approved(self, bundle, config):
        state = self._proposed(bundle, config)
        assessment = RiskManager(config).run(bundle, state)
        assert assessment.verdict in (
            RiskVerdict.APPROVE, RiskVerdict.APPROVE_WITH_CHANGES
        )
        assert assessment.sizing_rationale
        assert assessment.stress_test
        assert assessment.sebi_compliance_notes

    def test_blocker_forces_reject_and_zero_size(self, bundle, config):
        state = self._proposed(bundle, config)
        bundle.micro.is_circuit_locked = True

        assessment = RiskManager(config).run(bundle, state)
        assert assessment.verdict == RiskVerdict.REJECT
        assert assessment.approved_position_size_pct == 0.0
        assert assessment.blocking

    def test_portfolio_manager_cannot_approve_a_rejected_trade(self, bundle, config):
        state = self._proposed(bundle, config)
        bundle.micro.is_circuit_locked = True
        state["risk_assessment"] = RiskManager(config).run(bundle, state)

        decision = PortfolioManager(config).run(bundle, state)
        assert not decision.approved
        assert decision.final_action == Action.HOLD
        assert decision.rejection_reasons

    def test_portfolio_manager_declines_on_book_level_grounds(self, bundle, config):
        state = self._proposed(bundle, config)
        state["risk_assessment"] = RiskManager(config).run(bundle, state)
        # The risk manager cleared the trade; the book has no room for it.
        state["portfolio"] = PortfolioSnapshot(
            capital_inr=config["capital_inr"],
            cash_inr=config["capital_inr"],
            portfolio_heat_pct=5.9,
            open_positions=3,
        )
        decision = PortfolioManager(config).run(bundle, state)
        assert not decision.approved
        assert any("risk" in r.lower() or "heat" in r.lower() for r in decision.rejection_reasons)

    def test_drawdown_halt_blocks_new_risk(self, bundle, config):
        state = self._proposed(bundle, config)
        state["portfolio"] = PortfolioSnapshot(
            capital_inr=config["capital_inr"],
            cash_inr=config["capital_inr"],
            current_drawdown_pct=18.0,
        )
        assessment = RiskManager(config).run(bundle, state)
        assert assessment.verdict == RiskVerdict.REJECT


class TestDebateQuality:
    """The deterministic debate must not look rigged.

    Two failure modes that would quietly discredit the whole transcript: a side
    quoting evidence that argues against itself, and a side with nothing to say.
    """

    def test_bull_does_not_cite_bearish_evidence_as_support(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.BULLISH
        # A bullish technical read can still contain a bearish lower timeframe.
        state["technical_report"].key_points = [
            "1d: uptrend, signal bullish, RSI 59",
            "1h: downtrend, signal bearish, RSI 36",
        ]

        bull = BullResearcher(config).run(bundle, state)
        claims = " ".join(p.claim for p in bull.points).lower()
        assert "uptrend" in claims
        assert "downtrend" not in claims

    def test_bear_builds_a_case_from_risks_when_no_analyst_is_bearish(self, bundle, config):
        state = run_analysts(bundle, config)
        for report in state.values():
            report.stance = Stance.BULLISH
            report.risks = ["Valuation leaves no room for a single quarterly miss"]

        bear = BearResearcher(config).run(bundle, state)
        assert bear.points, "the bear must have a case even against a unanimous bull tape"
        assert any("miss" in p.claim for p in bear.points)
        # And it must still concede the strength it is arguing against.
        assert bear.conceded

    def test_manager_discounts_a_side_that_never_concedes(self, bundle, config):
        from tradingagents_in.agents.researchers.manager import _side_score

        state = run_analysts(bundle, config)
        run_debate(bundle, config, state, rounds=2)
        debate = state["debate"]

        stubborn = DebateState(
            rounds_completed=debate.rounds_completed,
            turns=[t.model_copy(update={"conceded": []}) for t in debate.turns],
        )
        for side in ("bull", "bear"):
            if any(t.conceded for t in debate.turns if t.side == side):
                assert _side_score(stubborn, side) < _side_score(debate, side)
