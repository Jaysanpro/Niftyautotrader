"""The deterministic risk engine: costs, sizing and the rulebook.

These are the tests that matter most. Everything else in the framework can be
wrong and produce a bad *suggestion*; if this layer is wrong it produces a
position the desk cannot exit.
"""

from __future__ import annotations

import pytest

from tradingagents_in.models.enums import (
    Action,
    Instrument,
    LegSide,
    OptionType,
    ProductType,
    Severity,
    SurveillanceFlag,
    TimeHorizon,
)
from tradingagents_in.models.risk import PortfolioSnapshot
from tradingagents_in.models.trade import EntryZone, OptionLeg, Target, TradeProposal
from tradingagents_in.risk import costs, rules, sizing


def make_proposal(**overrides) -> TradeProposal:
    defaults = dict(
        role="trader",
        symbol="RELIANCE.NS",
        action=Action.BUY,
        instrument=Instrument.CASH,
        product_type=ProductType.CNC,
        time_horizon=TimeHorizon.SWING,
        entry_zone=EntryZone(low=2_480, high=2_520, reference_price=2_500),
        stop_loss=2_400.0,
        targets=[Target(price=2_700, book_pct=100)],
        quantity=100,
        position_size_pct=2.5,
        capital_deployed_inr=250_000.0,
        risk_amount_inr=10_000.0,
        summary="test proposal",
    )
    defaults.update(overrides)
    return TradeProposal(**defaults)


# --------------------------------------------------------------------- costs
class TestCosts:
    def test_delivery_charges_are_dominated_by_stt(self):
        est = costs.estimate_costs(
            entry_price=2_500, exit_price=2_700, quantity=100,
            instrument=Instrument.CASH, product_type=ProductType.CNC,
        )
        # 0.1% STT on both legs of a ~5.2 lakh round trip is ~520 rupees.
        assert est.stt_inr == pytest.approx(520.0, rel=0.01)
        assert est.stt_inr > est.brokerage_inr + est.exchange_txn_inr
        assert est.total_inr > est.stt_inr

    def test_intraday_stt_is_sell_side_only_and_cheaper(self):
        delivery = costs.estimate_costs(
            entry_price=2_500, exit_price=2_500, quantity=100,
            product_type=ProductType.CNC,
        )
        intraday = costs.estimate_costs(
            entry_price=2_500, exit_price=2_500, quantity=100,
            product_type=ProductType.MIS,
        )
        assert intraday.stt_inr < delivery.stt_inr

    def test_gst_excludes_stt_and_stamp_duty(self):
        est = costs.estimate_costs(entry_price=1_000, exit_price=1_000, quantity=1_000)
        expected = (est.brokerage_inr + est.exchange_txn_inr + est.sebi_charges_inr) * 0.18
        assert est.gst_inr == pytest.approx(expected, abs=0.01)  # fields are rounded to paise

    def test_options_charge_on_premium_not_notional(self):
        est = costs.estimate_costs(
            entry_price=120, exit_price=150, quantity=500, instrument=Instrument.OPTIONS,
        )
        # Sell-side STT is 0.1% of the 75,000 premium turnover, i.e. 75 rupees.
        assert est.stt_inr == pytest.approx(75.0, rel=0.01)

    def test_costs_can_flip_a_marginal_reward_risk(self):
        # Gross R:R is exactly 2.0; a 0.4% stop on delivery does not survive it.
        rr, est = costs.net_reward_risk(
            entry_price=1_000, stop_loss=996, target=1_008, quantity=100,
            product_type=ProductType.CNC,
        )
        assert rr is not None and rr < 2.0
        assert est.total_inr > 0

    def test_breakeven_move_is_reported(self):
        est = costs.estimate_costs(entry_price=500, exit_price=500, quantity=200)
        assert costs.breakeven_move_pct(est, 500, 200) > 0


# -------------------------------------------------------------------- sizing
class TestSizing:
    def test_size_is_the_risk_budget_divided_by_the_stop(self, config_factory):
        # The single-name cap has to be lifted for the risk budget to be the
        # binding constraint - which is itself the point of the next test.
        config = config_factory(max_position_pct=60.0)
        result = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=1_000, stop_loss=980, config=config
        )
        # 1% of 10 lakh is 10,000; a 20-rupee stop buys 500 shares.
        assert result.quantity == 500
        assert result.risk_amount_inr == pytest.approx(10_000)
        assert result.binding_constraint == "risk budget"

    def test_single_name_cap_binds_before_the_risk_budget_by_default(self, config):
        # At the default 10% cap, a 2% stop cannot deploy the full risk budget:
        # 500 shares at Rs 1,000 would be half the book in one name.
        result = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=1_000, stop_loss=980, config=config
        )
        assert result.quantity == 100
        assert result.binding_constraint == "max position size"
        assert result.risk_amount_inr < sizing.risk_budget_inr(1_000_000, config)

    def test_a_wider_stop_gives_a_smaller_position(self, config_factory):
        config = config_factory(max_position_pct=100.0)
        tight = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=1_000, stop_loss=990, config=config
        )
        wide = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=1_000, stop_loss=950, config=config
        )
        assert wide.quantity < tight.quantity
        # Risk stays constant; only the size moves. That is the whole rule.
        assert wide.risk_amount_inr == pytest.approx(tight.risk_amount_inr, rel=0.02)

    def test_max_position_cap_binds_before_the_risk_budget(self, config_factory):
        config = config_factory(max_position_pct=5.0, risk_per_trade_pct=2.0)
        result = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=100, stop_loss=99.5, config=config
        )
        assert result.binding_constraint == "max position size"
        assert result.position_pct <= 5.0 + 1e-6
        assert any("capped by" in n for n in result.notes)

    def test_no_stop_distance_is_not_tradable(self, config):
        result = sizing.size_cash_position(
            capital_inr=1_000_000, entry_price=1_000, stop_loss=1_000, config=config
        )
        assert not result.is_tradable

    def test_one_unaffordable_lot_means_zero_lots(self, config):
        # A wide stop on a big-lot contract cannot be sized down below one lot.
        result = sizing.size_futures_position(
            capital_inr=200_000, entry_price=24_000, stop_loss=23_000,
            base_symbol="NIFTY", config=config, lot_size=75, is_index=True,
        )
        assert result.lots == 0
        assert not result.is_tradable
        assert "indivisible" in " ".join(result.notes)

    def test_futures_exposure_and_deployed_capital_differ(self, config):
        # One NIFTY lot is ~18 lakh of notional, so the book has to be large
        # enough for that to sit inside the 10% single-name cap.
        result = sizing.size_futures_position(
            capital_inr=100_000_000, entry_price=24_000, stop_loss=23_900,
            base_symbol="NIFTY", config=config, lot_size=75, is_index=True,
        )
        assert result.lots >= 1
        # Margin posted is a fraction of notional; conflating them is a real bug.
        assert result.capital_deployed_inr < result.exposure_inr

    def test_long_option_risk_is_the_premium(self, config):
        result = sizing.size_option_position(
            capital_inr=1_000_000, premium=100, base_symbol="RELIANCE",
            config=config, lot_size=500,
        )
        if result.is_tradable:
            assert result.risk_amount_inr == pytest.approx(result.capital_deployed_inr)


# ------------------------------------------------------------------ rulebook
class TestRulebook:
    def test_clean_proposal_has_no_blockers(self, bundle, flat_book, config):
        breaches = rules.check_all(make_proposal(), bundle, flat_book, config)
        assert not [b for b in breaches if b.severity == Severity.BLOCKER]

    def test_missing_stop_blocks(self, bundle, flat_book, config):
        breaches = rules.check_all(make_proposal(stop_loss=None), bundle, flat_book, config)
        assert any(b.rule == "missing_stop_loss" and b.severity == Severity.BLOCKER
                   for b in breaches)

    def test_stop_on_the_wrong_side_blocks(self, bundle, flat_book, config):
        breaches = rules.check_all(make_proposal(stop_loss=2_600), bundle, flat_book, config)
        assert any(b.rule == "stop_on_wrong_side" for b in breaches)

    def test_circuit_locked_blocks(self, bundle, flat_book, config):
        bundle.micro.is_circuit_locked = True
        breaches = rules.check_tradability(make_proposal(), bundle, config)
        assert any(b.rule == "circuit_locked" and b.severity == Severity.BLOCKER
                   for b in breaches)

    def test_gsm_blocks_but_asm_only_warns(self, bundle, config):
        bundle.micro.surveillance = SurveillanceFlag.GSM
        gsm = rules.check_tradability(make_proposal(), bundle, config)
        assert any(b.rule == "surveillance_framework" and b.severity == Severity.BLOCKER
                   for b in gsm)

        bundle.micro.surveillance = SurveillanceFlag.ASM_SHORT_TERM
        asm = rules.check_tradability(make_proposal(), bundle, config)
        breach = next(b for b in asm if b.rule == "surveillance_framework")
        assert breach.severity == Severity.WARNING

    def test_fno_ban_blocks_a_derivative_but_not_cash(self, bundle, config):
        bundle.micro.in_fno_ban = True
        cash = rules.check_tradability(make_proposal(), bundle, config)
        assert not any(b.rule == "fno_ban_period" for b in cash)

        future = make_proposal(instrument=Instrument.FUTURES, product_type=ProductType.NRML)
        derivative = rules.check_tradability(future, bundle, config)
        assert any(b.rule == "fno_ban_period" and b.severity == Severity.BLOCKER
                   for b in derivative)

    def test_overnight_cash_short_blocks(self, bundle, config):
        proposal = make_proposal(
            action=Action.SHORT, instrument=Instrument.CASH,
            stop_loss=2_600, targets=[Target(price=2_300)],
        )
        # The model validator already forces MIS/intraday; a mandate that bars
        # cash shorts entirely must still block it.
        breaches = rules.check_tradability(proposal, bundle, config)
        assert any(b.rule == "cash_short_not_permitted" for b in breaches)

    def test_stop_beyond_the_circuit_band_blocks(self, bundle, config):
        bundle.micro.circuit_band_pct = 5.0
        proposal = make_proposal(stop_loss=(bundle.last_price or 2_500) * 0.85)
        breaches = rules.check_tradability(proposal, bundle, config)
        assert any(b.rule == "stop_outside_circuit_band" and b.severity == Severity.BLOCKER
                   for b in breaches)

    def test_thin_liquidity_blocks(self, bundle, config):
        bundle.micro.avg_daily_turnover_inr = 1_00_00_000  # Rs 1 crore
        breaches = rules.check_tradability(make_proposal(), bundle, config)
        assert any(b.rule == "insufficient_liquidity" for b in breaches)

    def test_portfolio_heat_blocks(self, bundle, config):
        book = PortfolioSnapshot(
            capital_inr=1_000_000, cash_inr=500_000, portfolio_heat_pct=5.8, open_positions=4
        )
        breaches = rules.check_portfolio_risk(make_proposal(), bundle, book, config)
        assert any(b.rule == "portfolio_heat" and b.severity == Severity.BLOCKER
                   for b in breaches)

    def test_sector_concentration_warns(self, bundle, config):
        book = PortfolioSnapshot(
            capital_inr=1_000_000,
            sector_exposure_pct={bundle.ref.sector or "Unclassified": 24.0},
        )
        breaches = rules.check_portfolio_risk(make_proposal(), bundle, book, config)
        assert any(b.rule == "sector_concentration" for b in breaches)

    def test_max_drawdown_blocks_even_a_hold(self, config):
        book = PortfolioSnapshot(capital_inr=1_000_000, current_drawdown_pct=16.0)
        breaches = rules.check_portfolio_state(book, config)
        assert any(b.rule == "max_drawdown_breached" and b.severity == Severity.BLOCKER
                   for b in breaches)

    def test_drawdown_warning_precedes_the_block(self, config):
        book = PortfolioSnapshot(capital_inr=1_000_000, current_drawdown_pct=11.0)
        breaches = rules.check_portfolio_state(book, config)
        assert [b.rule for b in breaches] == ["drawdown_warning"]

    def test_stress_test_covers_gaps_and_the_circuit(self, bundle, config):
        result = rules.stress_test(make_proposal(), bundle, config)
        assert set(result) >= {"stop_hit", "gap_down_5pct", "lower_circuit_hit"}
        assert all(v <= 0 for v in result.values())

    def test_option_loss_is_floored_at_the_premium(self, bundle, config):
        proposal = make_proposal(
            instrument=Instrument.OPTIONS, product_type=ProductType.NRML,
            capital_deployed_inr=20_000.0, quantity=500,
            option_legs=[
                OptionLeg(
                    side=LegSide.BUY, option_type=OptionType.CE, strike=2_600,
                    expiry="2026-08-27", lots=1, lot_size=500, estimated_premium=40.0,
                )
            ],
            option_strategy="long CE",
        )
        result = rules.stress_test(proposal, bundle, config)
        assert min(result.values()) >= -20_000.0

    def test_sebi_notes_mention_settlement_and_peak_margin(self, bundle, config):
        notes = " ".join(rules.sebi_compliance_notes(make_proposal(), bundle, config))
        assert "T+1" in notes and "eak-margin" in notes
