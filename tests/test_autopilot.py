"""The autonomous pass.

The thing being checked is that a pass needs no symbol from anyone, and that
it does the safe things in the safe order: mark before entering, so an exit
beats an entry; never stack on a name the book already has; and bound how
much new risk one pass may take regardless of how good the screen looks.
"""

from __future__ import annotations

import pytest

from tradingagents_in.autopilot import render, run_pass
from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.models.enums import Action, ProductType
from tradingagents_in.paper import Order, OrderType, PaperBroker

CASH = 1_000_000.0


@pytest.fixture(scope="module")
def config():
    return build_config({
        "data_provider": "synthetic", "llm_provider": "offline",
        "online_tools": False, "synthetic_seed": 7,
    })


@pytest.fixture(scope="module")
def app(config):
    return IndianTradingAgents(config)


@pytest.fixture
def book(config):
    return PaperBroker(starting_cash_inr=CASH, config=config)


# ------------------------------------------------------------ the basic pass


def test_a_pass_picks_names_without_being_given_one(app, book):
    result = run_pass(app, book, "NIFTYIT", max_new_positions=3)
    assert result.scanned > 0
    assert result.selected, "the screen found nothing to do"
    assert len(result.selected) <= 3
    # Every entry should have a stop beside it.
    assert len(result.placed) == 2 * len(result.selected)


def test_a_pass_places_a_stop_with_every_entry(app, book):
    run_pass(app, book, "NIFTYIT", max_new_positions=2)
    stops = [o for o in book.orders() if o.order_type == OrderType.SL]
    assert stops, "no stop was placed"
    for stop in stops:
        assert stop.trigger_price and stop.trigger_price > 0


def test_a_fill_does_not_conjure_equity(app, book):
    before = book.equity_inr()
    run_pass(app, book, "NIFTYIT", max_new_positions=3)
    assert book.equity_inr() == pytest.approx(before, abs=1.0), (
        "equity moved on entry; the book gained money it never made"
    )


def test_the_pass_limit_bounds_new_risk(app, book):
    result = run_pass(app, book, "NIFTY50", max_new_positions=1)
    assert len(result.selected) == 1
    assert any("pass limit" in why for why in result.skipped.values())


def test_zero_new_positions_still_manages_what_is_open(app, book):
    run_pass(app, book, "NIFTYIT", max_new_positions=2)
    held = {p.symbol for p in book.positions()}
    assert held

    result = run_pass(app, book, "NIFTYIT", max_new_positions=0)
    assert result.selected == []
    assert {p.symbol for p in book.positions()} == held


# ------------------------------------------------------------ not stacking


def test_a_held_name_is_not_entered_again(app, book):
    run_pass(app, book, "NIFTYIT", max_new_positions=3)
    held = {p.symbol for p in book.positions()}
    assert held

    second = run_pass(app, book, "NIFTYIT", max_new_positions=3)
    for symbol in held:
        assert symbol not in second.selected
        assert second.skipped.get(symbol) == "already held"


def test_a_name_with_a_resting_order_is_not_entered_again(app, config):
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    # A resting entry nobody has filled yet.
    b.place(Order(symbol="TCS", side=Action.BUY, quantity=1,
                  order_type=OrderType.LIMIT, limit_price=1.0))
    result = run_pass(app, b, "NIFTYIT", max_new_positions=5)
    assert "TCS" not in result.selected


# -------------------------------------------------------- order of operations


def test_marking_happens_before_entering(app, config):
    """An exit must beat an entry, or a pass can enter on money it is losing."""
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    run_pass(app, b, "NIFTYIT", max_new_positions=2)

    stops = {o.symbol: o.trigger_price for o in b.orders(open_only=True)
             if o.order_type == OrderType.SL}
    assert stops

    # Drive everything through its stop on the next pass.
    crash = {s: round(t * 0.97, 2) for s, t in stops.items()}
    result = run_pass(b and app, b, "NIFTYIT", prices=crash, max_new_positions=0)

    assert result.filled, "the stops did not fire"
    assert all(p.is_flat for p in b.positions(include_flat=True))
    assert b.trades, "a stopped-out trade should be booked"


def test_stops_close_positions_and_book_a_loss(app, config):
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    run_pass(app, b, "NIFTYIT", max_new_positions=2)
    stops = {o.symbol: o.trigger_price for o in b.orders(open_only=True)
             if o.order_type == OrderType.SL}
    run_pass(app, b, "NIFTYIT", prices={s: t * 0.97 for s, t in stops.items()},
             max_new_positions=0)

    assert b.trades
    for trade in b.trades:
        assert trade.net_pnl_inr < 0, "a stop-out should not be a winner"
        assert trade.charges_inr > 0
    assert b.equity_inr() < CASH


def test_a_losing_pass_stays_inside_the_risk_budget(app, config):
    """Each stop-out should cost about the per-trade budget, not more."""
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    run_pass(app, b, "NIFTYIT", max_new_positions=3)
    stops = {o.symbol: o.trigger_price for o in b.orders(open_only=True)
             if o.order_type == OrderType.SL}
    run_pass(app, b, "NIFTYIT", prices={s: t * 0.99 for s, t in stops.items()},
             max_new_positions=0)

    budget = CASH * float(b.config["risk_per_trade_pct"]) / 100.0
    for trade in b.trades:
        # Slippage and charges can push slightly past the budget; a multiple
        # of it would mean the sizing or the stop was not respected.
        assert abs(trade.net_pnl_inr) < budget * 1.5, (
            f"{trade.symbol} lost {abs(trade.net_pnl_inr):,.0f} against a "
            f"{budget:,.0f} budget"
        )


# ---------------------------------------------------------------- square off


def test_square_off_closes_intraday_positions(app, config):
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    run_pass(app, b, "NIFTYIT", max_new_positions=3)
    intraday = [p for p in b.positions() if p.product == ProductType.MIS]
    if not intraday:
        pytest.skip("this tape produced no intraday positions to square off")

    marks = {p.symbol: p.average_price for p in b.positions()}
    result = run_pass(app, b, "NIFTYIT", prices=marks, square_off=True,
                      max_new_positions=0)
    assert result.squared_off
    assert not [p for p in b.positions() if p.product == ProductType.MIS]


# ----------------------------------------------------------------- failures


def test_a_broken_screen_is_reported_not_raised(app, book):
    result = run_pass(app, book, "Enrgy")  # a misspelled sector
    assert result.errors
    assert result.selected == []
    assert result.equity_inr == pytest.approx(CASH)


def test_one_bad_symbol_does_not_stop_the_pass(app, config, monkeypatch):
    b = PaperBroker(starting_cash_inr=CASH, config=config)
    real = app.analyse
    calls = {"n": 0}

    def flaky(symbol, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("feed exploded")
        return real(symbol, **kwargs)

    monkeypatch.setattr(app, "analyse", flaky)
    result = run_pass(app, b, "NIFTYIT", max_new_positions=3)
    assert any("feed exploded" in e for e in result.errors)


def test_marking_an_unknown_symbol_is_harmless(app, book):
    result = run_pass(app, book, "NIFTYIT", prices={"NOTREAL.NS": 100.0},
                      max_new_positions=0)
    assert result.equity_inr == pytest.approx(CASH)


# ------------------------------------------------------------------- output


def test_render_describes_the_pass_and_the_book(app, book):
    result = run_pass(app, book, "NIFTYIT", max_new_positions=2)
    text = render(result, book)
    assert "scanned" in text and "equity" in text
    for symbol in result.selected:
        assert symbol[:13] in text


def test_summary_is_serialisable(app, book):
    import json

    result = run_pass(app, book, "NIFTYIT", max_new_positions=1)
    assert json.loads(json.dumps(result.summary(), default=str))


def test_the_book_survives_between_passes(app, config, tmp_path):
    """The loop is stateless; the book on disk is the state."""
    path = tmp_path / "book.json"
    first = PaperBroker(starting_cash_inr=CASH, config=config, state_path=path)
    run_pass(app, first, "NIFTYIT", max_new_positions=2)
    first.save()
    held = {p.symbol for p in first.positions()}
    resting = len(first.orders(open_only=True))

    resumed = PaperBroker.load(path, config=config)
    assert {p.symbol for p in resumed.positions()} == held
    assert len(resumed.orders(open_only=True)) == resting

    # And the resumed book does not re-enter what it already holds.
    result = run_pass(app, resumed, "NIFTYIT", max_new_positions=3)
    for symbol in held:
        assert symbol not in result.selected
