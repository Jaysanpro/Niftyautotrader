"""The simulated broker.

Accounting bugs here are silent and flattering - a book that quietly gains
money it never made reads like a strategy that works. So the invariant these
tests keep returning to is conservation: equity must not change merely because
an order filled, only because a price moved or a charge was paid.
"""

from __future__ import annotations

import json

import pytest

from tradingagents_in.models.enums import Action, AgentRole, Instrument, ProductType, TimeHorizon
from tradingagents_in.models.trade import EntryZone, Target, TradeProposal
from tradingagents_in.paper import (
    Order,
    OrderStatus,
    OrderType,
    PaperBroker,
    orders_for_decision,
)

CASH = 1_000_000.0


def broker(**kwargs) -> PaperBroker:
    kwargs.setdefault("starting_cash_inr", CASH)
    kwargs.setdefault("slippage_bps", 0.0)  # off unless a test is about it
    return PaperBroker(**kwargs)


def market(symbol="X.NS", side=Action.BUY, qty=100, product=ProductType.CNC) -> Order:
    return Order(symbol=symbol, side=side, quantity=qty, product=product,
                 order_type=OrderType.MARKET)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("order,expected", [
    (Order(symbol="X.NS", side=Action.BUY, quantity=0), "quantity"),
    (Order(symbol="X.NS", side=Action.BUY, quantity=-5), "quantity"),
    (Order(symbol="X.NS", side=Action.BUY, quantity=1, order_type=OrderType.LIMIT), "limit"),
    (Order(symbol="X.NS", side=Action.BUY, quantity=1, order_type=OrderType.SL), "trigger"),
])
def test_malformed_orders_are_rejected_with_a_reason(order, expected):
    placed = broker().place(order)
    assert placed.status == OrderStatus.REJECTED
    assert expected in placed.message.lower()


def test_delivery_beyond_cash_is_rejected():
    b = broker(starting_cash_inr=100_000.0)
    b.mark("X.NS", 1500.0)
    placed = b.place(market(qty=100))  # 150,000 needed
    assert placed.status == OrderStatus.REJECTED
    assert "insufficient funds" in placed.message


def test_intraday_leverage_allows_what_delivery_cannot():
    b = broker(starting_cash_inr=100_000.0, mis_leverage=5.0)
    b.mark("X.NS", 1500.0)
    # 150,000 notional needs 30,000 margin at 5x - affordable, unlike delivery.
    placed = b.place(market(qty=100, product=ProductType.MIS))
    assert placed.status == OrderStatus.FILLED
    assert b.funds().blocked_margin_inr == pytest.approx(30_000.0)


def test_closing_a_position_is_never_refused_for_funds():
    """Otherwise a fully invested book could not get out."""
    b = broker(starting_cash_inr=150_000.0)
    b.mark("X.NS", 1500.0)
    b.place(market(qty=100))
    assert b.funds().cash_inr == pytest.approx(0.0)

    exit_order = b.place(market(side=Action.SELL, qty=100))
    assert exit_order.status == OrderStatus.FILLED


# ------------------------------------------------------------------ fills


def test_a_buy_limit_only_fills_at_or_below_its_price():
    b = broker()
    order = b.place(Order(symbol="X.NS", side=Action.BUY, quantity=10,
                          order_type=OrderType.LIMIT, limit_price=1500.0))
    b.mark("X.NS", 1600.0)
    assert order.status == OrderStatus.PENDING, "filled above the limit"
    b.mark("X.NS", 1500.0)
    assert order.status == OrderStatus.FILLED
    assert order.fill_price == pytest.approx(1500.0), "a limit fills at its own price"


def test_a_sell_limit_only_fills_at_or_above_its_price():
    b = broker()
    b.mark("X.NS", 1500.0)
    b.place(market(qty=10))
    order = b.place(Order(symbol="X.NS", side=Action.SELL, quantity=10,
                          order_type=OrderType.LIMIT, limit_price=1600.0))
    b.mark("X.NS", 1550.0)
    assert order.status == OrderStatus.PENDING
    b.mark("X.NS", 1610.0)
    assert order.status == OrderStatus.FILLED
    assert order.fill_price == pytest.approx(1600.0)


def test_a_stop_triggers_only_once_price_crosses_it():
    b = broker()
    b.mark("X.NS", 1500.0)
    b.place(market(qty=10))
    stop = b.place(Order(symbol="X.NS", side=Action.SELL, quantity=10,
                         order_type=OrderType.SL, trigger_price=1450.0))
    b.mark("X.NS", 1460.0)
    assert stop.status == OrderStatus.PENDING
    b.mark("X.NS", 1445.0)
    assert stop.status == OrderStatus.FILLED


def test_slippage_always_runs_against_the_order():
    b = PaperBroker(starting_cash_inr=CASH, slippage_bps=10.0)
    b.mark("X.NS", 1000.0)
    buy = b.place(market(qty=10))
    assert buy.fill_price > 1000.0, "a buy should pay up"

    sell = b.place(market(side=Action.SELL, qty=10))
    assert sell.fill_price < 1000.0, "a sell should receive less"


def test_a_limit_order_does_not_slip():
    b = PaperBroker(starting_cash_inr=CASH, slippage_bps=50.0)
    order = b.place(Order(symbol="X.NS", side=Action.BUY, quantity=10,
                          order_type=OrderType.LIMIT, limit_price=1000.0))
    b.mark("X.NS", 990.0)
    assert order.fill_price == pytest.approx(1000.0)


def test_a_market_order_rests_until_a_price_is_known():
    b = broker()
    order = b.place(market(qty=10))
    assert order.status == OrderStatus.PENDING
    b.mark("X.NS", 1200.0)
    assert order.status == OrderStatus.FILLED


def test_cancelling_a_resting_order_stops_it_filling():
    b = broker()
    order = b.place(Order(symbol="X.NS", side=Action.BUY, quantity=10,
                          order_type=OrderType.LIMIT, limit_price=900.0))
    assert b.cancel(order.order_id) is True
    assert b.cancel(order.order_id) is False, "cancelling twice should not succeed"
    b.mark("X.NS", 800.0)
    assert order.status == OrderStatus.CANCELLED
    assert b.positions() == []


# ----------------------------------------------------------- conservation


def test_a_delivery_fill_does_not_change_equity():
    """The regression that matters: cash becomes stock, and nothing is created.

    Counting only unrealised P&L here understated a delivery book by its
    entire cost basis.
    """
    b = broker()
    b.mark("X.NS", 1500.0)
    before = b.equity_inr()
    b.place(market(qty=100))
    assert b.equity_inr() == pytest.approx(before), "equity moved on a fill"
    assert b.funds().cash_inr == pytest.approx(CASH - 150_000.0)


def test_an_intraday_fill_does_not_change_equity_either():
    b = broker()
    b.mark("Y.NS", 1000.0)
    before = b.equity_inr()
    b.place(market(symbol="Y.NS", qty=100, product=ProductType.MIS))
    assert b.equity_inr() == pytest.approx(before)
    assert b.funds().cash_inr == pytest.approx(CASH), "intraday blocks margin, not cash"


@pytest.mark.parametrize("product", [ProductType.CNC, ProductType.MIS])
def test_equity_tracks_the_mark(product):
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=100, product=product))
    b.mark("X.NS", 1100.0)
    assert b.equity_inr() == pytest.approx(CASH + 10_000.0)
    b.mark("X.NS", 900.0)
    assert b.equity_inr() == pytest.approx(CASH - 10_000.0)


# ----------------------------------------------------------- the position


def test_adding_to_a_position_averages_the_basis():
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=100))
    b.mark("X.NS", 1200.0)
    b.place(market(qty=100))
    position = b.position("X.NS")
    assert position.quantity == 200
    assert position.average_price == pytest.approx(1100.0)


def test_a_round_trip_books_pnl_net_of_charges():
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=100))
    b.mark("X.NS", 1100.0)
    b.place(market(side=Action.SELL, qty=100))

    assert b.position("X.NS").is_flat
    assert len(b.trades) == 1
    trade = b.trades[0]
    assert trade.gross_pnl_inr == pytest.approx(10_000.0)
    assert trade.charges_inr > 0, "an Indian round trip is never free"
    assert trade.net_pnl_inr == pytest.approx(trade.gross_pnl_inr - trade.charges_inr)
    # The whole result, charges included, lands on cash.
    assert b.funds().cash_inr == pytest.approx(CASH + trade.net_pnl_inr)


def test_a_partial_exit_leaves_the_rest_open():
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=100))
    b.place(market(side=Action.SELL, qty=40))
    position = b.position("X.NS")
    assert position.quantity == 60
    assert position.average_price == pytest.approx(1000.0), "basis should not move"
    assert len(b.trades) == 1 and b.trades[0].quantity == 40


def test_delivery_and_intraday_are_separate_positions():
    """Netting them would hide an overnight exposure that is not held."""
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=50, product=ProductType.CNC))
    b.place(market(qty=50, product=ProductType.MIS))
    assert len(b.positions()) == 2
    assert b.position("X.NS", ProductType.CNC).quantity == 50
    assert b.position("X.NS", ProductType.MIS).quantity == 50


# -------------------------------------------------------------- lifecycle


def test_square_off_closes_intraday_and_leaves_delivery_alone():
    b = broker()
    b.mark("X.NS", 1000.0)
    b.place(market(qty=50, product=ProductType.CNC))
    b.place(market(qty=50, product=ProductType.MIS))

    closed = b.square_off_intraday()
    assert len(closed) == 1
    assert b.position("X.NS", ProductType.MIS).is_flat
    assert b.position("X.NS", ProductType.CNC).quantity == 50
    assert b.funds().blocked_margin_inr == pytest.approx(0.0)


def test_cancel_all_clears_the_resting_book():
    b = broker()
    for price in (900.0, 950.0):
        b.place(Order(symbol="X.NS", side=Action.BUY, quantity=10,
                      order_type=OrderType.LIMIT, limit_price=price))
    assert b.cancel_all() == 2
    assert b.orders(open_only=True) == []


# ------------------------------------------------------------ persistence


def test_a_book_survives_a_round_trip_to_disk(tmp_path):
    path = tmp_path / "book.json"
    b = broker(state_path=path)
    b.mark("X.NS", 1000.0)
    b.place(market(qty=100))
    b.place(Order(symbol="X.NS", side=Action.SELL, quantity=100,
                  order_type=OrderType.SL, trigger_price=950.0))
    b.save()

    restored = PaperBroker.load(path)
    assert restored.funds().cash_inr == pytest.approx(b.funds().cash_inr)
    assert restored.position("X.NS").quantity == 100
    assert restored.position("X.NS").average_price == pytest.approx(1000.0)
    assert len(restored.orders(open_only=True)) == 1, "the stop must come back"

    # And the restored book still works: the stop should still fire.
    restored.mark("X.NS", 940.0)
    assert restored.position("X.NS").is_flat


def test_the_event_log_records_what_happened(tmp_path):
    path = tmp_path / "book.json"
    b = broker(state_path=path)
    b.mark("X.NS", 1000.0)
    b.place(market(qty=10))
    b.place(market(qty=0))  # rejected

    events = [json.loads(line) for line in
              (tmp_path / "book.events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "order_placed" in kinds and "order_filled" in kinds
    assert "order_rejected" in kinds


def test_saving_without_a_path_is_an_error():
    with pytest.raises(ValueError):
        broker().save()


# ------------------------------------------------- decision -> orders


def _decision(*, approved=True, action=Action.BUY, quantity=100, stop=950.0):
    class _D:
        symbol = "X.NS"
        run_id = "RUN-1"
    d = _D()
    d.approved = approved
    d.proposal = TradeProposal(
        role=AgentRole.TRADER, symbol="X.NS", summary="s",
        action=action, instrument=Instrument.CASH,
        product_type=ProductType.CNC, time_horizon=TimeHorizon.SWING,
        entry_zone=EntryZone(low=990.0, high=1010.0), stop_loss=stop,
        targets=[Target(price=1100.0, book_pct=100.0)],
        position_size_pct=10.0, quantity=quantity, risk_amount_inr=5_000.0,
    ) if quantity else None
    return d


def test_an_approved_decision_becomes_an_entry_and_a_stop():
    orders = orders_for_decision(_decision())
    assert len(orders) == 2
    entry, stop = orders
    assert entry.order_type == OrderType.LIMIT
    # The top of the band for a buy: inside it, without chasing.
    assert entry.limit_price == pytest.approx(1010.0)
    assert entry.side == Action.BUY
    assert stop.order_type == OrderType.SL
    assert stop.side == Action.SELL, "the stop closes the position"
    assert stop.trigger_price == pytest.approx(950.0)
    assert entry.tag == "RUN-1"


def test_a_sell_decision_enters_at_the_bottom_of_the_band():
    orders = orders_for_decision(_decision(action=Action.SELL))
    assert orders[0].limit_price == pytest.approx(990.0)
    assert orders[1].side == Action.BUY


@pytest.mark.parametrize("kwargs", [
    {"approved": False},
    {"action": Action.HOLD},
    {"quantity": 0},
])
def test_nothing_is_ordered_for_a_decision_that_asks_for_no_position(kwargs):
    assert orders_for_decision(_decision(**kwargs)) == []


def test_a_decision_with_no_stop_still_gets_its_entry():
    orders = orders_for_decision(_decision(stop=None))
    assert len(orders) == 1
    assert orders[0].order_type == OrderType.LIMIT


def test_none_is_handled():
    assert orders_for_decision(None) == []


# ---------------------------------------------------------------- summary


def test_summary_reports_a_losing_and_a_winning_trade():
    b = broker()
    b.mark("A.NS", 1000.0)
    b.place(market(symbol="A.NS", qty=100))
    b.mark("A.NS", 1200.0)
    b.place(market(symbol="A.NS", side=Action.SELL, qty=100))

    b.mark("B.NS", 1000.0)
    b.place(market(symbol="B.NS", qty=100))
    b.mark("B.NS", 900.0)
    b.place(market(symbol="B.NS", side=Action.SELL, qty=100))

    s = b.summary()
    assert s["closed_trades"] == 2
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["charges_inr"] > 0
    assert s["net_pnl_inr"] == pytest.approx(s["gross_pnl_inr"] - s["charges_inr"])


def test_an_empty_book_summarises_without_trades():
    s = broker().summary()
    assert s["closed_trades"] == 0
    assert s["hit_rate"] is None
    assert s["equity_inr"] == pytest.approx(CASH)
