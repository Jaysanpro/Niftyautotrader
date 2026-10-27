"""Settlement logic for the walk-forward harness.

The arithmetic here is simple and the failure modes are not. An off-by-one in
which bars count as "after the decision" does not raise; it just quietly pays
the strategy for knowing the future. These tests pin the boundary with
hand-built price frames so the expected answer is arithmetic, not opinion.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tradingagents_in.backtest import (
    EXIT_HORIZON,
    EXIT_STOP,
    EXIT_TARGET,
    NEVER_ENTERED,
    render_summary,
    settle,
    trading_sessions,
    walk_forward,
)
from tradingagents_in.models.enums import (
    Action,
    AgentRole,
    Instrument,
    ProductType,
    TimeHorizon,
)
from tradingagents_in.models.trade import CostEstimate, EntryZone, Target, TradeProposal


DECIDED = date(2026, 3, 16)


def _frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [1_000] * len(rows),
        },
        index=idx,
    )


class _Decision:
    """The slice of FinalDecision that settle() actually reads."""

    def __init__(self, proposal, symbol="TEST.NS", as_of=f"{DECIDED} 15:30 IST"):
        self.proposal = proposal
        self.symbol = symbol
        self.as_of = as_of


def _proposal(
    *, action=Action.BUY, low=100.0, high=102.0, stop=95.0, target=115.0,
    quantity=100, risk=500.0, charges=0.0,
):
    return TradeProposal(
        role=AgentRole.TRADER,
        symbol="TEST.NS",
        summary="test proposal",
        action=action,
        instrument=Instrument.CASH,
        product_type=ProductType.CNC,
        time_horizon=TimeHorizon.SWING,
        entry_zone=EntryZone(low=low, high=high),
        stop_loss=stop,
        targets=[Target(price=target, book_pct=100.0)] if target else [],
        position_size_pct=10.0,
        quantity=quantity,
        risk_amount_inr=risk,
        costs=CostEstimate(total_inr=charges) if charges else None,
    )


# ------------------------------------------------------------- the boundary


def test_the_decision_day_bar_is_never_tradable():
    """A decision dated d may not transact on d's own bar."""
    prices = _frame([
        # The decision day itself sits perfectly inside the entry band and then
        # rockets. If this bar is usable the strategy prints free money.
        (str(DECIDED), 101.0, 130.0, 100.5, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 103.0, 100.5, 101.0),
    ])
    trade = settle(_Decision(_proposal()), prices)
    assert trade is not None
    assert trade.entered_on == DECIDED + timedelta(days=1), (
        "entry must come from the first bar after the decision"
    )
    assert trade.entry_price == 101.0


def test_no_bars_after_the_decision_means_no_trade():
    prices = _frame([(str(DECIDED), 101.0, 102.0, 100.0, 101.0)])
    trade = settle(_Decision(_proposal()), prices)
    assert trade is not None
    assert not trade.was_taken
    assert trade.exit_reason == NEVER_ENTERED


# ----------------------------------------------------------------- entries


def test_price_never_reaching_the_band_is_not_filled():
    """The desk said only inside the band; filling outside scores a different trade."""
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 120.0, 125.0, 119.0, 124.0),
        (str(DECIDED + timedelta(days=2)), 126.0, 130.0, 125.0, 129.0),
    ])
    trade = settle(_Decision(_proposal(low=100.0, high=102.0)), prices)
    assert not trade.was_taken
    assert trade.pnl_inr == 0.0


def test_require_entry_band_false_fills_at_the_next_close():
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 120.0, 125.0, 119.0, 124.0),
    ])
    trade = settle(_Decision(_proposal()), prices, require_entry_band=False)
    assert trade.was_taken
    assert trade.entry_price == 124.0


# ------------------------------------------------------------------- exits


def test_a_long_stopped_out_loses_its_risk():
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 102.0, 100.5, 101.0),   # fill at 101
        (str(DECIDED + timedelta(days=2)), 100.0, 100.5, 94.0, 94.5),     # low 94 < stop 95
    ])
    trade = settle(_Decision(_proposal(stop=95.0, quantity=100)), prices)
    assert trade.exit_reason == EXIT_STOP
    assert trade.exit_price == 95.0
    # (95 - 101) * 100 = -600
    assert trade.pnl_inr == -600.0
    assert trade.r_multiple == pytest.approx(-1.0, abs=1e-6)


def test_a_long_hitting_its_target_wins():
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 102.0, 100.5, 101.0),
        (str(DECIDED + timedelta(days=2)), 110.0, 116.0, 109.0, 115.5),   # high >= 115
    ])
    trade = settle(_Decision(_proposal(stop=95.0, target=115.0, quantity=100)), prices)
    assert trade.exit_reason == EXIT_TARGET
    assert trade.pnl_inr == pytest.approx((115.0 - 101.0) * 100)
    # risk/share 6, reward/share 14
    assert trade.r_multiple == pytest.approx(14.0 / 6.0, abs=1e-3)


def test_a_bar_touching_both_stop_and_target_is_read_as_the_stop():
    """A daily bar cannot say which came first, so assume the worse one."""
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 102.0, 100.5, 101.0),
        (str(DECIDED + timedelta(days=2)), 101.0, 120.0, 90.0, 110.0),    # both touched
    ])
    trade = settle(_Decision(_proposal(stop=95.0, target=115.0)), prices)
    assert trade.exit_reason == EXIT_STOP


def test_an_untriggered_trade_exits_at_the_horizon():
    rows = [(str(DECIDED), 101.0, 102.0, 100.0, 101.0)]
    for i in range(1, 6):
        rows.append((str(DECIDED + timedelta(days=i)), 101.0, 103.0, 99.0, 101.0 + i * 0.1))
    trade = settle(_Decision(_proposal(stop=95.0, target=115.0)), _frame(rows), max_hold_days=4)
    assert trade.exit_reason == EXIT_HORIZON
    assert trade.was_taken


def test_a_short_mirrors_the_long_logic():
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 102.0, 100.5, 101.0),
        (str(DECIDED + timedelta(days=2)), 101.0, 107.0, 100.0, 106.0),   # high >= stop 105
    ])
    trade = settle(
        _Decision(_proposal(action=Action.SELL, stop=105.0, target=90.0, quantity=50)),
        prices,
    )
    assert trade.exit_reason == EXIT_STOP
    # Short: (entry - exit) * qty = (101 - 105) * 50 = -200
    assert trade.pnl_inr == -200.0


# ------------------------------------------------------------------ charges


def test_charges_come_out_of_both_pnl_and_r():
    """A winner that pays its costs is a smaller winner, in rupees and in R."""
    prices = _frame([
        (str(DECIDED), 101.0, 102.0, 100.0, 101.0),
        (str(DECIDED + timedelta(days=1)), 101.0, 102.0, 100.5, 101.0),
        (str(DECIDED + timedelta(days=2)), 110.0, 116.0, 109.0, 115.5),
    ])
    free = settle(_Decision(_proposal(charges=0.0)), prices)
    costly = settle(_Decision(_proposal(charges=400.0)), prices)

    gross = (115.0 - 101.0) * 100
    assert free.pnl_inr == pytest.approx(gross)
    assert costly.pnl_inr == pytest.approx(gross - 400.0)
    assert costly.r_multiple < free.r_multiple


# ------------------------------------------------------------- the harness


def test_a_hold_is_not_settled():
    proposal = _proposal(action=Action.HOLD, quantity=0)
    assert settle(_Decision(proposal), _frame([(str(DECIDED), 1, 1, 1, 1)])) is None


def test_trading_sessions_skips_weekends():
    # 2026-03-14 is a Saturday, 03-15 a Sunday.
    days = trading_sessions(date(2026, 3, 13), date(2026, 3, 17))
    assert date(2026, 3, 14) not in days
    assert date(2026, 3, 15) not in days
    assert date(2026, 3, 13) in days and date(2026, 3, 16) in days


def test_walk_forward_runs_end_to_end_on_the_synthetic_tape():
    """A smoke test: the loop decides, settles and reports without network."""
    from tradingagents_in.config.default_config import build_config
    from tradingagents_in.graph.runner import IndianTradingAgents

    app = IndianTradingAgents(build_config({
        "data_provider": "synthetic", "llm_provider": "offline",
        "online_tools": False, "synthetic_seed": 11,
    }))

    # Settlement prices are injected, because the synthetic provider builds a
    # fresh tape per gather and so has no consistent timeline to score against.
    rows = []
    start = date(2026, 3, 2)
    price = 2_500.0
    for i in range(30):
        day = start + timedelta(days=i)
        price *= 1.004 if i % 3 else 0.997
        rows.append((str(day), price, price * 1.02, price * 0.98, price))

    window = (date(2026, 3, 2), date(2026, 3, 6))
    result = walk_forward(
        app, ["RELIANCE.NS"], *window,
        prices={"RELIANCE.NS": _frame(rows)}, max_hold_days=5,
    )

    # 2026-03-04 is Holi, so the window holds four sessions, not five. Asserted
    # against the calendar rather than a hardcoded count, which is what caught
    # the holiday in the first place.
    expected = trading_sessions(*window)
    assert len(expected) == 4
    assert result.sessions == len(expected)
    assert result.decisions == len(expected)
    assert result.errors == []

    summary = result.summary()
    assert summary["sessions"] == len(expected)
    assert summary["decisions"] == len(expected)
    assert "sessions" in render_summary(result)
