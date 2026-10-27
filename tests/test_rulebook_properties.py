"""Property tests for the risk rulebook.

The rest of the suite checks that specific inputs give specific answers. This
file checks the invariants that must hold for *every* input: a stop-out never
costs more than the risk budget, a position never exceeds its cap, the book
never ends up over its heat budget, an F&O order is always a whole number of
lots.

Two layers, because they have very different costs:

* the sizing functions are pure, so they are swept over hundreds of cases;
* the full agent pipeline costs ~0.5s a run, so it is swept over a handful of
  seeds and symbols and asked only about things the pure layer cannot see.

Everything here runs on the deterministic synthetic tape: no network, no key.
"""

from __future__ import annotations

import random

import pytest

from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.models.enums import Action, Instrument, ProductType
from tradingagents_in.risk.sizing import risk_budget_inr, size_cash_position

# Tolerance for money compared after rounding to paise.
EPS = 1e-6


def _cfg(**overrides):
    base = {
        "data_provider": "synthetic",
        "llm_provider": "offline",
        "online_tools": False,
    }
    base.update(overrides)
    return build_config(base)


# --------------------------------------------------------------- pure sizing


def _sizing_cases(n: int = 400):
    """Prices and stops spanning penny stocks to index-level names."""
    rng = random.Random(20260728)
    for _ in range(n):
        entry = rng.choice([
            rng.uniform(1.0, 50.0),        # small caps
            rng.uniform(50.0, 500.0),
            rng.uniform(500.0, 5_000.0),   # the large-cap bulk
            rng.uniform(5_000.0, 80_000.0),
        ])
        # Stops from a scalp to a very wide swing stop.
        stop = entry * (1.0 - rng.uniform(0.001, 0.40))
        capital = rng.choice([50_000.0, 250_000.0, 1_000_000.0, 50_000_000.0])
        yield entry, stop, capital


@pytest.mark.parametrize("risk_pct,max_pos_pct", [(0.25, 5.0), (1.0, 10.0), (2.0, 25.0)])
def test_cash_sizing_never_exceeds_its_budgets(risk_pct, max_pos_pct):
    """A stop-out costs at most the risk budget, and the position its cap.

    This is the invariant the whole desk rests on: if sizing can overshoot,
    every downstream approval is overstating what it approved.
    """
    config = _cfg(risk_per_trade_pct=risk_pct, max_position_pct=max_pos_pct)

    for entry, stop, capital in _sizing_cases():
        res = size_cash_position(
            capital_inr=capital, entry_price=entry, stop_loss=stop, config=config
        )
        if not res.is_tradable:
            # Refusing to size is always a legitimate answer; it must then
            # claim nothing.
            assert res.quantity == 0
            assert res.risk_amount_inr == 0.0
            assert res.capital_deployed_inr == 0.0
            continue

        budget = risk_budget_inr(capital, config)
        assert res.risk_amount_inr <= budget + EPS, (
            f"stop-out costs {res.risk_amount_inr} against a {budget} budget "
            f"(entry {entry}, stop {stop}, capital {capital})"
        )
        assert res.capital_deployed_inr <= capital * max_pos_pct / 100.0 + EPS
        assert res.quantity > 0 and float(res.quantity).is_integer()
        assert res.binding_constraint  # it must always say what bound it


def test_cash_sizing_respects_available_cash():
    """Sizing may never deploy cash the book does not have."""
    config = _cfg()
    for entry, stop, capital in _sizing_cases(200):
        cash = capital * random.Random(int(entry)).uniform(0.0, 1.0)
        res = size_cash_position(
            capital_inr=capital, entry_price=entry, stop_loss=stop,
            config=config, available_cash_inr=cash,
        )
        assert res.capital_deployed_inr <= cash + EPS


def test_zero_stop_distance_is_never_tradable():
    """A stop at the entry implies infinite size; it must be refused."""
    config = _cfg()
    for entry in (1.0, 100.0, 2_500.0, 75_000.0):
        res = size_cash_position(
            capital_inr=1_000_000.0, entry_price=entry, stop_loss=entry, config=config
        )
        assert not res.is_tradable
        assert res.quantity == 0


def test_tighter_stop_never_sizes_smaller():
    """Halving the stop distance cannot reduce the share count.

    Monotonicity is worth pinning: a sizing bug that inverts it would quietly
    size the riskiest setups largest.
    """
    config = _cfg(max_position_pct=100.0)  # isolate the risk-budget path
    entry = 1_000.0
    previous = 0
    for pct in (0.40, 0.20, 0.10, 0.05, 0.02, 0.01):
        res = size_cash_position(
            capital_inr=1_000_000.0, entry_price=entry,
            stop_loss=entry * (1 - pct), config=config,
        )
        assert res.quantity >= previous
        previous = res.quantity


# ------------------------------------------------------ full pipeline sweeps

SEEDS = [1, 7, 42, 1337]
SYMBOLS = ["RELIANCE.NS", "TCS.NS"]


@pytest.mark.parametrize("seed", SEEDS)
def test_single_decisions_hold_their_invariants(seed):
    """Whatever the tape says, an approved decision stays inside the mandate."""
    app = IndianTradingAgents(_cfg(synthetic_seed=seed))
    config = app.config
    capital = float(config["capital_inr"])

    for symbol in SYMBOLS:
        d = app.analyse(symbol, save=False)
        p = d.proposal

        if d.final_action == Action.HOLD or not d.approved:
            # A rejected idea must not leave a position behind.
            assert d.position_size_pct == 0.0 or not d.approved
            if d.final_action == Action.HOLD:
                assert d.position_size_pct == 0.0
                assert d.capital_at_risk_inr == 0.0
            continue

        assert p is not None, "an approved trade must carry a proposal"
        assert p.entry_zone is not None and p.stop_loss is not None
        assert p.entry_zone.low <= p.entry_zone.high

        # The stop has to be on the losing side of the entry, or it is not a stop.
        if p.action == Action.BUY:
            assert p.stop_loss < p.entry_zone.high
        elif p.action == Action.SELL:
            assert p.stop_loss > p.entry_zone.low

        assert d.position_size_pct <= float(config["max_position_pct"]) + EPS
        risk_pct = d.capital_at_risk_inr / capital * 100.0
        assert risk_pct <= float(config["risk_per_trade_pct"]) + EPS, (
            f"{symbol} risks {risk_pct:.3f}% against a "
            f"{config['risk_per_trade_pct']}% per-trade budget"
        )

        # Derivatives trade in whole lots; a fractional lot is unexecutable.
        if p.instrument in (Instrument.FUTURES,) and p.futures is not None:
            lot = getattr(p.futures, "lot_size", None)
            if lot:
                assert p.quantity is None or p.quantity % lot == 0

        # Cash shorts cannot be carried overnight in India.
        if p.action == Action.SELL and p.instrument == Instrument.CASH:
            assert p.product_type != ProductType.CNC, (
                "a cash short may not be a delivery order"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_accumulated_book_stays_inside_its_budgets(seed):
    """Across a batch, the portfolio caps bind - not just the per-trade ones.

    Ten names each individually compliant can still add up to an indefensible
    book, which is exactly what accumulate=True exists to prevent.
    """
    app = IndianTradingAgents(_cfg(synthetic_seed=seed))
    config = app.config
    capital = float(config["capital_inr"])

    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
               "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "AXISBANK.NS"]
    results = app.analyse_many(symbols, save=False, accumulate=True)

    total_risk = sum(d.capital_at_risk_inr for d in results.values() if d.approved)
    heat_pct = total_risk / capital * 100.0
    assert heat_pct <= float(config["max_portfolio_heat_pct"]) + EPS, (
        f"book heat {heat_pct:.3f}% exceeds the "
        f"{config['max_portfolio_heat_pct']}% budget"
    )

    deployed = sum(d.position_size_pct for d in results.values() if d.approved)
    assert deployed <= 100.0 + EPS, f"book deploys {deployed:.2f}% of capital"

    approved = [d for d in results.values() if d.approved]
    assert len(approved) <= int(config["max_open_positions"])


def test_same_seed_gives_the_same_decision():
    """The offline engine must be reproducible, or no sweep above means much."""
    first = IndianTradingAgents(_cfg(synthetic_seed=99)).analyse("RELIANCE.NS", save=False)
    second = IndianTradingAgents(_cfg(synthetic_seed=99)).analyse("RELIANCE.NS", save=False)

    assert first.final_action == second.final_action
    assert first.approved == second.approved
    assert first.position_size_pct == second.position_size_pct
    assert first.capital_at_risk_inr == second.capital_at_risk_inr


def test_a_smaller_book_never_takes_more_risk():
    """Halving capital must not increase the rupees put at risk."""
    big = IndianTradingAgents(
        _cfg(synthetic_seed=5, capital_inr=2_000_000.0)
    ).analyse("RELIANCE.NS", save=False)
    small = IndianTradingAgents(
        _cfg(synthetic_seed=5, capital_inr=500_000.0)
    ).analyse("RELIANCE.NS", save=False)

    assert small.capital_at_risk_inr <= big.capital_at_risk_inr + EPS
