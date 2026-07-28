"""Indian transaction-cost model.

Charges matter here in a way they do not in most markets. A 1:1.5 reward-risk
idea on a 0.4% stop is *negative* after STT, stamp duty, exchange charges and
GST, so the risk manager checks net reward-risk, not gross.

Rates are the standard discount-broker schedule (equity delivery, equity
intraday, equity/index futures, options). They are configurable because
exchanges revise them and full-service brokers charge more.

Reference: NSE/BSE transaction-charge circulars, Finance Act STT schedule, and
the Indian Stamp Act 2019 uniform rates. Verify against your broker's contract
note before relying on the absolute numbers - the shape of the calculation is
what matters to the sizing decision.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.models.enums import Action, Instrument, ProductType
from tradingagents_in.models.trade import CostEstimate

# Percentages of turnover unless the name says otherwise.
DEFAULT_RATES: dict[str, float] = {
    # Brokerage: discount brokers charge zero on delivery and a capped
    # percentage elsewhere.
    "brokerage_pct_delivery": 0.0,
    "brokerage_pct_intraday": 0.03,
    "brokerage_cap_inr": 20.0,  # per executed order
    "brokerage_flat_fno_inr": 20.0,
    # Securities Transaction Tax
    "stt_pct_delivery_buy": 0.1,
    "stt_pct_delivery_sell": 0.1,
    "stt_pct_intraday_sell": 0.025,
    "stt_pct_futures_sell": 0.02,
    "stt_pct_options_sell_premium": 0.1,
    # Exchange transaction charges
    "exchange_txn_pct": 0.00297,  # equity cash, NSE
    "exchange_txn_pct_futures": 0.00173,
    "exchange_txn_pct_options": 0.03503,  # on premium turnover
    # Regulator
    "sebi_turnover_pct": 0.0001,  # Rs 10 per crore
    # Stamp duty, buy side only, uniform under the Indian Stamp Act 2019
    "stamp_duty_pct_buy_delivery": 0.015,
    "stamp_duty_pct_buy_intraday": 0.003,
    "stamp_duty_pct_buy_futures": 0.002,
    "stamp_duty_pct_buy_options": 0.003,
    # GST applies to brokerage + exchange + SEBI charges, not to STT or stamp
    "gst_pct_on_charges": 18.0,
}


def _rates(config: dict[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_RATES)
    merged.update((config or {}).get("costs", {}) or {})
    return merged


def estimate_costs(
    *,
    entry_price: float,
    exit_price: float | None = None,
    quantity: int = 0,
    instrument: Instrument = Instrument.CASH,
    product_type: ProductType = ProductType.CNC,
    action: Action = Action.BUY,
    config: dict[str, Any] | None = None,
) -> CostEstimate:
    """Round-trip charges for one position.

    ``quantity`` is shares for cash, or lots x lot_size for F&O. For options,
    ``entry_price``/``exit_price`` are premiums per share: STT and exchange
    charges on options are levied on premium turnover, not notional, which is
    why an option trade looks cheap until the sell-side STT lands.
    """
    r = _rates(config)
    exit_price = exit_price if exit_price is not None else entry_price
    if quantity <= 0 or entry_price <= 0:
        return CostEstimate()

    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    turnover = buy_turnover + sell_turnover

    is_delivery = instrument == Instrument.CASH and product_type == ProductType.CNC

    # ---------------------------------------------------------- brokerage
    if instrument == Instrument.CASH:
        if is_delivery:
            per_leg = min(
                buy_turnover * r["brokerage_pct_delivery"] / 100, r["brokerage_cap_inr"]
            )
            brokerage = per_leg * 2
        else:
            brokerage = 2 * min(
                buy_turnover * r["brokerage_pct_intraday"] / 100, r["brokerage_cap_inr"]
            )
    else:
        brokerage = 2 * r["brokerage_flat_fno_inr"]

    # ---------------------------------------------------------------- STT
    if instrument == Instrument.CASH:
        if is_delivery:
            stt = (
                buy_turnover * r["stt_pct_delivery_buy"] / 100
                + sell_turnover * r["stt_pct_delivery_sell"] / 100
            )
        else:
            stt = sell_turnover * r["stt_pct_intraday_sell"] / 100
    elif instrument == Instrument.FUTURES:
        stt = sell_turnover * r["stt_pct_futures_sell"] / 100
    else:
        stt = sell_turnover * r["stt_pct_options_sell_premium"] / 100

    # ------------------------------------------------- exchange and SEBI
    txn_pct = {
        Instrument.CASH: r["exchange_txn_pct"],
        Instrument.FUTURES: r["exchange_txn_pct_futures"],
        Instrument.OPTIONS: r["exchange_txn_pct_options"],
    }[instrument]
    exchange_txn = turnover * txn_pct / 100
    sebi = turnover * r["sebi_turnover_pct"] / 100

    # -------------------------------------------------------- stamp duty
    stamp_pct = {
        Instrument.CASH: (
            r["stamp_duty_pct_buy_delivery"] if is_delivery
            else r["stamp_duty_pct_buy_intraday"]
        ),
        Instrument.FUTURES: r["stamp_duty_pct_buy_futures"],
        Instrument.OPTIONS: r["stamp_duty_pct_buy_options"],
    }[instrument]
    # A short-first trade still pays stamp duty on the buy-to-cover leg.
    stamp_base = sell_turnover if action == Action.SHORT else buy_turnover
    stamp = stamp_base * stamp_pct / 100

    gst = (brokerage + exchange_txn + sebi) * r["gst_pct_on_charges"] / 100
    total = brokerage + stt + exchange_txn + sebi + stamp + gst

    return CostEstimate(
        brokerage_inr=round(brokerage, 2),
        stt_inr=round(stt, 2),
        exchange_txn_inr=round(exchange_txn, 2),
        sebi_charges_inr=round(sebi, 2),
        stamp_duty_inr=round(stamp, 2),
        gst_inr=round(gst, 2),
        total_inr=round(total, 2),
        total_pct_of_turnover=round(total / turnover * 100, 4) if turnover else 0.0,
    )


def breakeven_move_pct(costs: CostEstimate, entry_price: float, quantity: int) -> float | None:
    """How far price must move just to cover the round trip.

    This is the number that kills scalping small-cap cash positions: if the
    breakeven move is 0.35% and the setup targets 0.8%, nearly half the edge is
    already gone.
    """
    if entry_price <= 0 or quantity <= 0 or costs.total_inr <= 0:
        return None
    return round(costs.total_inr / (entry_price * quantity) * 100, 4)


def net_reward_risk(
    *,
    entry_price: float,
    stop_loss: float,
    target: float,
    quantity: int,
    instrument: Instrument = Instrument.CASH,
    product_type: ProductType = ProductType.CNC,
    action: Action = Action.BUY,
    config: dict[str, Any] | None = None,
) -> tuple[float | None, CostEstimate]:
    """Reward-risk after charges on both legs.

    Costs are charged against the reward and added to the risk, which is the
    conservative and correct way round: you pay them whether you win or lose.
    """
    if entry_price <= 0 or quantity <= 0:
        return None, CostEstimate()

    long_side = action in (Action.BUY,)
    gross_reward = (target - entry_price) if long_side else (entry_price - target)
    gross_risk = (entry_price - stop_loss) if long_side else (stop_loss - entry_price)
    if gross_risk <= 0:
        return None, CostEstimate()

    win_costs = estimate_costs(
        entry_price=entry_price, exit_price=target, quantity=quantity,
        instrument=instrument, product_type=product_type, action=action, config=config,
    )
    loss_costs = estimate_costs(
        entry_price=entry_price, exit_price=stop_loss, quantity=quantity,
        instrument=instrument, product_type=product_type, action=action, config=config,
    )

    net_reward = gross_reward * quantity - win_costs.total_inr
    net_risk = gross_risk * quantity + loss_costs.total_inr
    if net_risk <= 0:
        return None, win_costs
    return round(max(0.0, net_reward) / net_risk, 3), win_costs


def describe(costs: CostEstimate) -> str:
    from tradingagents_in.utils.formatting import inr_plain

    return (
        f"brokerage {inr_plain(costs.brokerage_inr)}, STT {inr_plain(costs.stt_inr)}, "
        f"exchange {inr_plain(costs.exchange_txn_inr)}, SEBI {inr_plain(costs.sebi_charges_inr)}, "
        f"stamp {inr_plain(costs.stamp_duty_inr)}, GST {inr_plain(costs.gst_inr)} "
        f"= {inr_plain(costs.total_inr)} ({costs.total_pct_of_turnover:.3f}% of turnover)"
    )
