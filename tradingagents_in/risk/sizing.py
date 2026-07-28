"""Position sizing.

One rule drives everything: the loss at the stop is a fixed fraction of
capital. Size is an *output* of the stop distance, never an input. A wide stop
means a small position, not a wider risk budget.

Two India-specific complications are handled explicitly:

* **Lot indivisibility.** F&O trades in lots. If one lot of BANKNIFTY risks
  more than the whole risk budget, the honest answer is zero lots, not "round
  down to something". The sizer says so rather than quietly over-risking.
* **Margin, not notional.** A futures position deploys margin (SPAN +
  exposure), so the capital check and the concentration check look at different
  numbers. Treating notional as deployed capital blocks trades that are
  affordable; treating margin as the exposure understates concentration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingagents_in.config.universe import lot_size as universe_lot_size
from tradingagents_in.dataflows.fno import approx_futures_margin
from tradingagents_in.models.enums import Action, Instrument


@dataclass
class SizingResult:
    """What the sizer decided, and why it could not do better."""

    quantity: int = 0
    lots: int = 0
    lot_size: int | None = None
    capital_deployed_inr: float = 0.0
    exposure_inr: float = 0.0  # notional; differs from deployed for F&O
    risk_amount_inr: float = 0.0
    position_pct: float = 0.0
    risk_pct_of_capital: float = 0.0
    binding_constraint: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def is_tradable(self) -> bool:
        return self.quantity > 0


def risk_budget_inr(capital_inr: float, config: dict[str, Any]) -> float:
    return capital_inr * float(config.get("risk_per_trade_pct", 1.0)) / 100.0


def size_cash_position(
    *,
    capital_inr: float,
    entry_price: float,
    stop_loss: float,
    config: dict[str, Any],
    available_cash_inr: float | None = None,
) -> SizingResult:
    """Shares to buy (or short) so that a stop-out costs exactly the risk budget."""
    result = SizingResult()
    stop_distance = abs(entry_price - stop_loss)
    if entry_price <= 0 or stop_distance <= 0:
        result.binding_constraint = "invalid entry/stop"
        result.notes.append("Stop-loss must be a real distance from entry; cannot size.")
        return result

    budget = risk_budget_inr(capital_inr, config)
    qty_by_risk = int(budget // stop_distance)

    max_position_inr = capital_inr * float(config.get("max_position_pct", 10.0)) / 100.0
    qty_by_position = int(max_position_inr // entry_price)

    cash = capital_inr if available_cash_inr is None else available_cash_inr
    qty_by_cash = int(max(0.0, cash) // entry_price)

    candidates = {
        "risk budget": qty_by_risk,
        "max position size": qty_by_position,
        "available cash": qty_by_cash,
    }
    quantity = min(candidates.values())
    result.binding_constraint = min(candidates, key=lambda k: candidates[k])

    if quantity <= 0:
        result.notes.append(
            f"Not tradable: the {result.binding_constraint} allows zero shares at "
            f"Rs {entry_price:,.2f} with a Rs {stop_distance:,.2f} stop."
        )
        return result

    result.quantity = quantity
    result.lot_size = 1
    result.capital_deployed_inr = round(quantity * entry_price, 2)
    result.exposure_inr = result.capital_deployed_inr
    result.risk_amount_inr = round(quantity * stop_distance, 2)
    result.position_pct = round(result.capital_deployed_inr / capital_inr * 100, 3)
    result.risk_pct_of_capital = round(result.risk_amount_inr / capital_inr * 100, 3)

    if result.binding_constraint != "risk budget":
        result.notes.append(
            f"Size capped by {result.binding_constraint}, so the trade risks "
            f"{result.risk_pct_of_capital:.2f}% rather than the full "
            f"{config.get('risk_per_trade_pct', 1.0):.2f}% budget."
        )
    return result


def size_futures_position(
    *,
    capital_inr: float,
    entry_price: float,
    stop_loss: float,
    base_symbol: str,
    config: dict[str, Any],
    lot_size: int | None = None,
    is_index: bool = False,
    available_cash_inr: float | None = None,
) -> SizingResult:
    """Lots of a futures contract, sized on stop distance and margin."""
    result = SizingResult()
    size = lot_size or universe_lot_size(base_symbol)
    if not size:
        result.binding_constraint = "unknown lot size"
        result.notes.append(
            f"No lot size on record for {base_symbol}. Exchange lot sizes are revised "
            "periodically - confirm from the NSE F&O contract file before trading."
        )
        return result

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        result.binding_constraint = "invalid stop"
        return result

    result.lot_size = size
    budget = risk_budget_inr(capital_inr, config)
    risk_per_lot = stop_distance * size
    margin_per_lot = approx_futures_margin(entry_price, base_symbol, 1, is_index) or (
        entry_price * size * 0.20
    )

    lots_by_risk = int(budget // risk_per_lot)
    cash = capital_inr if available_cash_inr is None else available_cash_inr
    lots_by_margin = int(max(0.0, cash) // margin_per_lot)
    # Concentration is measured on notional exposure, not on margin posted.
    max_exposure = capital_inr * float(config.get("max_position_pct", 10.0)) / 100.0
    lots_by_exposure = int(max_exposure // (entry_price * size))

    candidates = {
        "risk budget": lots_by_risk,
        "available margin": lots_by_margin,
        "max notional exposure": lots_by_exposure,
    }
    lots = min(candidates.values())
    result.binding_constraint = min(candidates, key=lambda k: candidates[k])

    if lots <= 0:
        result.notes.append(
            f"One lot ({size} qty) risks Rs {risk_per_lot:,.0f} against a budget of "
            f"Rs {budget:,.0f} and needs about Rs {margin_per_lot:,.0f} margin. "
            "The contract is indivisible, so the correct size is zero lots - take the "
            "trade in the cash segment or with a defined-risk option spread instead."
        )
        return result

    result.lots = lots
    result.quantity = lots * size
    result.capital_deployed_inr = round(lots * margin_per_lot, 2)
    result.exposure_inr = round(lots * size * entry_price, 2)
    result.risk_amount_inr = round(lots * risk_per_lot, 2)
    result.position_pct = round(result.exposure_inr / capital_inr * 100, 3)
    result.risk_pct_of_capital = round(result.risk_amount_inr / capital_inr * 100, 3)
    result.notes.append(
        f"Margin is an estimate ({margin_per_lot:,.0f}/lot); the broker's SPAN file is "
        "authoritative and peak-margin rules apply intraday."
    )
    return result


def size_option_position(
    *,
    capital_inr: float,
    premium: float,
    base_symbol: str,
    config: dict[str, Any],
    lot_size: int | None = None,
    max_loss_per_share: float | None = None,
) -> SizingResult:
    """Lots of a long option (or a debit spread).

    For a long option the maximum loss is the premium paid, so the premium
    outlay *is* the risk budget. ``max_loss_per_share`` lets a spread pass its
    net debit instead.
    """
    result = SizingResult()
    size = lot_size or universe_lot_size(base_symbol)
    if not size or premium <= 0:
        result.binding_constraint = "unknown lot size or premium"
        return result

    result.lot_size = size
    loss_per_share = max_loss_per_share if max_loss_per_share is not None else premium
    budget = risk_budget_inr(capital_inr, config)
    risk_per_lot = loss_per_share * size
    cost_per_lot = premium * size

    lots_by_risk = int(budget // risk_per_lot)
    lots_by_cash = int((capital_inr * float(config.get("max_position_pct", 10.0)) / 100.0)
                       // cost_per_lot)
    candidates = {"risk budget": lots_by_risk, "premium outlay cap": lots_by_cash}
    lots = min(candidates.values())
    result.binding_constraint = min(candidates, key=lambda k: candidates[k])

    if lots <= 0:
        result.notes.append(
            f"One lot costs Rs {cost_per_lot:,.0f} in premium against a Rs {budget:,.0f} "
            "risk budget. Zero lots is the correct answer - a smaller strike distance or "
            "a spread would reduce the debit."
        )
        return result

    result.lots = lots
    result.quantity = lots * size
    result.capital_deployed_inr = round(lots * cost_per_lot, 2)
    result.exposure_inr = result.capital_deployed_inr
    result.risk_amount_inr = round(lots * risk_per_lot, 2)
    result.position_pct = round(result.capital_deployed_inr / capital_inr * 100, 3)
    result.risk_pct_of_capital = round(result.risk_amount_inr / capital_inr * 100, 3)
    result.notes.append(
        "Long-option risk is the full premium: the position can lose 100% without the "
        "underlying ever hitting the technical stop."
    )
    return result


def size_for_proposal(
    *,
    capital_inr: float,
    entry_price: float,
    stop_loss: float | None,
    instrument: Instrument,
    action: Action,
    base_symbol: str,
    config: dict[str, Any],
    lot_size: int | None = None,
    is_index: bool = False,
    available_cash_inr: float | None = None,
    option_premium: float | None = None,
    option_max_loss_per_share: float | None = None,
) -> SizingResult:
    """Dispatch to the right sizer for the proposed instrument."""
    if action == Action.HOLD:
        return SizingResult(binding_constraint="no position proposed")

    if instrument == Instrument.OPTIONS:
        return size_option_position(
            capital_inr=capital_inr,
            premium=option_premium or 0.0,
            base_symbol=base_symbol,
            config=config,
            lot_size=lot_size,
            max_loss_per_share=option_max_loss_per_share,
        )

    if stop_loss is None:
        return SizingResult(
            binding_constraint="no stop-loss",
            notes=["A position without a stop cannot be sized; risk is undefined."],
        )

    if instrument == Instrument.FUTURES:
        return size_futures_position(
            capital_inr=capital_inr,
            entry_price=entry_price,
            stop_loss=stop_loss,
            base_symbol=base_symbol,
            config=config,
            lot_size=lot_size,
            is_index=is_index,
            available_cash_inr=available_cash_inr,
        )

    return size_cash_position(
        capital_inr=capital_inr,
        entry_price=entry_price,
        stop_loss=stop_loss,
        config=config,
        available_cash_inr=available_cash_inr,
    )
