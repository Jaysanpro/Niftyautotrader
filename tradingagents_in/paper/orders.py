"""Turn an approved decision into the orders that express it.

The desk's output is a plan: a band to enter in, a stop that invalidates it,
targets to book at. This is the one translation step to orders, kept apart
from the broker so that a live adapter reuses it unchanged - the mapping from
plan to order is the same whoever fills it.

Two choices worth stating, because both could reasonably go the other way:

* **Entry is a limit at the top of the band, not a market order.** The desk's
  own note says chasing above the band changes the reward-risk the approval
  rested on. A market order is exactly that chase.
* **The stop goes in with the entry**, as a resting order, because the
  portfolio manager's conditions say so in as many words. A stop placed
  "after the fill" is a stop that is missing whenever it is needed most.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.models.enums import Action, Instrument, ProductType
from tradingagents_in.paper.models import Order, OrderType


def orders_for_decision(decision: Any, *, tag: str | None = None) -> list[Order]:
    """Entry and stop for one approved decision, or [] if it is not tradable.

    Returns an empty list for a HOLD, a rejection, or a proposal with no
    quantity - all three mean the desk is not asking for a position, and
    turning them into orders anyway would trade on an answer nobody gave.
    """
    if decision is None or not getattr(decision, "approved", False):
        return []

    proposal = getattr(decision, "proposal", None)
    if proposal is None:
        return []
    action = proposal.action
    quantity = int(proposal.quantity or 0)
    if action == Action.HOLD or quantity <= 0:
        return []
    if proposal.entry_zone is None:
        return []

    product = proposal.product_type or ProductType.CNC
    instrument = proposal.instrument or Instrument.CASH
    tag = tag or getattr(decision, "run_id", "") or ""
    long_side = action == Action.BUY

    # Enter at the edge of the band that does not require chasing: the top for
    # a buy, the bottom for a sell. Inside the band is a fill; beyond it is a
    # different trade from the one that was approved.
    entry_price = proposal.entry_zone.high if long_side else proposal.entry_zone.low

    orders = [
        Order(
            symbol=decision.symbol,
            side=action,
            quantity=quantity,
            product=product,
            instrument=instrument,
            order_type=OrderType.LIMIT,
            limit_price=float(entry_price),
            tag=tag,
        )
    ]

    if proposal.stop_loss:
        orders.append(
            Order(
                symbol=decision.symbol,
                # The stop closes the position, so it is the opposite side.
                side=Action.SELL if long_side else Action.BUY,
                quantity=quantity,
                product=product,
                instrument=instrument,
                order_type=OrderType.SL,
                trigger_price=float(proposal.stop_loss),
                tag=f"{tag}:stop" if tag else "stop",
            )
        )
    return orders
