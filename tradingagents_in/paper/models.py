"""Order, fill and position types, plus the execution interface itself.

The interface is the point of this package. The framework already decides
what to trade; nothing in it could place an order, and the existing
``BrokerFeed`` is read-only (quotes and candles). ``ExecutionBroker`` is the
other half, defined here against a simulator so that a live adapter is a swap
rather than a rewrite - and so the awkward parts (rejections, partial state,
charges, square-off) are met in simulation first, where they cost nothing.

Nothing here talks to a broker or the network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from tradingagents_in.models.enums import Action, Instrument, ProductType
from tradingagents_in.utils.timeutils import now_ist


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    # Stop-loss: rests until price crosses the trigger, then executes at market.
    SL = "SL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"      # accepted, resting, not yet executable
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    symbol: str
    side: Action                     # BUY or SELL
    quantity: int
    product: ProductType = ProductType.CNC
    instrument: Instrument = Instrument.CASH
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    trigger_price: float | None = None
    tag: str = ""                    # free-text, e.g. the run id that produced it

    order_id: str = field(default_factory=lambda: f"PO-{uuid.uuid4().hex[:10]}")
    status: OrderStatus = OrderStatus.PENDING
    placed_at: datetime = field(default_factory=now_ist)
    filled_at: datetime | None = None
    fill_price: float | None = None
    # Why a rejection happened, in the words the desk would use.
    message: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == OrderStatus.PENDING

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side == Action.BUY else -self.quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": str(self.side),
            "quantity": self.quantity,
            "product": str(self.product),
            "instrument": str(self.instrument),
            "order_type": str(self.order_type),
            "limit_price": self.limit_price,
            "trigger_price": self.trigger_price,
            "tag": self.tag,
            "status": str(self.status),
            "placed_at": self.placed_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": self.fill_price,
            "message": self.message,
        }


@dataclass
class Position:
    """A netted holding in one (symbol, product).

    Product is part of the identity because CNC and MIS are different
    positions at a real broker - one is delivery, the other is squared off the
    same day - and netting them would hide an overnight exposure that is not
    actually held.
    """

    symbol: str
    product: ProductType = ProductType.CNC
    instrument: Instrument = Instrument.CASH
    quantity: int = 0                # signed: negative is short
    average_price: float = 0.0
    realised_pnl_inr: float = 0.0
    charges_inr: float = 0.0
    last_price: float | None = None

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def exposure_inr(self) -> float:
        price = self.last_price if self.last_price is not None else self.average_price
        return abs(self.quantity) * price

    @property
    def unrealised_pnl_inr(self) -> float:
        if self.quantity == 0 or self.last_price is None:
            return 0.0
        return round((self.last_price - self.average_price) * self.quantity, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "product": str(self.product),
            "instrument": str(self.instrument),
            "quantity": self.quantity,
            "average_price": round(self.average_price, 2),
            "last_price": self.last_price,
            "realised_pnl_inr": round(self.realised_pnl_inr, 2),
            "unrealised_pnl_inr": self.unrealised_pnl_inr,
            "charges_inr": round(self.charges_inr, 2),
        }


@dataclass
class Funds:
    """Cash and what it is currently supporting."""

    starting_cash_inr: float
    cash_inr: float
    # Margin set aside for intraday and F&O positions, released when they close.
    blocked_margin_inr: float = 0.0
    realised_pnl_inr: float = 0.0
    charges_inr: float = 0.0

    @property
    def available_inr(self) -> float:
        return round(self.cash_inr - self.blocked_margin_inr, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "starting_cash_inr": round(self.starting_cash_inr, 2),
            "cash_inr": round(self.cash_inr, 2),
            "blocked_margin_inr": round(self.blocked_margin_inr, 2),
            "available_inr": self.available_inr,
            "realised_pnl_inr": round(self.realised_pnl_inr, 2),
            "charges_inr": round(self.charges_inr, 2),
        }


@runtime_checkable
class ExecutionBroker(Protocol):
    """What the framework needs from anything that can place an order.

    Deliberately small. A live adapter has to implement exactly this, so every
    method here is one whose live behaviour the simulator can be honest about.
    Anything a simulator could only pretend to know - queue position, partial
    fill sequencing, exchange-side order modification - is left out rather
    than faked.
    """

    def place(self, order: Order) -> Order:
        """Accept or reject an order. Returns the same order, status set."""
        ...

    def cancel(self, order_id: str) -> bool:
        """Cancel a resting order. False if it is unknown or already done."""
        ...

    def orders(self, *, open_only: bool = False) -> list[Order]:
        ...

    def positions(self, *, include_flat: bool = False) -> list[Position]:
        ...

    def funds(self) -> Funds:
        ...
