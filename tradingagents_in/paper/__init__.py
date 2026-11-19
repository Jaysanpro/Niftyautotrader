"""Simulated execution: the half the framework was missing.

The desk decides; this places, fills, holds and closes. It exists so that
decisions can be acted on and scored without money at risk, and so the
execution interface is defined and exercised before a live broker is wired to
it.

``ExecutionBroker`` is the contract. ``PaperBroker`` is the only
implementation today; a Kite adapter would implement the same protocol.
"""

from tradingagents_in.paper.broker import (
    DEFAULT_MIS_LEVERAGE,
    DEFAULT_SLIPPAGE_BPS,
    PaperBroker,
    Trade,
)
from tradingagents_in.paper.models import (
    ExecutionBroker,
    Funds,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from tradingagents_in.paper.orders import orders_for_decision

__all__ = [
    "DEFAULT_MIS_LEVERAGE",
    "DEFAULT_SLIPPAGE_BPS",
    "ExecutionBroker",
    "Funds",
    "Order",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "Position",
    "Trade",
    "orders_for_decision",
]
