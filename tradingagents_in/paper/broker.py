"""A simulated broker: orders, fills, positions, cash and charges.

Driven by price, not by a clock. Call :meth:`PaperBroker.mark` with a price
and resting orders are reconsidered against it; nothing fills on its own.
That makes a run reproducible, lets the same object serve a live loop (mark
on each quote) and a backtest (mark on each bar), and keeps the awkward
question - *would this actually have filled?* - explicit instead of assumed.

Three things it refuses to pretend about, because a paper broker that flatters
is worse than none:

* **Charges.** Round-trip charges are taken out of realised P&L using the
  desk's own cost model. On small-R Indian trades they are the difference
  between a winner and a loser.
* **Slippage.** A market or stop-triggered fill is worse than the price that
  triggered it. Only a limit order fills at its own price, because that is
  the one case where the price was yours to name.
* **Cash.** Delivery needs the full amount; intraday is leveraged but still
  finite. An order the book cannot pay for is rejected, not quietly accepted.

What it does not simulate: queue position, partial fills, exchange-side
modification, or an order book. A simulator cannot know those, so they are
absent rather than invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents_in.models.enums import Action, Instrument, ProductType
from tradingagents_in.paper.models import (
    ExecutionBroker,
    Funds,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from tradingagents_in.risk.costs import estimate_costs
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import now_ist

log = get_logger(__name__)

# Intraday products are leveraged. 5x is a conservative stand-in for what
# Indian brokers offer on liquid cash intraday; it is a setting, not a law.
DEFAULT_MIS_LEVERAGE = 5.0
# A market fill is not the price you saw. Charged against the order either way.
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class Trade:
    """One completed round trip, net of charges."""

    symbol: str
    product: ProductType
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl_inr: float
    charges_inr: float
    net_pnl_inr: float
    opened_at: datetime
    closed_at: datetime
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "product": str(self.product),
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "gross_pnl_inr": round(self.gross_pnl_inr, 2),
            "charges_inr": round(self.charges_inr, 2),
            "net_pnl_inr": round(self.net_pnl_inr, 2),
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "tag": self.tag,
        }


class PaperBroker(ExecutionBroker):
    """An in-memory broker that can be written to disk and read back."""

    def __init__(
        self,
        starting_cash_inr: float = 1_000_000.0,
        *,
        config: dict[str, Any] | None = None,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        mis_leverage: float = DEFAULT_MIS_LEVERAGE,
        state_path: str | Path | None = None,
    ):
        self.config = config or {}
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.mis_leverage = max(1.0, float(mis_leverage))
        self.state_path = Path(state_path) if state_path else None

        self._funds = Funds(
            starting_cash_inr=float(starting_cash_inr), cash_inr=float(starting_cash_inr)
        )
        self._orders: dict[str, Order] = {}
        self._positions: dict[tuple[str, ProductType], Position] = {}
        self._opened_at: dict[tuple[str, ProductType], datetime] = {}
        self._trades: list[Trade] = []
        self._last_price: dict[str, float] = {}

    # ------------------------------------------------------------- queries

    def funds(self) -> Funds:
        return self._funds

    def orders(self, *, open_only: bool = False) -> list[Order]:
        out = list(self._orders.values())
        if open_only:
            out = [o for o in out if o.is_open]
        return sorted(out, key=lambda o: o.placed_at)

    def positions(self, *, include_flat: bool = False) -> list[Position]:
        out = list(self._positions.values())
        if not include_flat:
            out = [p for p in out if not p.is_flat]
        return sorted(out, key=lambda p: p.symbol)

    def position(self, symbol: str, product: ProductType = ProductType.CNC) -> Position | None:
        return self._positions.get((symbol.upper(), product))

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def equity_inr(self) -> float:
        """Cash plus what the open positions are worth.

        The two products contribute differently, and conflating them
        understates a delivery book by its whole cost basis. Buying delivery
        moves cash into stock, so the holding's market value is what came
        back. Intraday blocks margin instead of spending cash, so only the
        move counts.
        """
        total = self._funds.cash_inr
        for position in self.positions():
            price = position.last_price if position.last_price is not None \
                else position.average_price
            if position.product == ProductType.CNC:
                total += position.quantity * price
            else:
                total += position.unrealised_pnl_inr
        return round(total, 2)

    # ------------------------------------------------------------- placing

    def place(self, order: Order) -> Order:
        order.symbol = order.symbol.upper()
        self._orders[order.order_id] = order

        problem = self._reject_reason(order)
        if problem:
            order.status = OrderStatus.REJECTED
            order.message = problem
            log.info("paper: rejected %s %s - %s", order.side, order.symbol, problem)
            self._write_event("order_rejected", order.to_dict())
            return order

        self._write_event("order_placed", order.to_dict())

        # A market order needs no further price to act on, if one is known.
        if order.order_type == OrderType.MARKET:
            price = self._last_price.get(order.symbol)
            if price is not None:
                self._fill(order, self._slipped(price, order.side))
            # Without a known price it rests until the first mark.
        return order

    def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.is_open:
            return False
        order.status = OrderStatus.CANCELLED
        order.message = "cancelled"
        self._write_event("order_cancelled", order.to_dict())
        return True

    def _reject_reason(self, order: Order) -> str:
        if order.quantity <= 0:
            return "quantity must be positive"
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            return "a limit order needs a limit price"
        if order.order_type == OrderType.SL and not order.trigger_price:
            return "a stop-loss order needs a trigger price"

        # Reducing or closing an existing position frees cash rather than
        # needing it, so it is never refused for funds.
        if self._is_reducing(order):
            return ""

        reference = (
            order.limit_price
            or order.trigger_price
            or self._last_price.get(order.symbol)
        )
        if reference is None:
            # Nothing to price the requirement against; let it rest and be
            # checked when a price arrives.
            return ""
        needed = self._requirement(order.quantity, reference, order.product)
        if needed > self._funds.available_inr + 1e-6:
            kind = "delivery needs the full amount" if order.product == ProductType.CNC \
                else "margin"
            return (
                f"insufficient funds: {kind} Rs {needed:,.2f}, "
                f"available Rs {self._funds.available_inr:,.2f}"
            )
        return ""

    def _is_reducing(self, order: Order) -> bool:
        held = self._positions.get((order.symbol, order.product))
        if held is None or held.is_flat:
            return False
        return (held.quantity > 0) != (order.side == Action.BUY)

    def _requirement(self, quantity: int, price: float, product: ProductType) -> float:
        notional = quantity * price
        if product == ProductType.CNC:
            return round(notional, 2)
        return round(notional / self.mis_leverage, 2)

    # -------------------------------------------------------------- prices

    def mark(self, symbol: str, price: float) -> list[Order]:
        """Record a price and execute whatever it makes executable.

        Returns the orders that filled on this mark, so a caller can react
        without diffing state.
        """
        symbol = symbol.upper()
        price = float(price)
        self._last_price[symbol] = price

        for position in self._positions.values():
            if position.symbol == symbol:
                position.last_price = price

        filled: list[Order] = []
        # Sorted so the outcome does not depend on dict ordering.
        for order in sorted(
            (o for o in self._orders.values() if o.is_open and o.symbol == symbol),
            key=lambda o: o.placed_at,
        ):
            fill_price = self._executable_at(order, price)
            if fill_price is None:
                continue
            problem = self._reject_reason(order)
            if problem:
                order.status = OrderStatus.REJECTED
                order.message = problem
                self._write_event("order_rejected", order.to_dict())
                continue
            self._fill(order, fill_price)
            filled.append(order)
        self._save_snapshot()
        return filled

    def mark_many(self, prices: dict[str, float]) -> list[Order]:
        filled: list[Order] = []
        for symbol, price in prices.items():
            filled.extend(self.mark(symbol, price))
        return filled

    def _executable_at(self, order: Order, price: float) -> float | None:
        """The fill price for this order at this market price, or None."""
        if order.order_type == OrderType.MARKET:
            return self._slipped(price, order.side)

        if order.order_type == OrderType.LIMIT:
            # A limit fills at its own price: that is what naming it buys you.
            if order.side == Action.BUY and price <= order.limit_price:
                return float(order.limit_price)
            if order.side == Action.SELL and price >= order.limit_price:
                return float(order.limit_price)
            return None

        # Stop-loss: a sell triggers at or below, a buy-to-cover at or above.
        # Once triggered it is a market order, so it slips.
        if order.side == Action.SELL and price <= order.trigger_price:
            return self._slipped(price, order.side)
        if order.side == Action.BUY and price >= order.trigger_price:
            return self._slipped(price, order.side)
        return None

    def _slipped(self, price: float, side: Action) -> float:
        """Market fills go against you, never for you."""
        drift = price * self.slippage_bps / 10_000.0
        return round(price + drift if side == Action.BUY else price - drift, 2)

    # --------------------------------------------------------------- fills

    def _fill(self, order: Order, price: float) -> None:
        order.status = OrderStatus.FILLED
        order.fill_price = round(price, 2)
        order.filled_at = now_ist()

        key = (order.symbol, order.product)
        position = self._positions.get(key)
        if position is None:
            position = Position(
                symbol=order.symbol, product=order.product, instrument=order.instrument
            )
            self._positions[key] = position
        position.last_price = price

        before = position.quantity
        delta = order.signed_quantity

        if before == 0 or (before > 0) == (delta > 0):
            self._open_or_add(position, order, price, delta, key)
        else:
            self._reduce_or_close(position, order, price, delta, key)

        self._write_event("order_filled", order.to_dict())
        self._save_snapshot()

    def _open_or_add(
        self, position: Position, order: Order, price: float, delta: int,
        key: tuple[str, ProductType],
    ) -> None:
        total = position.quantity + delta
        # Weighted average, so adding to a position does not reset its basis.
        position.average_price = (
            (position.average_price * abs(position.quantity) + price * abs(delta))
            / abs(total)
        )
        position.quantity = total
        self._opened_at.setdefault(key, order.filled_at or now_ist())

        requirement = self._requirement(abs(delta), price, order.product)
        if order.product == ProductType.CNC:
            self._funds.cash_inr -= requirement
        else:
            self._funds.blocked_margin_inr += requirement

    def _reduce_or_close(
        self, position: Position, order: Order, price: float, delta: int,
        key: tuple[str, ProductType],
    ) -> None:
        closing = min(abs(delta), abs(position.quantity))
        direction = 1 if position.quantity > 0 else -1
        gross = (price - position.average_price) * closing * direction

        charges = estimate_costs(
            entry_price=position.average_price,
            exit_price=price,
            quantity=closing,
            instrument=position.instrument,
            product_type=position.product,
            action=Action.BUY if direction > 0 else Action.SELL,
            config=self.config,
        )
        charge_inr = float(charges.total_inr or 0.0)
        net = gross - charge_inr

        position.realised_pnl_inr += net
        position.charges_inr += charge_inr
        self._funds.realised_pnl_inr += net
        self._funds.charges_inr += charge_inr

        released = self._requirement(closing, position.average_price, position.product)
        if position.product == ProductType.CNC:
            # The basis comes back, and the trade's result lands on cash.
            self._funds.cash_inr += released + net
        else:
            self._funds.blocked_margin_inr = max(
                0.0, self._funds.blocked_margin_inr - released
            )
            self._funds.cash_inr += net

        self._trades.append(Trade(
            symbol=position.symbol, product=position.product, quantity=closing,
            entry_price=position.average_price, exit_price=price,
            gross_pnl_inr=round(gross, 2), charges_inr=round(charge_inr, 2),
            net_pnl_inr=round(net, 2),
            opened_at=self._opened_at.get(key, order.placed_at),
            closed_at=order.filled_at or now_ist(), tag=order.tag,
        ))

        remaining = position.quantity + delta
        position.quantity = remaining
        if remaining == 0:
            position.average_price = 0.0
            self._opened_at.pop(key, None)
        elif (remaining > 0) != (direction > 0):
            # Flipped through flat: the overshoot opens the other way.
            position.average_price = price
            self._opened_at[key] = order.filled_at or now_ist()

    # ------------------------------------------------------------ lifecycle

    def square_off_intraday(self, prices: dict[str, float] | None = None) -> list[Order]:
        """Close every MIS position, as a broker does near the close.

        Intraday positions are not optional to close: a real broker squares
        them off and charges for the privilege. A paper run that carries them
        overnight is reporting a trade that could not have happened.
        """
        prices = {k.upper(): v for k, v in (prices or {}).items()}
        closed: list[Order] = []
        for position in list(self._positions.values()):
            if position.product != ProductType.MIS or position.is_flat:
                continue
            price = prices.get(position.symbol) or position.last_price \
                or self._last_price.get(position.symbol)
            if price is None:
                log.warning(
                    "paper: cannot square off %s - no price", position.symbol
                )
                continue
            order = Order(
                symbol=position.symbol,
                side=Action.SELL if position.is_long else Action.BUY,
                quantity=abs(position.quantity),
                product=ProductType.MIS,
                instrument=position.instrument,
                order_type=OrderType.MARKET,
                tag="square-off",
            )
            self._orders[order.order_id] = order
            self._fill(order, self._slipped(price, order.side))
            closed.append(order)
        return closed

    def cancel_all(self) -> int:
        return sum(1 for o in self.orders(open_only=True) if self.cancel(o.order_id))

    # ------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        trades = self._trades
        wins = [t for t in trades if t.net_pnl_inr > 0]
        return {
            "equity_inr": self.equity_inr(),
            "funds": self._funds.to_dict(),
            "open_positions": len(self.positions()),
            "open_orders": len(self.orders(open_only=True)),
            "closed_trades": len(trades),
            "hit_rate": round(len(wins) / len(trades), 3) if trades else None,
            "gross_pnl_inr": round(sum(t.gross_pnl_inr for t in trades), 2),
            "charges_inr": round(sum(t.charges_inr for t in trades), 2),
            "net_pnl_inr": round(sum(t.net_pnl_inr for t in trades), 2),
            "return_pct": round(
                (self.equity_inr() - self._funds.starting_cash_inr)
                / self._funds.starting_cash_inr * 100, 3
            ),
        }

    # ---------------------------------------------------------- persistence

    def _write_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Append-only log beside the snapshot: the snapshot says where the
        book stands, this says how it got there."""
        if self.state_path is None:
            return
        try:
            path = self.state_path.with_suffix(".events.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"at": now_ist().isoformat(), "event": kind, **payload}, default=str
                ) + "\n")
        except OSError as exc:  # a full disk must not lose the fill itself
            log.warning("paper: could not write event log: %s", exc)

    def _save_snapshot(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        except OSError as exc:
            log.warning("paper: could not write state: %s", exc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "funds": self._funds.to_dict(),
            "slippage_bps": self.slippage_bps,
            "mis_leverage": self.mis_leverage,
            "positions": [p.to_dict() for p in self.positions(include_flat=True)],
            "open_orders": [o.to_dict() for o in self.orders(open_only=True)],
            "trades": [t.to_dict() for t in self._trades],
            "summary": self.summary(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.state_path
        if target is None:
            raise ValueError("No state_path given, and none configured.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return target

    @classmethod
    def load(
        cls, path: str | Path, *, config: dict[str, Any] | None = None
    ) -> PaperBroker:
        """Restore a book. Open orders are restored; history is informational.

        Closed trades come back for reporting but are not replayed: the money
        they moved is already in the restored cash figure.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        funds = data.get("funds", {})
        broker = cls(
            starting_cash_inr=float(funds.get("starting_cash_inr", 1_000_000.0)),
            config=config,
            slippage_bps=float(data.get("slippage_bps", DEFAULT_SLIPPAGE_BPS)),
            mis_leverage=float(data.get("mis_leverage", DEFAULT_MIS_LEVERAGE)),
            state_path=path,
        )
        broker._funds.cash_inr = float(funds.get("cash_inr", broker._funds.cash_inr))
        broker._funds.blocked_margin_inr = float(funds.get("blocked_margin_inr", 0.0))
        broker._funds.realised_pnl_inr = float(funds.get("realised_pnl_inr", 0.0))
        broker._funds.charges_inr = float(funds.get("charges_inr", 0.0))

        for raw in data.get("positions", []):
            product = ProductType(raw.get("product", "CNC"))
            position = Position(
                symbol=raw["symbol"],
                product=product,
                instrument=Instrument(raw.get("instrument", "CASH")),
                quantity=int(raw.get("quantity", 0)),
                average_price=float(raw.get("average_price", 0.0)),
                realised_pnl_inr=float(raw.get("realised_pnl_inr", 0.0)),
                charges_inr=float(raw.get("charges_inr", 0.0)),
                last_price=raw.get("last_price"),
            )
            broker._positions[(position.symbol, product)] = position
            if position.last_price:
                broker._last_price[position.symbol] = float(position.last_price)

        for raw in data.get("open_orders", []):
            order = Order(
                symbol=raw["symbol"],
                side=Action(raw["side"]),
                quantity=int(raw["quantity"]),
                product=ProductType(raw.get("product", "CNC")),
                instrument=Instrument(raw.get("instrument", "CASH")),
                order_type=OrderType(raw.get("order_type", "MARKET")),
                limit_price=raw.get("limit_price"),
                trigger_price=raw.get("trigger_price"),
                tag=raw.get("tag", ""),
                order_id=raw["order_id"],
            )
            broker._orders[order.order_id] = order
        return broker
