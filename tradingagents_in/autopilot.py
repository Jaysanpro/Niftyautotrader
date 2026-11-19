"""One autonomous pass: screen, place, manage, report.

This is the loop that removes the symbol from the question. Nobody names a
stock; the screener ranks a universe and the book takes what fits, the orders
go to a broker, and the next pass marks what is open, lets stops do their
work and squares off intraday before the close.

It is written as **one pass, not a daemon**. A pass is idempotent-ish, holds
no state of its own beyond the broker's saved book, and ends. Scheduling is
somebody else's job - cron, a Railway/Cloud Run job, or a person running the
command - because a process that must stay alive to be correct is a process
that loses the book when it dies, and because a loop you can run by hand is
one you can inspect before you trust it.

Nothing here reaches a live broker. The only execution path is the simulator
in ``tradingagents_in.paper``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tradingagents_in.models.enums import ProductType
from tradingagents_in.paper import Order, PaperBroker, orders_for_decision
from tradingagents_in.screen import screen
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PassResult:
    universe: str
    as_of: str | None = None
    scanned: int = 0
    selected: list[str] = field(default_factory=list)
    placed: list[Order] = field(default_factory=list)
    filled: list[Order] = field(default_factory=list)
    squared_off: list[Order] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    equity_inr: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "as_of": self.as_of,
            "scanned": self.scanned,
            "selected": self.selected,
            "orders_placed": len(self.placed),
            "orders_filled": len(self.filled),
            "squared_off": len(self.squared_off),
            "skipped": self.skipped,
            "equity_inr": self.equity_inr,
            "errors": self.errors[:5],
        }


def _price_of(decision: Any) -> float | None:
    """The reference price the desk quoted, for marking a fresh name."""
    proposal = getattr(decision, "proposal", None)
    zone = getattr(proposal, "entry_zone", None) if proposal else None
    if zone is None:
        return None
    if zone.reference_price:
        return float(zone.reference_price)
    return round((zone.low + zone.high) / 2, 2)


def run_pass(
    app: Any,
    broker: PaperBroker,
    universe: str,
    *,
    as_of: str | date | None = None,
    prices: dict[str, float] | None = None,
    max_new_positions: int = 3,
    square_off: bool = False,
    rank_by: str = "reward_risk",
) -> PassResult:
    """Run one pass of the loop.

    ``prices`` are the marks for this pass - what the book is valued and
    triggered against. Anything held but not priced here keeps its last mark,
    which is stated rather than hidden: a stop cannot fire on a price nobody
    supplied.

    ``max_new_positions`` bounds how much new risk one pass may take on. The
    screener already respects the portfolio caps, but those bound the *book*;
    this bounds the *rate*, so a single bad day cannot fill the book in one
    go.
    """
    result = PassResult(universe=universe, as_of=str(as_of) if as_of else None)

    # 1. Mark first. Existing positions and resting stops get their chance
    #    before any new money is committed - an exit beats an entry.
    for symbol, price in (prices or {}).items():
        try:
            result.filled.extend(broker.mark(symbol, price))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"mark {symbol}: {exc}")

    # 2. Square off intraday if this pass is the one near the close.
    if square_off:
        try:
            result.squared_off = broker.square_off_intraday(prices)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"square-off: {exc}")

    # 3. Screen for what to do next.
    try:
        screened = screen(app, universe, as_of=as_of, rank_by=rank_by)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"screen: {exc}")
        result.equity_inr = broker.equity_inr()
        return result

    result.scanned = screened.scanned
    result.errors.extend(screened.errors)

    held = {p.symbol for p in broker.positions()}
    resting = {o.symbol for o in broker.orders(open_only=True)}
    taken = 0

    for candidate in screened.selected:
        if taken >= max_new_positions:
            result.skipped[candidate.symbol] = "pass limit reached"
            continue
        # Never stack a second entry on a name the book already has an
        # opinion about: the screener sized against a flat book for that name.
        if candidate.symbol in held:
            result.skipped[candidate.symbol] = "already held"
            continue
        if candidate.symbol in resting:
            result.skipped[candidate.symbol] = "order already resting"
            continue

        decision = getattr(candidate, "decision", None)
        if decision is None:
            # The screen reports candidates, not decisions; re-decide the
            # name so the order carries this pass's numbers rather than a
            # summary of them.
            try:
                decision = app.analyse(candidate.symbol, as_of=as_of, save=False)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{candidate.symbol}: {exc}")
                continue

        orders = orders_for_decision(decision)
        if not orders:
            result.skipped[candidate.symbol] = "no tradable order"
            continue

        for order in orders:
            placed = broker.place(order)
            result.placed.append(placed)
            if placed.status.value == "REJECTED":
                result.skipped[candidate.symbol] = placed.message

        # Give the entry a chance against a known price straight away, so a
        # pass that could fill does, rather than waiting for the next one.
        mark = (prices or {}).get(candidate.symbol) or _price_of(decision)
        if mark:
            try:
                result.filled.extend(broker.mark(candidate.symbol, mark))
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"mark {candidate.symbol}: {exc}")

        result.selected.append(candidate.symbol)
        taken += 1

    result.equity_inr = broker.equity_inr()
    return result


def render(result: PassResult, broker: PaperBroker) -> str:
    """What one pass did, in the CLI's register."""
    s = result.summary()
    lines = [
        f"{s['universe']} · {s['scanned']} scanned · "
        f"{s['orders_placed']} orders placed · {s['orders_filled']} filled"
        + (f" · as of {s['as_of']}" if s["as_of"] else ""),
    ]
    if result.squared_off:
        lines.append(f"squared off {len(result.squared_off)} intraday position(s)")

    book = broker.summary()
    lines += [
        f"equity Rs {book['equity_inr']:,.2f} ({book['return_pct']:+.2f}%) · "
        f"cash Rs {book['funds']['cash_inr']:,.2f} · "
        f"{book['open_positions']} open · {book['open_orders']} resting",
    ]
    positions = broker.positions()
    if positions:
        lines.append("")
        lines.append(f"   {'symbol':<14}{'prod':<5}{'qty':>7}{'avg':>11}{'last':>11}{'P&L':>12}")
        for p in positions:
            last = p.last_price if p.last_price is not None else p.average_price
            lines.append(
                f"   {p.symbol[:13]:<14}{str(p.product):<5}{p.quantity:>7}"
                f"{p.average_price:>11,.2f}{last:>11,.2f}{p.unrealised_pnl_inr:>12,.2f}"
            )
    if broker.trades:
        lines.append("")
        lines.append(
            f"closed {book['closed_trades']} trade(s) · "
            f"hit rate {('n/a' if book['hit_rate'] is None else f'{book['hit_rate'] * 100:.0f}%')} · "
            f"net Rs {book['net_pnl_inr']:,.2f} after Rs {book['charges_inr']:,.2f} charges"
        )
    for symbol, why in list(result.skipped.items())[:6]:
        lines.append(f"  skipped {symbol}: {why}")
    for err in s["errors"]:
        lines.append(f"error: {err}")
    return "\n".join(lines)
