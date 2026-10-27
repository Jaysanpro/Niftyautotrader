"""Walk-forward evaluation of the desk's decisions.

The pipeline answers "what would you do today". This walks it across a range
of past sessions and scores what it said, so the question becomes "what would
this have done", which is the only version worth money.

Two rules make the result honest, and both are easy to get wrong:

* a decision dated *d* is taken with data up to the close on *d* (the as-of
  plumbing does that), and settled from the **first close after** *d*. A
  decision never scores on a bar it could have seen.
* a trade only opens if price is actually inside the entry band the desk
  specified. The desk's own note says chasing outside it changes the
  reward-risk the approval rested on, so filling there would be scoring a
  trade the desk did not approve.

What this is not: the news and sentiment feeds are RSS, and RSS has no
archive. A run dated far enough back reads no news at all, so what is being
measured there is the price, fundamentals and risk engine, not the full
agent. :func:`coverage_warnings` says so in the result rather than leaving it
to be discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tradingagents_in.dataflows.calendar_in import is_trading_day
from tradingagents_in.models.enums import Action
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import parse_date

log = get_logger(__name__)

# Exit reasons, kept as plain strings so a result serialises without a codec.
EXIT_TARGET = "target"
EXIT_STOP = "stop"
EXIT_HORIZON = "horizon"
NEVER_ENTERED = "never_entered"


@dataclass
class SettledTrade:
    """One decision, carried to an exit."""

    symbol: str
    decided_on: date
    action: str
    planned_entry_low: float
    planned_entry_high: float
    stop_loss: float
    first_target: float | None
    quantity: int
    risk_inr: float
    charges_inr: float = 0.0

    entered_on: date | None = None
    entry_price: float | None = None
    exited_on: date | None = None
    exit_price: float | None = None
    exit_reason: str = NEVER_ENTERED
    pnl_inr: float = 0.0
    r_multiple: float | None = None

    @property
    def was_taken(self) -> bool:
        return self.entry_price is not None

    @property
    def won(self) -> bool:
        return self.was_taken and self.pnl_inr > 0


@dataclass
class WalkForwardResult:
    trades: list[SettledTrade] = field(default_factory=list)
    sessions: int = 0
    decisions: int = 0
    approved: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- summary
    @property
    def taken(self) -> list[SettledTrade]:
        return [t for t in self.trades if t.was_taken]

    @property
    def hit_rate(self) -> float | None:
        taken = self.taken
        if not taken:
            return None
        return sum(1 for t in taken if t.won) / len(taken)

    @property
    def total_pnl_inr(self) -> float:
        return round(sum(t.pnl_inr for t in self.taken), 2)

    @property
    def avg_r(self) -> float | None:
        rs = [t.r_multiple for t in self.taken if t.r_multiple is not None]
        return round(sum(rs) / len(rs), 3) if rs else None

    @property
    def expectancy_r(self) -> float | None:
        """Average R per *decision taken* - the number that compounds."""
        return self.avg_r

    @property
    def equity_curve(self) -> list[tuple[date, float]]:
        """Cumulative realised P&L, ordered by exit."""
        closed = sorted(
            (t for t in self.taken if t.exited_on), key=lambda t: t.exited_on  # type: ignore[arg-type,return-value]
        )
        running = 0.0
        out: list[tuple[date, float]] = []
        for t in closed:
            running += t.pnl_inr
            out.append((t.exited_on, round(running, 2)))  # type: ignore[arg-type]
        return out

    @property
    def max_drawdown_inr(self) -> float:
        peak = 0.0
        worst = 0.0
        running = 0.0
        for _, running in self.equity_curve:
            peak = max(peak, running)
            worst = min(worst, running - peak)
        return round(worst, 2)

    def summary(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "decisions": self.decisions,
            "approved": self.approved,
            "taken": len(self.taken),
            "never_entered": sum(1 for t in self.trades if not t.was_taken),
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 3),
            "total_pnl_inr": self.total_pnl_inr,
            "avg_r": self.avg_r,
            "max_drawdown_inr": self.max_drawdown_inr,
            "exits": {
                reason: sum(1 for t in self.taken if t.exit_reason == reason)
                for reason in (EXIT_TARGET, EXIT_STOP, EXIT_HORIZON)
            },
            "warnings": self.warnings,
            "errors": self.errors[:5],
        }


# --------------------------------------------------------------- settlement


def _bars_after(prices: pd.DataFrame, day: date) -> pd.DataFrame:
    """Daily bars strictly after ``day``.

    Strictly: including the decision day's own bar is the classic off-by-one
    that lets a backtest buy at a price it already knew closed higher.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(prices.index)
    mask = idx.date > day
    return prices.loc[mask]


def settle(
    decision: Any,
    prices: pd.DataFrame,
    *,
    max_hold_days: int = 10,
    require_entry_band: bool = True,
) -> SettledTrade | None:
    """Carry one approved decision forward to an exit.

    Returns ``None`` when the decision was not a tradable position in the
    first place (a HOLD, or an approval with no proposal behind it).
    """
    proposal = getattr(decision, "proposal", None)
    if proposal is None or proposal.entry_zone is None or proposal.stop_loss is None:
        return None
    action = proposal.action
    if action == Action.HOLD or not proposal.quantity:
        return None

    decided_on = parse_date(decision.as_of.split()[0]) if decision.as_of else None
    if decided_on is None:
        return None

    targets = sorted(
        proposal.targets, key=lambda t: t.price, reverse=(action != Action.BUY)
    )
    trade = SettledTrade(
        symbol=decision.symbol,
        decided_on=decided_on,
        action=str(action),
        planned_entry_low=proposal.entry_zone.low,
        planned_entry_high=proposal.entry_zone.high,
        stop_loss=proposal.stop_loss,
        first_target=targets[0].price if targets else None,
        quantity=int(proposal.quantity),
        risk_inr=float(proposal.risk_amount_inr or 0.0),
        charges_inr=float(getattr(proposal.costs, "total_inr", 0.0) or 0.0),
    )

    future = _bars_after(prices, decided_on)
    if future.empty:
        return trade  # nothing to settle against; stays never_entered

    long = action == Action.BUY
    sign = 1.0 if long else -1.0

    # ---- entry: the first session whose close sits inside the planned band.
    entry_idx = None
    for pos in range(min(len(future), max_hold_days)):
        close = float(future["close"].iloc[pos])
        inside = trade.planned_entry_low <= close <= trade.planned_entry_high
        if inside or not require_entry_band:
            entry_idx = pos
            trade.entry_price = close
            trade.entered_on = pd.to_datetime(future.index[pos]).date()
            break
    if entry_idx is None:
        return trade  # price never came back to the band; the desk sat it out

    # ---- exit: stop first, then target, else the horizon.
    # Stop is checked before target within a bar. Both can be touched on one
    # bar and the daily series cannot say which came first, so the pessimistic
    # reading is the only defensible one.
    remaining = future.iloc[entry_idx + 1 : entry_idx + 1 + max_hold_days]
    for pos in range(len(remaining)):
        bar = remaining.iloc[pos]
        day = pd.to_datetime(remaining.index[pos]).date()
        low, high = float(bar["low"]), float(bar["high"])

        hit_stop = low <= trade.stop_loss if long else high >= trade.stop_loss
        if hit_stop:
            trade.exited_on, trade.exit_price, trade.exit_reason = (
                day, trade.stop_loss, EXIT_STOP,
            )
            break
        if trade.first_target is not None:
            hit_target = high >= trade.first_target if long else low <= trade.first_target
            if hit_target:
                trade.exited_on, trade.exit_price, trade.exit_reason = (
                    day, trade.first_target, EXIT_TARGET,
                )
                break
    else:
        if len(remaining):
            trade.exited_on = pd.to_datetime(remaining.index[-1]).date()
            trade.exit_price = float(remaining["close"].iloc[-1])
            trade.exit_reason = EXIT_HORIZON

    if trade.exit_price is None:  # entered on the last available bar
        trade.exit_reason = EXIT_HORIZON
        trade.exited_on = trade.entered_on
        trade.exit_price = trade.entry_price

    gross = (trade.exit_price - trade.entry_price) * trade.quantity * sign
    trade.pnl_inr = round(gross - trade.charges_inr, 2)

    risk_per_share = abs(trade.entry_price - trade.stop_loss)
    if risk_per_share > 0:
        # R is measured net of charges too: a 1R winner that pays 0.3R in
        # costs is not a 1R winner.
        trade.r_multiple = round(trade.pnl_inr / (risk_per_share * trade.quantity), 3)
    return trade


# ------------------------------------------------------------- the main loop


def trading_sessions(start: date, end: date, step_days: int = 1) -> list[date]:
    days: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            days.append(day)
        day += timedelta(days=step_days)
    return days


def walk_forward(
    app: Any,
    symbols: list[str],
    start: str | date,
    end: str | date,
    *,
    max_hold_days: int = 10,
    step_days: int = 1,
    require_entry_band: bool = True,
    prices: dict[str, pd.DataFrame] | None = None,
) -> WalkForwardResult:
    """Decide every session in the range, then settle what was approved.

    ``prices`` supplies the settlement series per symbol. Left out, it is
    fetched once per symbol for the whole window - once, because refetching
    per session would be both slow and a good way to get rate-limited into a
    run with no data.
    """
    start, end = parse_date(start), parse_date(end)
    result = WalkForwardResult()
    sessions = trading_sessions(start, end, step_days)
    result.sessions = len(sessions)
    if not sessions:
        result.errors.append(f"No trading sessions between {start} and {end}.")
        return result

    settlement = dict(prices or {})
    if prices is None:
        from tradingagents_in.dataflows import market_data

        horizon_end = end + timedelta(days=max_hold_days * 2 + 10)
        span = (horizon_end - start).days + 400
        for symbol in symbols:
            try:
                settlement[symbol] = market_data.fetch_ohlcv(
                    symbol, "1d", lookback_days=span, end=horizon_end, config=app.config
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"settlement prices for {symbol}: {exc}")

    for symbol in symbols:
        frame = settlement.get(symbol)
        if frame is None or frame.empty:
            result.warnings.append(
                f"{symbol}: no settlement prices, so its decisions are unscored."
            )

    for day in sessions:
        for symbol in symbols:
            try:
                decision = app.analyse(symbol, as_of=day, save=False)
            except Exception as exc:  # noqa: BLE001 - one bad session must not stop the run
                result.errors.append(f"{symbol} on {day}: {exc}")
                continue

            result.decisions += 1
            if not decision.approved:
                continue
            result.approved += 1

            frame = settlement.get(symbol)
            if frame is None or frame.empty:
                continue
            trade = settle(
                decision, frame,
                max_hold_days=max_hold_days,
                require_entry_band=require_entry_band,
            )
            if trade is not None:
                result.trades.append(trade)

    result.warnings.extend(coverage_warnings(start))
    return result


def coverage_warnings(start: date) -> list[str]:
    """Say plainly where the inputs thin out, rather than letting it surprise."""
    out: list[str] = []
    age_days = (date.today() - start).days
    if age_days > 30:
        out.append(
            "News and sentiment come from RSS feeds, which carry no archive: for "
            "sessions more than about a month back those sources are empty, so "
            "this measures the price, fundamentals and risk engine rather than "
            "the full agent."
        )
    return out


def render_summary(result: WalkForwardResult) -> str:
    """A compact text report, in the CLI's register."""
    s = result.summary()
    lines = [
        f"sessions {s['sessions']} · decisions {s['decisions']} · "
        f"approved {s['approved']} · taken {s['taken']} "
        f"(never entered {s['never_entered']})",
    ]
    if s["taken"]:
        hit = "n/a" if s["hit_rate"] is None else f"{s['hit_rate'] * 100:.1f}%"
        lines += [
            f"hit rate {hit} · avg R {s['avg_r']} · "
            f"net P&L Rs {s['total_pnl_inr']:,.2f} · "
            f"max drawdown Rs {s['max_drawdown_inr']:,.2f}",
            f"exits: {s['exits'][EXIT_TARGET]} target · {s['exits'][EXIT_STOP]} stop "
            f"· {s['exits'][EXIT_HORIZON]} horizon",
        ]
    else:
        lines.append("No trade was taken, so there is nothing to score.")
    for warning in s["warnings"]:
        lines.append(f"! {warning}")
    for err in s["errors"]:
        lines.append(f"error: {err}")
    return "\n".join(lines)
