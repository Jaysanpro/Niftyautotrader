"""The deterministic risk rulebook.

Every check here is machine-evaluated against data, never against an LLM's
opinion. The risk-manager agent is allowed to *explain* a breach and to be more
conservative than the rulebook; it is not allowed to overturn a BLOCKER. That
asymmetry is the whole point - a language model should not be the last line of
defence between a book and a position it cannot exit.

Checks fall into four groups:

1. **Coherence** - is the proposal internally valid at all (stop on the right
   side, a stop that exists, a reward-risk that survives charges)?
2. **Tradability** - India market plumbing: circuit bands, ASM/GSM, F&O ban,
   liquidity, expiry mechanics, product/segment legality.
3. **Position risk** - size, single-name concentration, stop distance.
4. **Portfolio risk** - sector concentration, aggregate heat, drawdown state,
   open-position count, overnight derivative exposure.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import (
    Action,
    Instrument,
    ProductType,
    Severity,
    SurveillanceFlag,
    TimeHorizon,
)
from tradingagents_in.models.risk import PortfolioSnapshot, RuleBreach
from tradingagents_in.models.trade import TradeProposal
from tradingagents_in.risk.costs import net_reward_risk
from tradingagents_in.utils.formatting import inr, inr_plain


def check_all(
    proposal: TradeProposal,
    bundle: MarketBundle,
    portfolio: PortfolioSnapshot,
    config: dict[str, Any],
) -> list[RuleBreach]:
    """Run the whole rulebook. Order is stable so reports diff cleanly."""
    if proposal.action == Action.HOLD:
        return check_portfolio_state(portfolio, config)

    breaches: list[RuleBreach] = []
    breaches += check_coherence(proposal, bundle, config)
    breaches += check_tradability(proposal, bundle, config)
    breaches += check_position_risk(proposal, bundle, portfolio, config)
    breaches += check_portfolio_risk(proposal, bundle, portfolio, config)
    breaches += check_portfolio_state(portfolio, config)
    return breaches


# ---------------------------------------------------------------- coherence
def check_coherence(
    proposal: TradeProposal, bundle: MarketBundle, config: dict[str, Any]
) -> list[RuleBreach]:
    out: list[RuleBreach] = []
    entry = proposal.entry_zone.mid if proposal.entry_zone else bundle.last_price

    if entry is None:
        out.append(
            RuleBreach(
                rule="entry_price_unknown",
                severity=Severity.BLOCKER,
                detail="No entry zone and no live price: the trade cannot be priced or sized.",
                remedy="Re-run when a price feed is available.",
            )
        )
        return out

    if proposal.stop_loss is None:
        out.append(
            RuleBreach(
                rule="missing_stop_loss",
                severity=Severity.BLOCKER,
                detail="A position without a stop has undefined risk and cannot be sized.",
                remedy="Set a stop at a level that invalidates the thesis.",
            )
        )
        return out

    long_side = proposal.action == Action.BUY
    if long_side and proposal.stop_loss >= entry:
        out.append(
            RuleBreach(
                rule="stop_on_wrong_side",
                severity=Severity.BLOCKER,
                detail="Long trade with the stop at or above entry.",
                observed=f"entry {entry:,.2f}, stop {proposal.stop_loss:,.2f}",
                remedy="Place the stop below the entry zone.",
            )
        )
    if proposal.action in (Action.SELL, Action.SHORT) and proposal.stop_loss <= entry:
        out.append(
            RuleBreach(
                rule="stop_on_wrong_side",
                severity=Severity.BLOCKER,
                detail="Short trade with the stop at or below entry.",
                observed=f"entry {entry:,.2f}, stop {proposal.stop_loss:,.2f}",
                remedy="Place the stop above the entry zone.",
            )
        )

    stop_pct = abs(entry - proposal.stop_loss) / entry * 100
    if stop_pct < 0.15:
        out.append(
            RuleBreach(
                rule="stop_too_tight",
                severity=Severity.WARNING,
                detail="Stop is inside normal bid-ask noise and will be taken out by "
                       "microstructure rather than by the thesis being wrong.",
                observed=f"{stop_pct:.2f}% from entry",
                limit="at least 0.15%, and normally at least 1 ATR",
            )
        )

    atr_pct = _atr_pct(bundle)
    if atr_pct and stop_pct < atr_pct * 0.5:
        out.append(
            RuleBreach(
                rule="stop_inside_atr",
                severity=Severity.WARNING,
                detail="Stop is under half the daily ATR, so ordinary volatility will hit "
                       "it before the idea has a chance to work.",
                observed=f"stop {stop_pct:.2f}% vs ATR {atr_pct:.2f}%",
                remedy="Widen the stop and cut the size, or wait for a tighter entry.",
            )
        )

    if not proposal.targets:
        out.append(
            RuleBreach(
                rule="missing_target",
                severity=Severity.WARNING,
                detail="No target set, so reward-risk cannot be evaluated.",
            )
        )
    else:
        rr, costs = net_reward_risk(
            entry_price=entry,
            stop_loss=proposal.stop_loss,
            target=proposal.targets[0].price,
            quantity=max(proposal.quantity or 1, 1),
            instrument=proposal.instrument,
            product_type=proposal.product_type,
            action=proposal.action,
            config=config,
        )
        minimum = float(config.get("min_reward_risk", 1.5))
        if rr is not None and rr < minimum:
            out.append(
                RuleBreach(
                    rule="reward_risk_below_minimum",
                    severity=Severity.WARNING if rr >= minimum * 0.7 else Severity.BLOCKER,
                    detail="Reward-risk after Indian transaction charges (STT, stamp duty, "
                           "exchange charges and GST) does not clear the desk minimum.",
                    observed=f"net R:R {rr:.2f} (round-trip cost {inr_plain(costs.total_inr)})",
                    limit=f"{minimum:.2f}",
                    remedy="Improve the entry, widen the first target, or skip the trade.",
                )
            )
    return out


# -------------------------------------------------------------- tradability
def check_tradability(
    proposal: TradeProposal, bundle: MarketBundle, config: dict[str, Any]
) -> list[RuleBreach]:
    out: list[RuleBreach] = []
    micro = bundle.micro
    is_derivative = proposal.instrument in (Instrument.FUTURES, Instrument.OPTIONS)

    if micro.is_circuit_locked:
        out.append(
            RuleBreach(
                rule="circuit_locked",
                severity=Severity.BLOCKER,
                detail="The scrip is locked at its circuit limit. There is no counterparty "
                       "on the exit side, so the stop-loss is theoretical.",
                observed=f"band {micro.circuit_band_pct}%",
                remedy="Wait for the band to release before taking a position.",
            )
        )

    if micro.surveillance != SurveillanceFlag.NONE and config.get("avoid_asm_gsm", True):
        is_gsm = micro.surveillance == SurveillanceFlag.GSM
        out.append(
            RuleBreach(
                rule="surveillance_framework",
                severity=Severity.BLOCKER if is_gsm else Severity.WARNING,
                detail=(
                    "GSM scrips can be restricted to periodic call auctions with 100% "
                    "margin and price caps."
                    if is_gsm else
                    "ASM raises margins and can force trade-for-trade settlement, which "
                    "removes intraday leverage and thins the book."
                ),
                observed=micro.surveillance.value,
                remedy="Trade only with full delivery intent and reduced size, or stand aside.",
            )
        )

    if is_derivative and micro.in_fno_ban and config.get("avoid_fno_ban_period", True):
        out.append(
            RuleBreach(
                rule="fno_ban_period",
                severity=Severity.BLOCKER,
                detail="Market-wide position limit exceeded: only position-reducing orders "
                       "are permitted, and new positions attract an exchange penalty.",
                remedy="Express the view in the cash segment or wait for the ban to lift.",
            )
        )

    if is_derivative and not config.get("allow_fno", True):
        out.append(
            RuleBreach(
                rule="fno_disabled",
                severity=Severity.BLOCKER,
                detail="Derivatives are disabled for this mandate (allow_fno=false).",
                remedy="Use the cash segment.",
            )
        )

    if is_derivative and not (bundle.ref.fno_eligible or bundle.ref.is_index):
        out.append(
            RuleBreach(
                rule="not_fno_eligible",
                severity=Severity.BLOCKER,
                detail=f"{bundle.ref.base_symbol} is not in the F&O segment, so futures and "
                       "options do not exist for it.",
                remedy="Use the cash segment.",
            )
        )

    if (
        proposal.action == Action.SHORT
        and proposal.instrument == Instrument.CASH
        and not config.get("allow_short_cash", False)
    ):
        out.append(
            RuleBreach(
                rule="cash_short_not_permitted",
                severity=Severity.BLOCKER,
                detail="Short selling in the cash segment must be squared off the same day; "
                       "carrying it forward causes a delivery default and auction penalty. "
                       "This mandate does not permit cash shorts.",
                remedy="Short the future, or buy a put instead.",
            )
        )

    if (
        proposal.action == Action.SHORT
        and proposal.instrument == Instrument.CASH
        and proposal.time_horizon != TimeHorizon.INTRADAY
    ):
        out.append(
            RuleBreach(
                rule="cash_short_overnight",
                severity=Severity.BLOCKER,
                detail="A cash-segment short cannot be held overnight in India.",
                remedy="Square off before 15:20, or use futures for a carry-forward short.",
            )
        )

    if (
        is_derivative
        and proposal.time_horizon != TimeHorizon.INTRADAY
        and not config.get("allow_overnight_fno", True)
    ):
        out.append(
            RuleBreach(
                rule="overnight_fno_not_permitted",
                severity=Severity.BLOCKER,
                detail="This mandate does not carry derivative positions overnight; gap risk "
                       "on Indian single stocks regularly exceeds the intraday stop.",
                remedy="Convert to an intraday MIS trade or use the cash segment.",
            )
        )

    if micro.is_expiry_day and is_derivative and not config.get("allow_expiry_day_entry", False):
        out.append(
            RuleBreach(
                rule="expiry_day_entry",
                severity=Severity.WARNING,
                detail="New derivative positions on expiry day face pinning toward high-OI "
                       "strikes and a collapse in time value through the final hour.",
                remedy="Use the next expiry, or wait for the following session.",
            )
        )

    if (
        proposal.instrument == Instrument.OPTIONS
        and proposal.option_legs
        and micro.days_to_expiry is not None
        and micro.days_to_expiry <= 2
        and any(leg.side.value == "BUY" for leg in proposal.option_legs)
    ):
        out.append(
            RuleBreach(
                rule="long_option_near_expiry",
                severity=Severity.WARNING,
                detail="Long options this close to expiry are mostly theta and gamma; the "
                       "premium can go to zero while the directional call is still correct.",
                observed=f"{micro.days_to_expiry} session(s) to expiry",
                remedy="Roll to the next expiry or convert to a debit spread.",
            )
        )

    turnover = micro.avg_daily_turnover_inr
    floor = float(config.get("min_avg_daily_turnover_inr", 5_00_00_000))
    if turnover is not None and turnover < floor:
        out.append(
            RuleBreach(
                rule="insufficient_liquidity",
                severity=Severity.BLOCKER,
                detail="Average traded value is below the desk floor. Impact cost on entry "
                       "and, worse, on a forced exit will swamp the modelled edge.",
                observed=inr(turnover),
                limit=inr(floor),
            )
        )

    if micro.circuit_band_pct and proposal.stop_loss and bundle.last_price:
        stop_pct = abs(bundle.last_price - proposal.stop_loss) / bundle.last_price * 100
        if stop_pct > micro.circuit_band_pct:
            out.append(
                RuleBreach(
                    rule="stop_outside_circuit_band",
                    severity=Severity.BLOCKER,
                    detail="The stop sits beyond the daily price band, so the scrip would "
                           "lock limit-down before the stop could ever be filled. The real "
                           "risk is unbounded across sessions, not the modelled amount.",
                    observed=f"stop {stop_pct:.2f}% away, band {micro.circuit_band_pct:.0f}%",
                    remedy="Tighten the stop inside the band or size for a multi-session gap.",
                )
            )

    if bundle.corporate_actions:
        from tradingagents_in.dataflows.corporate_actions import upcoming_actions

        pending = upcoming_actions(bundle.corporate_actions, within_days=30,
                                   as_of=bundle.session.trade_date)
        if pending:
            out.append(
                RuleBreach(
                    rule="pending_corporate_action",
                    severity=Severity.WARNING,
                    detail="A corporate action falls inside the intended holding period. "
                           "Prices, stops and F&O lot sizes are all adjusted on the ex-date.",
                    observed="; ".join(str(a) for a in pending[:3]),
                    remedy="Re-base the entry, stop and targets after the ex-date.",
                )
            )
    return out


# ------------------------------------------------------------ position risk
def check_position_risk(
    proposal: TradeProposal,
    bundle: MarketBundle,
    portfolio: PortfolioSnapshot,
    config: dict[str, Any],
) -> list[RuleBreach]:
    out: list[RuleBreach] = []
    capital = float(config.get("capital_inr", portfolio.capital_inr or 0)) or 1.0

    max_position = float(config.get("max_position_pct", 10.0))
    if proposal.position_size_pct > max_position + 1e-6:
        out.append(
            RuleBreach(
                rule="position_size_limit",
                severity=Severity.WARNING,
                detail="Single-name exposure above the mandate limit.",
                observed=f"{proposal.position_size_pct:.2f}%",
                limit=f"{max_position:.2f}%",
                remedy=f"Trim to {max_position:.2f}% of capital.",
            )
        )

    risk_pct = (proposal.risk_amount_inr or 0.0) / capital * 100
    max_risk = float(config.get("risk_per_trade_pct", 1.0))
    if risk_pct > max_risk * 1.05:
        out.append(
            RuleBreach(
                rule="risk_per_trade_limit",
                severity=Severity.BLOCKER,
                detail="Loss at the stop exceeds the per-trade risk budget.",
                observed=f"{risk_pct:.2f}% of capital ({inr_plain(proposal.risk_amount_inr)})",
                limit=f"{max_risk:.2f}%",
                remedy="Reduce quantity until the stop-out equals the budget.",
            )
        )

    deployed = proposal.capital_deployed_inr or 0.0
    if deployed > portfolio.cash_inr > 0 and proposal.product_type == ProductType.CNC:
        out.append(
            RuleBreach(
                rule="insufficient_cash",
                severity=Severity.BLOCKER,
                detail="Delivery (CNC) requires the full amount upfront; there is not enough "
                       "free cash in the book.",
                observed=inr(deployed),
                limit=inr(portfolio.cash_inr),
            )
        )
    return out


# ----------------------------------------------------------- portfolio risk
def check_portfolio_risk(
    proposal: TradeProposal,
    bundle: MarketBundle,
    portfolio: PortfolioSnapshot,
    config: dict[str, Any],
) -> list[RuleBreach]:
    out: list[RuleBreach] = []
    sector = bundle.ref.sector or "Unclassified"

    existing_sector = portfolio.sector_exposure_pct.get(sector, 0.0)
    combined_sector = existing_sector + proposal.position_size_pct
    max_sector = float(config.get("max_sector_pct", 25.0))
    if combined_sector > max_sector + 1e-6:
        out.append(
            RuleBreach(
                rule="sector_concentration",
                severity=Severity.WARNING,
                detail=f"Adding this position takes {sector} exposure past the sector cap. "
                       "Indian sector moves are highly correlated - a rate decision moves "
                       "every bank at once.",
                observed=f"{combined_sector:.2f}% ({existing_sector:.2f}% existing)",
                limit=f"{max_sector:.2f}%",
                remedy=f"Trim to {max(0.0, max_sector - existing_sector):.2f}% or rotate out "
                       "of an existing name in the sector.",
            )
        )

    existing_symbol = portfolio.symbol_exposure_pct.get(bundle.ref.base_symbol, 0.0)
    if existing_symbol > 0:
        combined = existing_symbol + proposal.position_size_pct
        if combined > float(config.get("max_position_pct", 10.0)):
            out.append(
                RuleBreach(
                    rule="symbol_already_held",
                    severity=Severity.WARNING,
                    detail="An existing position in this scrip plus the new one breaches the "
                           "single-name limit.",
                    observed=f"{combined:.2f}% combined",
                    limit=f"{config.get('max_position_pct', 10.0):.2f}%",
                    remedy="Treat this as an add-on and size the increment only.",
                )
            )

    risk_pct = (proposal.risk_amount_inr or 0.0) / max(
        float(config.get("capital_inr", 1)), 1.0
    ) * 100
    combined_heat = portfolio.portfolio_heat_pct + risk_pct
    max_heat = float(config.get("max_portfolio_heat_pct", 6.0))
    if combined_heat > max_heat + 1e-6:
        out.append(
            RuleBreach(
                rule="portfolio_heat",
                severity=Severity.BLOCKER,
                detail="Total open risk across the book would exceed the heat limit. "
                       "Correlated stops in a gap-down take all of it at once.",
                observed=f"{combined_heat:.2f}%",
                limit=f"{max_heat:.2f}%",
                remedy="Close or tighten an existing position before adding this one.",
            )
        )

    max_open = int(config.get("max_open_positions", 12))
    if portfolio.open_positions >= max_open:
        out.append(
            RuleBreach(
                rule="max_open_positions",
                severity=Severity.WARNING,
                detail="The book already holds the maximum number of positions it can "
                       "actively monitor.",
                observed=str(portfolio.open_positions),
                limit=str(max_open),
            )
        )

    if (
        proposal.instrument in (Instrument.FUTURES, Instrument.OPTIONS)
        and proposal.time_horizon != TimeHorizon.INTRADAY
    ):
        overnight = portfolio.overnight_fno_exposure_pct + proposal.position_size_pct
        cap = float(config.get("max_position_pct", 10.0)) * 3
        if overnight > cap:
            out.append(
                RuleBreach(
                    rule="overnight_fno_concentration",
                    severity=Severity.WARNING,
                    detail="Aggregate carry-forward derivative exposure is high. Indian "
                           "single-stock futures gap on overnight news with no chance to "
                           "act on the stop.",
                    observed=f"{overnight:.2f}%",
                    limit=f"{cap:.2f}%",
                )
            )
    return out


def check_portfolio_state(
    portfolio: PortfolioSnapshot, config: dict[str, Any]
) -> list[RuleBreach]:
    """State checks that apply regardless of the proposal, including HOLD."""
    out: list[RuleBreach] = []
    max_dd = float(config.get("max_drawdown_pct", 15.0))
    if portfolio.current_drawdown_pct >= max_dd:
        out.append(
            RuleBreach(
                rule="max_drawdown_breached",
                severity=Severity.BLOCKER,
                detail="The book is at or beyond its maximum drawdown. New risk is suspended "
                       "until the mandated review is complete.",
                observed=f"{portfolio.current_drawdown_pct:.2f}%",
                limit=f"{max_dd:.2f}%",
                remedy="Halt new positions, review the open book, then resume at half size.",
            )
        )
    elif portfolio.current_drawdown_pct >= max_dd * 0.66:
        out.append(
            RuleBreach(
                rule="drawdown_warning",
                severity=Severity.WARNING,
                detail="Drawdown is approaching the mandate limit; risk should be de-escalated "
                       "rather than added to.",
                observed=f"{portfolio.current_drawdown_pct:.2f}%",
                limit=f"{max_dd:.2f}%",
                remedy="Halve the per-trade risk budget until the drawdown recovers.",
            )
        )
    return out


# ------------------------------------------------------------------ helpers
def stress_test(
    proposal: TradeProposal, bundle: MarketBundle, config: dict[str, Any]
) -> dict[str, float]:
    """P&L under adverse Indian scenarios, in INR.

    The gap scenarios are the ones that matter: an Indian stock can open 5-10%
    away from the previous close on a results miss or a regulatory action, and
    the stop provides no protection across that gap.
    """
    entry = proposal.entry_zone.mid if proposal.entry_zone else bundle.last_price
    quantity = proposal.quantity or 0
    if not entry or quantity <= 0:
        return {}

    direction = -1 if proposal.action in (Action.SELL, Action.SHORT) else 1
    scenarios = {
        "stop_hit": (
            (proposal.stop_loss - entry) / entry * 100 if proposal.stop_loss else -2.0
        ),
        "gap_down_2pct": -2.0,
        "gap_down_5pct": -5.0,
        "lower_circuit_hit": -(bundle.micro.circuit_band_pct or 10.0),
        "index_breaker_10pct": -10.0,
    }
    out: dict[str, float] = {}
    for name, move_pct in scenarios.items():
        pnl = entry * move_pct / 100 * quantity * direction
        if proposal.instrument == Instrument.OPTIONS:
            # A long option cannot lose more than the premium paid.
            pnl = max(pnl, -(proposal.capital_deployed_inr or 0.0))
        out[name] = round(pnl, 2)
    return out


def sebi_compliance_notes(
    proposal: TradeProposal, bundle: MarketBundle, config: dict[str, Any]
) -> list[str]:
    """Standing regulatory context the risk manager must surface to the caller."""
    notes = [
        "Equity cash settles on T+1; sale proceeds are available for withdrawal on T+1 "
        "and the shares must be in the demat account before a delivery sale.",
        "Peak-margin reporting applies intraday: the broker checks margin at random "
        "snapshots, so a position that is compliant at entry can still attract a penalty.",
    ]
    if proposal.instrument in (Instrument.FUTURES, Instrument.OPTIONS):
        notes.append(
            "Derivative positions are subject to client-level and market-wide position "
            "limits; breaching the market-wide limit puts the scrip into the F&O ban list."
        )
        notes.append(
            "Physical settlement applies to stock F&O held into expiry: an in-the-money "
            "position becomes a delivery obligation with the full contract value."
        )
    if proposal.product_type == ProductType.MIS:
        notes.append(
            "MIS positions are auto-squared-off by the broker (typically 15:10-15:20) at "
            "whatever the market offers, not at your stop."
        )
    if bundle.ref.is_index:
        notes.append(
            "Index-level circuit breakers halt the entire market at 10/15/20% moves; the "
            "halt duration depends on the trigger time."
        )
    if proposal.action == Action.SHORT:
        notes.append(
            "Short positions must be disclosed as such at order entry; naked short selling "
            "in the cash segment is prohibited and institutional shorts require SLB."
        )
    return notes[:8]


def _atr_pct(bundle: MarketBundle) -> float | None:
    daily = bundle.enriched.get("1d")
    if daily is None or daily.empty or "atr_pct" not in daily.columns:
        return None
    value = daily["atr_pct"].iloc[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise(breaches: list[RuleBreach]) -> str:
    if not breaches:
        return "No rule breaches: the proposal is within every mandate limit."
    lines = []
    for b in breaches:
        marker = {"BLOCKER": "BLOCK", "WARNING": "WARN ", "INFO": "INFO "}[b.severity.value]
        observed = f" [observed {b.observed}" + (f", limit {b.limit}]" if b.limit else "]") \
            if b.observed else ""
        lines.append(f"- `{marker}` **{b.rule}**: {b.detail}{observed}")
    return "\n".join(lines)
