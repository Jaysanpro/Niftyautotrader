"""The Trader agent.

Turns a research view into an order a dealer could actually work: an entry
band, a stop that sits at a level rather than a round number, staged targets,
a size derived from the stop distance, and an instrument chosen for the
horizon rather than for excitement.

Instrument selection is a real decision in India, not a formality:

* **Cash / CNC** - the default. Full capital, T+1 settlement, no expiry, no
  margin call. Anything positional starts here.
* **Futures / NRML** - leverage and the only clean way to hold a short past the
  session. Costs a lot size worth of indivisibility and gap risk.
* **Options** - the right answer when the view has a date attached (a result, a
  policy meeting) or when defined risk matters more than delta. Long single
  options near expiry are how most retail P&L dies, so the deterministic engine
  prefers a debit spread and refuses to buy weeklies inside two sessions.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent
from tradingagents_in.agents.briefings import (
    corporate_actions_brief,
    header,
    microstructure_brief,
    technical_brief,
)
from tradingagents_in.config.universe import derivatives_underlying
from tradingagents_in.dataflows.fno import (
    black_scholes_price,
    resolve_expiry,
    round_to_strike,
    theta_warning,
)
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.enums import (
    Action,
    AgentRole,
    Instrument,
    LegSide,
    OptionType,
    ProductType,
    Stance,
    TimeHorizon,
)
from tradingagents_in.models.research import InvestmentPlan
from tradingagents_in.models.trade import (
    EntryZone,
    FuturesSpec,
    OptionLeg,
    Target,
    TradeProposal,
)
from tradingagents_in.risk.costs import estimate_costs, net_reward_risk
from tradingagents_in.risk.sizing import size_for_proposal
from tradingagents_in.utils.formatting import inr_plain

_LONG_STANCES = (Stance.BULLISH, Stance.MILDLY_BULLISH)
_SHORT_STANCES = (Stance.BEARISH, Stance.MILDLY_BEARISH)


class Trader(Agent[TradeProposal]):
    role = AgentRole.TRADER
    output_model = TradeProposal
    tier_key = "trader"

    def system_prompt(self) -> str:
        return """\
You are the Trader. The research view is settled; your job is execution.

Produce an order, not an opinion. Every field must be something a dealer could
act on this session:

- **Entry zone**: a band, with the condition that triggers it. "Buy on a close
  above 2,940 with the zone 2,905-2,945" is an order. "Buy on dips" is not.
- **Stop-loss**: at a structural level - below the swing low, below the
  supertrend flip, under the VWAP anchor - not at a round percentage. State
  what the level means.
- **Targets**: staged, with the portion to book at each. Indian retail books
  too late; a first target that pays for the trade is worth more than a heroic
  single target.
- **Position size**: derived from the stop distance and the risk budget. If the
  stop is wide, the position is small. Never widen the risk to fit a position.
- **Horizon**: intraday, swing or positional, and it must match the thesis and
  the product. A capex-cycle thesis is not an MIS trade.
- **Instrument**: cash/CNC by default. Use futures only when you need leverage
  or a carry-forward short. Use options when the view has a date attached or
  when defined risk matters - and prefer a debit spread to a naked long option
  inside the final week of expiry.

Hard India constraints you must respect:
- A cash-segment short is intraday-only (MIS). Never propose CNC on a short.
- Stock options and futures are monthly; only index derivatives have weeklies.
- Option strikes must sit on the exchange strike ladder, and every leg needs a
  real expiry date.
- If the scrip is circuit-locked, in the F&O ban list, under GSM, or too thin
  to exit, the correct proposal is HOLD with the reason stated. Do not hand a
  problem to the risk manager that you can see yourself.
- STT, stamp duty, exchange charges and GST are real. A trade whose first
  target barely clears the round trip is not a trade.

If the research view is NEUTRAL or the setup has no acceptable stop, propose
HOLD and say precisely what you are waiting for. A HOLD with a named trigger is
a professional answer; a marginal trade taken for the sake of activity is not.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        plan: InvestmentPlan | None = state.get("investment_plan")
        parts = [
            header(bundle),
            technical_brief(bundle),
            microstructure_brief(bundle),
            corporate_actions_brief(bundle),
        ]

        if plan is not None:
            parts.append(
                "## Research view (settled - do not re-litigate)\n\n"
                f"- Stance: **{plan.stance.value}**, conviction {plan.conviction:.0%}\n"
                f"- Winning side: {plan.winning_side}\n"
                f"- Verdict: {plan.verdict}\n"
                f"- Preferred horizon: {plan.preferred_horizon.value}\n"
                f"- Decisive arguments:\n"
                + "\n".join(f"  - {a}" for a in plan.decisive_arguments)
                + "\n- Monitorables:\n"
                + "\n".join(f"  - {m}" for m in plan.key_monitorables)
            )

        technical = state.get("technical_report")
        if technical is not None:
            parts.append(
                "## Technical analyst's levels\n\n"
                f"- Suggested stop: {technical.suggested_stop_loss}\n"
                f"- Suggested targets: {technical.suggested_targets}\n"
                f"- Volatility regime: {technical.volatility_regime}\n"
                f"- Multi-timeframe alignment: {technical.multi_timeframe_alignment}"
            )

        cfg = self.config
        parts.append(
            "## Mandate\n\n"
            f"- Capital: {inr_plain(cfg.get('capital_inr'))} ({cfg.get('profile')} profile)\n"
            f"- Risk per trade: {cfg.get('risk_per_trade_pct')}% "
            f"({inr_plain(float(cfg.get('capital_inr', 0)) * float(cfg.get('risk_per_trade_pct', 1)) / 100)})\n"
            f"- Max single-name exposure: {cfg.get('max_position_pct')}%\n"
            f"- Minimum reward:risk after costs: {cfg.get('min_reward_risk')}\n"
            f"- F&O permitted: {cfg.get('allow_fno')} | Overnight F&O: "
            f"{cfg.get('allow_overnight_fno')} | Cash shorts: {cfg.get('allow_short_cash')}"
        )

        review = state.get("risk_assessment")
        if review is not None:
            parts.append(
                "## Risk manager's review of your previous proposal\n\n"
                f"- Verdict: **{review.verdict.value}**, approved size "
                f"{review.approved_position_size_pct:.2f}%\n"
                "- Required changes:\n"
                + "\n".join(f"  - {c}" for c in review.required_changes)
                + "\n- Breaches:\n"
                + "\n".join(
                    f"  - [{b.severity.value}] {b.rule}: {b.detail}" for b in review.breaches
                )
                + "\n\nRevise the proposal to clear these. Do not argue with a BLOCKER - "
                "either restructure the trade so it no longer applies, or propose HOLD."
            )

        if state.get("memory_context"):
            parts.append("## Lessons from previous decisions\n\n" + state["memory_context"])

        parts.append(
            """## Your task

Return one TradeProposal. Fill entry_zone (with a trigger), stop_loss, targets
with book_pct, position_size_pct, quantity, capital_deployed_inr and
risk_amount_inr consistently - the numbers must agree with each other. Write
`invalidation` as the one observable event that makes the idea wrong, and put
the liquidity, circuit, expiry and square-off mechanics in execution_notes.
List what you considered and rejected in alternatives_considered."""
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------- offline
    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> TradeProposal:
        return build_proposal(bundle, state, self.config, role=self.role)


# ---------------------------------------------------------------------------
# Deterministic proposal construction
# ---------------------------------------------------------------------------
def build_proposal(
    bundle: MarketBundle,
    state: dict[str, Any],
    config: dict[str, Any],
    role: AgentRole = AgentRole.TRADER,
) -> TradeProposal:
    """A complete, sized, costed proposal computed from data alone."""
    plan: InvestmentPlan | None = state.get("investment_plan")
    technical = state.get("technical_report")
    price = bundle.last_price
    symbol = bundle.ref.yf_symbol

    stance = plan.stance if plan else Stance.NEUTRAL
    conviction = plan.conviction if plan else 0.3
    horizon = plan.preferred_horizon if plan else TimeHorizon.SWING

    if price is None:
        return _hold(
            symbol, role, "No live price available, so no entry, stop or size can be set.",
            bundle,
        )

    if stance == Stance.NEUTRAL:
        waiting = (plan.key_monitorables[:3] if plan else []) or [
            "a decisive break of the nearest technical level"
        ]
        return _hold(
            symbol, role,
            "Research view is neutral, so there is no directional edge to express. "
            "Waiting on: " + "; ".join(waiting),
            bundle, horizon=horizon,
        )

    blockers = _hard_blockers(bundle, config)
    if blockers:
        return _hold(
            symbol, role,
            "Directional view exists but the scrip is not tradable: " + "; ".join(blockers),
            bundle, horizon=horizon,
        )

    long_side = stance in _LONG_STANCES
    action = Action.BUY if long_side else _short_action(bundle, config)

    entry, stop, targets = _levels(bundle, technical, long_side, price)
    if stop is None:
        return _hold(
            symbol, role,
            "No structural level close enough to act as a stop; risk cannot be defined.",
            bundle, horizon=horizon,
        )

    instrument, product, horizon = _choose_instrument(
        bundle, config, action, horizon, conviction
    )

    # Derivative legs are built *before* construction: TradeProposal's coherence
    # validator downgrades an OPTIONS proposal that carries no legs to CASH, so
    # attaching them afterwards would silently turn every option idea into a
    # cash trade.
    legs, strategy, futures = _derivative_legs(instrument, bundle, config, price, action, horizon)

    proposal = TradeProposal(
        role=role,
        symbol=symbol,
        action=action,
        instrument=instrument,
        product_type=product,
        time_horizon=horizon,
        entry_zone=entry,
        stop_loss=round(stop, 2),
        targets=[Target(price=round(t, 2), book_pct=pct, rationale=why)
                 for t, pct, why in targets],
        futures=futures,
        option_legs=legs,
        option_strategy=strategy,
        confidence=round(min(0.9, conviction), 3),
        summary="",
        data_quality=bundle.quality(["prices", "quote", "microstructure"]),
    )

    _size_and_cost(proposal, bundle, config, price, state)
    _annotate(proposal, bundle, config, plan)
    return proposal


# ------------------------------------------------------------------ levels
def _levels(
    bundle: MarketBundle, technical: Any, long_side: bool, price: float
) -> tuple[EntryZone, float | None, list[tuple[float, float, str]]]:
    """Entry band, stop and staged targets, anchored on real structure."""
    atr = _atr(bundle) or price * 0.015
    supports, resistances = _levels_from_report(technical)

    if long_side:
        # Enter on a shallow pullback into the band, stop under the nearest
        # support (or one ATR, whichever is further and therefore safer).
        low, high = price - 0.4 * atr, price + 0.3 * atr
        structural = max([s for s in supports if s < price], default=None)
        stop = min(structural, price - atr) if structural else price - 1.5 * atr
        if technical is not None and technical.suggested_stop_loss:
            stop = min(stop, float(technical.suggested_stop_loss))
        risk = price - stop
        near_resistance = min([r for r in resistances if r > price], default=None)
        t1 = near_resistance if near_resistance and near_resistance > price + risk else price + 1.5 * risk
        targets = [
            (t1, 50.0, "First resistance / 1.5R - books enough to pay for the trade"),
            (price + 3.0 * risk, 30.0, "3R extension if the trend holds"),
            (price + 5.0 * risk, 20.0, "Runner, trailed on the supertrend flip"),
        ]
        trigger = (
            f"Entry valid while price holds above {stop:,.2f}; a close below it "
            "cancels the setup rather than triggering it."
        )
    else:
        low, high = price - 0.3 * atr, price + 0.4 * atr
        structural = min([r for r in resistances if r > price], default=None)
        stop = max(structural, price + atr) if structural else price + 1.5 * atr
        if technical is not None and technical.suggested_stop_loss:
            stop = max(stop, float(technical.suggested_stop_loss))
        risk = stop - price
        near_support = max([s for s in supports if s < price], default=None)
        t1 = near_support if near_support and near_support < price - risk else price - 1.5 * risk
        targets = [
            (t1, 50.0, "First support / 1.5R"),
            (price - 3.0 * risk, 30.0, "3R extension"),
            (price - 5.0 * risk, 20.0, "Runner"),
        ]
        trigger = (
            f"Entry valid while price stays below {stop:,.2f}; a close above it "
            "invalidates the short."
        )

    if risk <= 0:
        return EntryZone(low=low, high=high, reference_price=price), None, []

    zone = EntryZone(
        low=round(low, 2), high=round(high, 2), reference_price=round(price, 2), trigger=trigger
    )
    return zone, stop, targets


def _levels_from_report(technical: Any) -> tuple[list[float], list[float]]:
    if technical is None or not technical.timeframes:
        return [], []
    supports: list[float] = []
    resistances: list[float] = []
    for tf in technical.timeframes:
        supports.extend(tf.levels.supports)
        resistances.extend(tf.levels.resistances)
    return sorted(set(supports)), sorted(set(resistances))


def _atr(bundle: MarketBundle) -> float | None:
    daily = bundle.enriched.get("1d")
    if daily is None or daily.empty or "atr_14" not in daily.columns:
        return None
    try:
        value = float(daily["atr_14"].iloc[-1])
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- instrument
def _choose_instrument(
    bundle: MarketBundle,
    config: dict[str, Any],
    action: Action,
    horizon: TimeHorizon,
    conviction: float,
) -> tuple[Instrument, ProductType, TimeHorizon]:
    """Pick the segment and product that actually fit the trade."""
    fno_ok = bool(config.get("allow_fno", True)) and (
        bundle.ref.fno_eligible or bundle.ref.is_index
    ) and not bundle.micro.in_fno_ban

    # An index has no cash instrument a retail account can hold, so it is
    # derivatives or nothing.
    if bundle.ref.is_index:
        if not fno_ok:
            return Instrument.CASH, ProductType.CNC, horizon
        product = ProductType.MIS if horizon == TimeHorizon.INTRADAY else ProductType.NRML
        # Defined risk on an index is worth the spread; a naked index future
        # gaps hard on global cues overnight.
        instrument = Instrument.OPTIONS if horizon != TimeHorizon.INTRADAY else Instrument.FUTURES
        return instrument, product, horizon

    if action == Action.SHORT:
        if fno_ok and config.get("allow_overnight_fno", True) and horizon != TimeHorizon.INTRADAY:
            return Instrument.FUTURES, ProductType.NRML, horizon
        # Cash shorts cannot survive the close.
        return Instrument.CASH, ProductType.MIS, TimeHorizon.INTRADAY

    if horizon == TimeHorizon.INTRADAY:
        return Instrument.CASH, ProductType.MIS, horizon

    return Instrument.CASH, ProductType.CNC, horizon


def _derivative_legs(
    instrument: Instrument,
    bundle: MarketBundle,
    config: dict[str, Any],
    price: float,
    action: Action,
    horizon: TimeHorizon,
) -> tuple[list[OptionLeg], str | None, FuturesSpec | None]:
    """Build the futures contract or option structure for the chosen segment.

    Lot counts are placeholders of 1 here; the sizer sets the real number once
    it knows what the risk budget affords.
    """
    if instrument == Instrument.FUTURES:
        expiry = resolve_expiry(bundle.ref, when=bundle.session.trade_date)
        underlying = derivatives_underlying(bundle.ref.base_symbol)
        return [], None, FuturesSpec(
            expiry=str(expiry),
            lots=1,
            lot_size=bundle.ref.lot_size,
            contract_label=f"{underlying} {expiry:%d%b%y} FUT".upper(),
        )

    if instrument != Instrument.OPTIONS:
        return [], None, None

    long_side = action == Action.BUY
    expiry = resolve_expiry(bundle.ref, horizon_days=_horizon_days(horizon),
                            when=bundle.session.trade_date)
    dte = max((expiry - bundle.session.trade_date).days, 1)
    option_type = OptionType.CE if long_side else OptionType.PE

    # A debit spread rather than a naked long: it halves the premium, caps the
    # theta bleed, and the capped upside is an honest trade for the cost.
    long_strike = round_to_strike(price, bundle.ref.base_symbol, price)
    step_direction = 1 if long_side else -1
    short_strike = round_to_strike(
        price * (1 + step_direction * 0.04), bundle.ref.base_symbol, price
    )
    if short_strike == long_strike:
        short_strike = long_strike + step_direction * max(
            abs(round_to_strike(price * 1.02, bundle.ref.base_symbol, price) - long_strike), 1.0
        )

    vol = _implied_vol(bundle) or 22.0
    long_premium = _premium(bundle, long_strike, option_type, price, dte, vol)
    short_premium = _premium(bundle, short_strike, option_type, price, dte, vol)

    legs = [
        OptionLeg(
            side=LegSide.BUY, option_type=option_type, strike=long_strike,
            expiry=str(expiry), lots=1, lot_size=bundle.ref.lot_size,
            estimated_premium=long_premium,
        ),
        OptionLeg(
            side=LegSide.SELL, option_type=option_type, strike=short_strike,
            expiry=str(expiry), lots=1, lot_size=bundle.ref.lot_size,
            estimated_premium=short_premium,
        ),
    ]
    strategy = (
        f"{'bull call' if long_side else 'bear put'} spread "
        f"{long_strike:g}/{short_strike:g} {expiry:%d%b}"
    ).upper()
    return legs, strategy, None


def _premium(
    bundle: MarketBundle,
    strike: float,
    option_type: OptionType,
    spot: float,
    dte: int,
    vol_pct: float,
) -> float:
    """Live chain price where available, Black-Scholes estimate otherwise."""
    if bundle.option_chain is not None:
        quote = bundle.option_chain.quote(strike, option_type)
        if quote is not None and quote.mid:
            return float(quote.mid)
    return black_scholes_price(spot, strike, dte, vol_pct, option_type)


def _implied_vol(bundle: MarketBundle) -> float | None:
    if bundle.option_chain is not None:
        ivs = [
            q.implied_volatility
            for q in (bundle.option_chain.calls + bundle.option_chain.puts)
            if q.implied_volatility
        ]
        if ivs:
            return round(sum(ivs) / len(ivs), 2)
    daily = bundle.enriched.get("1d")
    if daily is not None and not daily.empty and "atr_pct" in daily.columns:
        try:
            return round(float(daily["atr_pct"].iloc[-1]) * 15.87, 2)  # sqrt(252)
        except (TypeError, ValueError):
            return None
    return None


def _horizon_days(horizon: TimeHorizon) -> int:
    return {TimeHorizon.INTRADAY: 1, TimeHorizon.SWING: 10, TimeHorizon.POSITIONAL: 45}[horizon]


# ------------------------------------------------------------ size & costs
def _size_and_cost(
    proposal: TradeProposal,
    bundle: MarketBundle,
    config: dict[str, Any],
    price: float,
    state: dict[str, Any],
) -> None:
    capital = float(config.get("capital_inr", 1_000_000))
    entry = proposal.entry_zone.mid if proposal.entry_zone else price

    option_premium = None
    option_max_loss = None
    if proposal.instrument == Instrument.OPTIONS and proposal.option_legs:
        debit = sum(
            (leg.estimated_premium or 0.0) * (1 if leg.side == LegSide.BUY else -1)
            for leg in proposal.option_legs
        )
        option_premium = round(max(debit, 0.05), 2)
        option_max_loss = option_premium  # a debit spread cannot lose more than its debit

    sizing = size_for_proposal(
        capital_inr=capital,
        entry_price=entry,
        stop_loss=proposal.stop_loss,
        instrument=proposal.instrument,
        action=proposal.action,
        base_symbol=bundle.ref.base_symbol,
        config=config,
        lot_size=bundle.ref.lot_size,
        is_index=bundle.ref.is_index,
        option_premium=option_premium,
        option_max_loss_per_share=option_max_loss,
    )

    if not sizing.is_tradable:
        proposal.action = Action.HOLD
        proposal.summary = (
            f"No tradable size on {bundle.ref.base_symbol}: "
            + (sizing.notes[0] if sizing.notes else f"blocked by {sizing.binding_constraint}.")
        )
        proposal.execution_notes = sizing.notes
        return

    # On a second pass the risk manager has already ruled on size. Honour it
    # rather than re-proposing the size that was just trimmed.
    review = state.get("risk_assessment")
    if review is not None and review.approved_position_size_pct > 0 and sizing.position_pct > 0:
        scale = min(1.0, review.approved_position_size_pct / sizing.position_pct)
        if scale < 1.0:
            sizing.quantity = int(sizing.quantity * scale)
            sizing.lots = int(sizing.lots * scale)
            if sizing.lots and sizing.lot_size:
                sizing.quantity = sizing.lots * sizing.lot_size
            sizing.capital_deployed_inr = round(sizing.capital_deployed_inr * scale, 2)
            sizing.risk_amount_inr = round(sizing.risk_amount_inr * scale, 2)
            sizing.position_pct = round(sizing.position_pct * scale, 3)
            sizing.notes.append(
                f"Size cut to the risk manager's approved "
                f"{review.approved_position_size_pct:.2f}% of capital."
            )
        if sizing.quantity <= 0:
            proposal.action = Action.HOLD
            proposal.summary = (
                "Risk-approved size rounds to zero units, so there is no position to take."
            )
            proposal.execution_notes = sizing.notes
            return

    proposal.quantity = sizing.quantity
    proposal.position_size_pct = min(sizing.position_pct, 100.0)
    proposal.capital_deployed_inr = sizing.capital_deployed_inr
    proposal.risk_amount_inr = sizing.risk_amount_inr

    if proposal.futures is not None:
        proposal.futures.lots = sizing.lots
    for leg in proposal.option_legs:
        leg.lots = sizing.lots

    if proposal.targets:
        rr, costs = net_reward_risk(
            entry_price=entry,
            stop_loss=proposal.stop_loss or entry,
            target=proposal.targets[0].price,
            quantity=sizing.quantity,
            instrument=proposal.instrument,
            product_type=proposal.product_type,
            action=proposal.action,
            config=config,
        )
        proposal.reward_risk = rr
        proposal.costs = costs
    else:
        proposal.costs = estimate_costs(
            entry_price=entry, quantity=sizing.quantity,
            instrument=proposal.instrument, product_type=proposal.product_type,
            action=proposal.action, config=config,
        )

    proposal.expected_holding_days = _horizon_days(proposal.time_horizon)
    proposal.execution_notes.extend(sizing.notes)


def _annotate(
    proposal: TradeProposal,
    bundle: MarketBundle,
    config: dict[str, Any],
    plan: InvestmentPlan | None,
) -> None:
    """Fill in the prose fields: summary, invalidation, execution mechanics."""
    if proposal.action == Action.HOLD:
        return

    micro = bundle.micro
    notes = proposal.execution_notes

    if micro.avg_daily_turnover_inr:
        notes.append(
            f"Average traded value {micro.avg_daily_turnover_inr / 1e7:,.1f} Cr/day; "
            f"this order is {(proposal.capital_deployed_inr or 0) / max(micro.avg_daily_turnover_inr, 1) * 100:.3f}% "
            "of a session's turnover, so work it rather than sending it at market."
        )
    if micro.circuit_band_pct:
        notes.append(
            f"Daily price band {micro.circuit_band_pct:.0f}%: the stop is only fillable "
            "while the scrip trades inside the band."
        )
    if proposal.product_type == ProductType.MIS:
        notes.append(
            "MIS: the broker auto-squares-off around 15:10-15:20 at market. Exit on your "
            "own terms before that rather than being closed at theirs."
        )
    if proposal.instrument == Instrument.FUTURES:
        notes.append(
            "Stock futures are physically settled at expiry - roll or close before the "
            "expiry session to avoid a delivery obligation for the full contract value."
        )
    if proposal.option_legs:
        warning = theta_warning(micro.days_to_expiry or 30, is_long_option=True)
        if warning:
            notes.append(warning)
        notes.append(
            "Enter the spread as a single two-leg order; legging in on an illiquid strike "
            "gives up more than the spread is worth."
        )
    if bundle.session.is_expiry_day:
        notes.append(
            "Expiry session: prices pin toward high open-interest strikes into the close, "
            "so intraday levels are unreliable."
        )

    direction = "above" if proposal.action == Action.BUY else "below"
    proposal.invalidation = (
        f"A daily close {direction} {proposal.stop_loss:,.2f} on rising volume. That level "
        "is the structure the entry depends on; losing it means the read was wrong, not "
        "that the position needs more room."
    )

    plan_note = f" Research conviction {plan.conviction:.0%}." if plan else ""
    proposal.summary = (
        f"{proposal.action.value} {bundle.ref.base_symbol} via {proposal.instrument.value} "
        f"({proposal.product_type.value}, {proposal.time_horizon.value}) in the "
        f"{proposal.entry_zone} band, stop {proposal.stop_loss:,.2f}, first target "
        f"{proposal.targets[0].price:,.2f}. Size {proposal.position_size_pct:.2f}% of capital "
        f"risking {inr_plain(proposal.risk_amount_inr)}"
        + (f" at {proposal.reward_risk:.2f}:1 after charges." if proposal.reward_risk else ".")
        + plan_note
    )
    proposal.reasoning = (
        "Size is derived from the stop distance and the risk budget, not from conviction. "
        "Instrument chosen for the horizon: "
        + _instrument_rationale(proposal, bundle)
    )
    proposal.alternatives_considered = _alternatives(proposal, bundle, config)


def _instrument_rationale(proposal: TradeProposal, bundle: MarketBundle) -> str:
    if proposal.instrument == Instrument.CASH:
        if proposal.product_type == ProductType.CNC:
            return (
                "cash delivery avoids expiry, margin calls and auto-square-off, which suits "
                "a multi-session hold."
            )
        return "intraday MIS, because the setup resolves within the session and carries no gap risk."
    if proposal.instrument == Instrument.FUTURES:
        return (
            f"futures give a carry-forward position with margin efficiency; lot size "
            f"{bundle.ref.lot_size} makes this the smallest expressible size."
        )
    return (
        "a debit spread caps the loss at the net premium and cuts the theta bleed that "
        "kills naked long options."
    )


def _alternatives(
    proposal: TradeProposal, bundle: MarketBundle, config: dict[str, Any]
) -> list[str]:
    out: list[str] = []
    if proposal.instrument == Instrument.CASH and (
        bundle.ref.fno_eligible or bundle.ref.is_index
    ):
        out.append(
            "Futures: rejected because the lot size forces a position larger than the risk "
            "budget allows at this stop distance."
            if bundle.ref.lot_size else
            "Futures: rejected, no lot size on record to size against."
        )
        out.append(
            "Long option: rejected in favour of cash - the directional view has no date "
            "attached, so paying theta buys nothing."
        )
    if proposal.instrument == Instrument.OPTIONS:
        out.append(
            "Naked long option: rejected - the same directional exposure at roughly double "
            "the premium and far worse decay."
        )
        out.append("Futures: rejected - undefined downside on an overnight gap.")
    if proposal.time_horizon != TimeHorizon.INTRADAY:
        out.append(
            "Intraday version: rejected - the thesis needs more than one session to play out "
            "and MIS auto-square-off would close it arbitrarily."
        )
    return out[:4]


# ----------------------------------------------------------------- helpers
def _short_action(bundle: MarketBundle, config: dict[str, Any]) -> Action:
    """SELL closes an existing long; SHORT opens a new one."""
    return Action.SHORT


def _hard_blockers(bundle: MarketBundle, config: dict[str, Any]) -> list[str]:
    micro = bundle.micro
    out: list[str] = []
    if micro.is_circuit_locked:
        out.append("circuit-locked, so no exit liquidity at the stop")
    if micro.surveillance.value == "GSM":
        out.append("under GSM surveillance with restricted trading")
    turnover = micro.avg_daily_turnover_inr
    floor = float(config.get("min_avg_daily_turnover_inr", 5_00_00_000))
    if turnover is not None and turnover < floor:
        out.append(
            f"average traded value {turnover / 1e7:.2f} Cr/day is below the "
            f"{floor / 1e7:.1f} Cr desk floor"
        )
    return out


def _hold(
    symbol: str,
    role: AgentRole,
    reason: str,
    bundle: MarketBundle,
    horizon: TimeHorizon = TimeHorizon.SWING,
) -> TradeProposal:
    return TradeProposal(
        role=role,
        symbol=symbol,
        action=Action.HOLD,
        instrument=Instrument.CASH,
        product_type=ProductType.CNC,
        time_horizon=horizon,
        confidence=0.5,
        summary=reason,
        reasoning=reason,
        invalidation="Not applicable - no position is being taken.",
        data_quality=bundle.quality(["prices", "quote", "microstructure"]),
    )
