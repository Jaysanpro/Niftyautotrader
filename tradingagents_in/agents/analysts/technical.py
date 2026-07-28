"""Technical Analyst.

Multi-timeframe read on price, with the indicator set an Indian desk actually
watches (Supertrend and VWAP as much as RSI/MACD), plus pattern recognition
for local quirks: gap-up opens driven by overnight US cues, expiry-week pinning
and circuit-driven moves.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingagents_in.agents.base import Agent, confidence_from_quality
from tradingagents_in.agents.briefings import header, microstructure_brief, technical_brief
from tradingagents_in.dataflows import indicators as ind
from tradingagents_in.dataflows import market_data
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.analyst import LevelSet, TechnicalReport, TimeframeSignal
from tradingagents_in.models.base import Evidence
from tradingagents_in.models.enums import AgentRole, Stance

EXPECTED_SOURCES = ["prices", "quote", "benchmark", "microstructure"]


class TechnicalAnalyst(Agent[TechnicalReport]):
    role = AgentRole.TECHNICAL_ANALYST
    output_model = TechnicalReport
    tier_key = "technical_analyst"

    def system_prompt(self) -> str:
        return """\
You are the Technical Analyst.

Your job is to read price and volume across timeframes and produce levels a
trader can actually use. You do not care about valuation or news except where
they show up in the tape.

Method:
- Establish the trend on the higher timeframe first, then look for entries on
  the lower one. When timeframes disagree, say so plainly - conflicting
  timeframes mean smaller size, not a louder opinion.
- Quote levels as prices, never as vague zones. A stop is a number.
- Anchor stops on structure (swing low, Supertrend line, VWAP) widened by ATR,
  not on a round percentage.
- Volume confirms. A breakout on below-average volume in an Indian mid cap is
  usually an operator move, not accumulation. Delivery percentage matters more
  than traded volume for positional calls.

India-specific patterns to check for explicitly:
- Gap-up/gap-down opens driven by overnight US and Asian cues: the 09:15 print
  is often the day's extreme, and the first 15 minutes routinely trap breakout
  buyers.
- Expiry-week pinning toward high-OI strikes, which flattens intraday trends
  on Wednesday and Thursday for NSE names.
- Circuit-driven moves: a series of upper-circuit closes is not momentum you
  can trade, it is an absence of sellers.
- The 09:15-09:30 and 14:30-15:15 windows carry most of the day's real volume.

Calibration: RSI above 70 is not automatically a sell in a trending Indian
large cap; ADX below 20 means the trend indicators are noise.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        return "\n\n".join(
            [
                header(bundle),
                technical_brief(bundle),
                microstructure_brief(bundle),
                """## Your task

Produce a TechnicalReport. Specifically:
1. Fill one TimeframeSignal per timeframe present in the briefing, using only
   the values given.
2. State the multi-timeframe alignment in one sentence.
3. Name concrete patterns you can actually see in the data, including any of
   the India-specific ones listed in your instructions. Do not list a pattern
   you cannot point at.
4. Give a suggested stop-loss and up to three targets as prices, derived from
   structure and ATR.
5. List what would change your mind, as price levels or observable events.""",
            ]
        )

    # ------------------------------------------------------------- offline
    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> TechnicalReport:
        quality = bundle.quality(EXPECTED_SOURCES)
        signals = [self._timeframe_signal(tf, frame) for tf, frame in bundle.enriched.items()]
        signals = [s for s in signals if s is not None]

        daily = bundle.enriched.get("1d", pd.DataFrame())
        last_price = bundle.last_price
        stance, alignment, score = _aggregate_stance(signals)

        stop, targets = _stop_and_targets(daily, last_price, score)
        patterns = _detect_patterns(bundle, daily)

        rs = (
            market_data.relative_strength(bundle.prices.get("1d", pd.DataFrame()), bundle.benchmark)
            if bundle.benchmark is not None
            else None
        )

        evidence: list[Evidence] = []
        for sig in signals:
            if sig.rsi_14 is not None:
                evidence.append(
                    Evidence(
                        claim=f"{sig.timeframe} RSI(14) at {sig.rsi_14:.1f} with trend {sig.trend}",
                        source="computed from OHLCV",
                        value=f"{sig.rsi_14:.1f}",
                        as_of=str(bundle.session.last_trading_day),
                    )
                )

        key_points = [
            f"{s.timeframe}: {s.trend}, signal {s.signal}"
            + (f", RSI {s.rsi_14:.0f}" if s.rsi_14 else "")
            for s in signals
        ]
        if rs is not None:
            key_points.append(
                f"60-session relative strength vs Nifty: {rs:+.1f} percentage points"
            )
        key_points.extend(patterns[:3])

        risks = _technical_risks(bundle, daily, signals)

        return TechnicalReport(
            role=self.role,
            symbol=bundle.ref.yf_symbol,
            stance=stance,
            confidence=confidence_from_quality(
                0.45 + 0.35 * abs(score), quality.completeness, penalties=len(risks) // 3
            ),
            summary=_technical_summary(bundle, signals, stance, alignment, last_price),
            reasoning=alignment,
            last_price=last_price,
            timeframes=signals,
            primary_timeframe="1d" if "1d" in bundle.enriched else (signals[0].timeframe if signals else "1d"),
            multi_timeframe_alignment=alignment,
            patterns=patterns,
            relative_strength_vs_nifty_pct=rs,
            suggested_stop_loss=stop,
            suggested_targets=targets,
            volatility_regime=ind.volatility_regime(daily),
            key_points=key_points[:10],
            risks=risks,
            what_would_change_my_mind=_invalidation_levels(daily, last_price, score),
            evidence=evidence[:6],
            data_quality=quality,
        )

    # -------------------------------------------------------------- pieces
    def _timeframe_signal(self, timeframe: str, frame: pd.DataFrame) -> TimeframeSignal | None:
        if frame.empty or len(frame) < 20:
            return None
        latest = ind.latest_values(frame)
        trend = ind.classify_trend(frame)
        supports, resistances = ind.swing_levels(frame)

        levels = LevelSet(supports=supports, resistances=resistances)
        if len(frame) > 1:
            prev = frame.iloc[-2]
            pivots = ind.classic_pivots(
                float(prev["high"]), float(prev["low"]), float(prev["close"])
            )
            levels.pivot = round(pivots["pivot"], 2)
            levels.r1, levels.r2 = round(pivots["r1"], 2), round(pivots["r2"], 2)
            levels.s1, levels.s2 = round(pivots["s1"], 2), round(pivots["s2"], 2)

        direction = latest.get("supertrend_dir")
        notes: list[str] = []
        rsi_val = latest.get("rsi_14")
        if rsi_val is not None:
            if rsi_val > 70:
                notes.append("RSI in overbought territory - momentum strong but entries are late")
            elif rsi_val < 30:
                notes.append("RSI oversold - bounce risk for shorts")
        adx_val = latest.get("adx")
        if adx_val is not None and adx_val < 20:
            notes.append("ADX below 20: range conditions, trend signals unreliable")
        vol_ratio = latest.get("volume_ratio_20")
        if vol_ratio is not None and vol_ratio > 2:
            notes.append(f"Volume {vol_ratio:.1f}x the 20-period average")

        return TimeframeSignal(
            timeframe=timeframe,
            trend=trend,
            close=latest.get("close") or float(frame["close"].iloc[-1]),
            rsi_14=_r(rsi_val),
            macd=_r(latest.get("macd")),
            macd_signal=_r(latest.get("macd_signal")),
            macd_histogram=_r(latest.get("macd_hist")),
            bb_upper=_r(latest.get("bb_upper")),
            bb_lower=_r(latest.get("bb_lower")),
            bb_percent_b=_r(latest.get("bb_percent_b"), 3),
            supertrend=_r(latest.get("supertrend")),
            supertrend_direction="long" if (direction or 0) > 0 else "short",
            vwap=_r(latest.get("vwap")),
            atr_14=_r(latest.get("atr_14")),
            atr_pct=_r(latest.get("atr_pct")),
            ema_20=_r(latest.get("ema_20")),
            ema_50=_r(latest.get("ema_50")),
            ema_200=_r(latest.get("ema_200")),
            adx_14=_r(latest.get("adx")),
            volume_ratio_20=_r(vol_ratio),
            levels=levels,
            signal=_signal_label(latest, trend),
            notes=notes,
        )


# ------------------------------------------------------------------ helpers
def _signal_label(latest: dict[str, float | None], trend: str) -> str:
    """Vote across the indicator set. Requires agreement, not a single trigger."""
    score = 0
    close = latest.get("close")
    if (latest.get("supertrend_dir") or 0) > 0:
        score += 1
    else:
        score -= 1
    macd_hist = latest.get("macd_hist")
    if macd_hist is not None:
        score += 1 if macd_hist > 0 else -1
    rsi_val = latest.get("rsi_14")
    if rsi_val is not None:
        score += 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
    ema20, vwap_val = latest.get("ema_20"), latest.get("vwap")
    if close and ema20:
        score += 1 if close > ema20 else -1
    if close and vwap_val:
        score += 1 if close > vwap_val else -1
    if trend == "uptrend":
        score += 1
    elif trend == "downtrend":
        score -= 1

    if score >= 3:
        return "bullish"
    if score <= -3:
        return "bearish"
    return "neutral"


def _aggregate_stance(signals: list[TimeframeSignal]) -> tuple[Stance, str, float]:
    """Combine timeframes, weighting the higher ones more heavily."""
    if not signals:
        return Stance.NEUTRAL, "No usable price history.", 0.0

    weights = {"1mo": 3.0, "1wk": 2.5, "1d": 2.0, "1h": 1.2, "60m": 1.2, "30m": 0.9,
               "15m": 0.8, "5m": 0.5, "1m": 0.3}
    total = score = 0.0
    for sig in signals:
        weight = weights.get(sig.timeframe, 1.0)
        value = {"bullish": 1.0, "bearish": -1.0}.get(sig.signal, 0.0)
        score += value * weight
        total += weight
    normalised = score / total if total else 0.0

    labels = {s.timeframe: s.signal for s in signals}
    distinct = set(labels.values())
    if len(distinct) == 1:
        alignment = f"All timeframes agree: {distinct.pop()}."
    else:
        parts = ", ".join(f"{tf} {sig}" for tf, sig in labels.items())
        alignment = (
            f"Timeframes disagree ({parts}). Conflicting timeframes argue for reduced "
            "size and waiting for the lower timeframe to align with the higher one."
        )

    if normalised >= 0.6:
        stance = Stance.BULLISH
    elif normalised >= 0.2:
        stance = Stance.MILDLY_BULLISH
    elif normalised <= -0.6:
        stance = Stance.BEARISH
    elif normalised <= -0.2:
        stance = Stance.MILDLY_BEARISH
    else:
        stance = Stance.NEUTRAL
    return stance, alignment, normalised


def _stop_and_targets(
    daily: pd.DataFrame, last_price: float | None, score: float
) -> tuple[float | None, list[float]]:
    """Structure-based stop widened by ATR, with R-multiple targets."""
    if daily.empty or last_price is None:
        return None, []
    latest = ind.latest_values(daily)
    atr = latest.get("atr_14") or last_price * 0.02
    supports, resistances = ind.swing_levels(daily)

    if score >= 0:
        structural = supports[0] if supports else last_price - 2 * atr
        stop = round(min(structural - 0.3 * atr, last_price - 1.2 * atr), 2)
        risk = max(last_price - stop, atr * 0.5)
        targets = [round(last_price + mult * risk, 2) for mult in (1.5, 2.5, 4.0)]
        # Snap the first target to the nearest real resistance if one sits closer.
        if resistances and resistances[0] > last_price:
            targets[0] = round(min(targets[0], resistances[0]), 2)
    else:
        structural = resistances[0] if resistances else last_price + 2 * atr
        stop = round(max(structural + 0.3 * atr, last_price + 1.2 * atr), 2)
        risk = max(stop - last_price, atr * 0.5)
        targets = [round(last_price - mult * risk, 2) for mult in (1.5, 2.5, 4.0)]
        if supports and supports[0] < last_price:
            targets[0] = round(max(targets[0], supports[0]), 2)
    return stop, targets


def _detect_patterns(bundle: MarketBundle, daily: pd.DataFrame) -> list[str]:
    patterns: list[str] = []
    if daily.empty or len(daily) < 25:
        return patterns

    latest = ind.latest_values(daily)
    close = float(daily["close"].iloc[-1])
    prev_close = float(daily["close"].iloc[-2])
    open_ = float(daily["open"].iloc[-1])

    # Gap analysis - the defining feature of the Indian open.
    gap_pct = (open_ - prev_close) / prev_close * 100 if prev_close else 0
    if abs(gap_pct) >= 1.0:
        direction = "up" if gap_pct > 0 else "down"
        filled = (close <= prev_close) if gap_pct > 0 else (close >= prev_close)
        patterns.append(
            f"Gap-{direction} open of {gap_pct:+.2f}% driven by overnight cues"
            + (" — gap filled during the session, which weakens the signal"
               if filled else " — gap held, which confirms it")
        )

    ema20, ema50, ema200 = latest.get("ema_20"), latest.get("ema_50"), latest.get("ema_200")
    if ema20 and ema50 and ema200:
        if ema20 > ema50 > ema200:
            patterns.append("Bullish EMA stack (20 > 50 > 200)")
        elif ema20 < ema50 < ema200:
            patterns.append("Bearish EMA stack (20 < 50 < 200)")

    bb_width = latest.get("bb_width")
    if bb_width is not None:
        history = daily["bb_width"].dropna().tail(120)
        if len(history) > 30 and bb_width <= float(history.quantile(0.15)):
            patterns.append(
                "Bollinger squeeze: volatility at the low end of its 6-month range, "
                "which typically precedes an expansion move"
            )

    high_52w = float(daily["close"].tail(250).max())
    low_52w = float(daily["close"].tail(250).min())
    if close >= high_52w * 0.99:
        patterns.append("Trading within 1% of the 52-week high")
    elif close <= low_52w * 1.01:
        patterns.append("Trading within 1% of the 52-week low")

    vol_ratio = latest.get("volume_ratio_20")
    if vol_ratio and vol_ratio > 2.5:
        patterns.append(f"Volume spike at {vol_ratio:.1f}x the 20-day average")

    # Circuit-driven moves masquerading as momentum.
    band = bundle.micro.circuit_band_pct
    if band:
        moves = (daily["close"].pct_change().abs() * 100).tail(10)
        at_band = int((moves >= band - 0.2).sum())
        if at_band >= 2:
            patterns.append(
                f"{at_band} circuit-band moves in the last 10 sessions - price discovery is "
                "constrained, so levels and stops may be unfillable"
            )

    if bundle.micro.is_expiry_day:
        patterns.append(
            "Expiry day: intraday price tends to pin toward high open-interest strikes"
        )
    elif (bundle.micro.days_to_expiry or 99) <= 2:
        patterns.append(
            f"Expiry week ({bundle.micro.days_to_expiry} sessions out): expect pinning and "
            "unreliable intraday trends"
        )
    return patterns[:8]


def _technical_risks(
    bundle: MarketBundle, daily: pd.DataFrame, signals: list[TimeframeSignal]
) -> list[str]:
    risks: list[str] = []
    if bundle.micro.is_circuit_locked:
        risks.append("Scrip is circuit-locked: technical levels are not executable right now")
    if bundle.micro.in_fno_ban:
        risks.append("In F&O ban period - only position-reducing derivative orders are allowed")
    regime = ind.volatility_regime(daily)
    if regime in ("high", "extreme"):
        risks.append(f"Volatility regime is {regime}: stops must be wider and size smaller")
    if len({s.signal for s in signals}) > 1:
        risks.append("Timeframes disagree, so any entry is a lower-conviction trade")
    if not daily.empty and len(daily) < 200:
        risks.append("Under 200 sessions of history - long-term averages are not yet meaningful")
    turnover = bundle.micro.avg_daily_turnover_inr
    if turnover is not None and turnover < 5_00_00_000:
        risks.append(
            f"Thin traded value (avg Rs {turnover / 1e7:.1f} cr/day): slippage will exceed "
            "the modelled entry"
        )
    return risks[:6]


def _invalidation_levels(
    daily: pd.DataFrame, last_price: float | None, score: float
) -> list[str]:
    if daily.empty or last_price is None:
        return ["Price history unavailable, so no level-based invalidation can be stated"]
    latest = ind.latest_values(daily)
    supertrend_line = latest.get("supertrend")
    ema50 = latest.get("ema_50")
    out: list[str] = []
    if supertrend_line:
        side = "below" if score >= 0 else "above"
        out.append(f"A daily close {side} the Supertrend line at {supertrend_line:,.2f}")
    if ema50:
        side = "below" if score >= 0 else "above"
        out.append(f"Loss of the 50-EMA at {ema50:,.2f} on a closing basis")
    out.append(
        "Two consecutive sessions against the thesis on above-average volume"
    )
    return out[:4]


def _technical_summary(
    bundle: MarketBundle,
    signals: list[TimeframeSignal],
    stance: Stance,
    alignment: str,
    last_price: float | None,
) -> str:
    if not signals:
        return f"No usable price history for {bundle.ref.base_symbol}; no technical view."
    primary = next((s for s in signals if s.timeframe == "1d"), signals[0])
    bits = [
        f"{bundle.ref.base_symbol} at Rs {last_price:,.2f}" if last_price else bundle.ref.base_symbol,
        f"is in a {primary.trend} on the {primary.timeframe} timeframe",
    ]
    if primary.rsi_14:
        bits.append(f"with RSI at {primary.rsi_14:.0f}")
    if primary.supertrend:
        bits.append(
            f"and Supertrend at {primary.supertrend:,.2f} pointing "
            f"{primary.supertrend_direction}"
        )
    return f"{' '.join(bits)}. {alignment} Technical stance: {stance.value}."


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
