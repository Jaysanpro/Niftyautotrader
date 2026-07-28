"""Render a :class:`MarketBundle` into the markdown briefings agents read.

Keeping this separate from the agents matters: the bull and the bear get byte
-identical briefings, so any difference in their conclusions comes from their
mandate, not from being shown different data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingagents_in.dataflows import indicators as ind
from tradingagents_in.dataflows import market_data
from tradingagents_in.dataflows.corporate_actions import adjustment_warning, summarise
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.utils.formatting import bullet_list, inr, num, pct


def header(bundle: MarketBundle) -> str:
    ref = bundle.ref
    quote = bundle.quote
    lines = [
        f"# {ref.name or ref.base_symbol} ({ref.yf_symbol}) - {ref.exchange.value}",
        f"- Sector: {ref.sector or 'unknown'}"
        f" | F&O eligible: {'yes' if ref.fno_eligible else 'no'}"
        + (f" (lot {ref.lot_size})" if ref.lot_size else ""),
        f"- Last price: Rs {num(bundle.last_price)}"
        f" | Session change: {pct(quote.get('change_pct'))}",
        f"- Day range: {num(quote.get('day_low'))} - {num(quote.get('day_high'))}"
        f" | 52w: {num(quote.get('year_low'))} - {num(quote.get('year_high'))}",
        "",
        "## Session context (Asia/Kolkata)",
        bundle.session.to_markdown(),
    ]
    return "\n".join(lines)


def technical_brief(bundle: MarketBundle) -> str:
    if not bundle.enriched:
        return "## Technicals\n\n(no price history available)"

    blocks = ["## Technicals"]
    for tf, frame in bundle.enriched.items():
        latest = ind.latest_values(frame)
        trend = ind.classify_trend(frame)
        supports, resistances = ind.swing_levels(frame)
        blocks.append(
            "\n".join(
                [
                    f"\n### Timeframe {tf} ({len(frame)} bars) - trend: {trend}",
                    f"- Close {num(latest.get('close') or frame['close'].iloc[-1])}"
                    f" | EMA20 {num(latest.get('ema_20'))}"
                    f" | EMA50 {num(latest.get('ema_50'))}"
                    f" | EMA200 {num(latest.get('ema_200'))}",
                    f"- RSI(14) {num(latest.get('rsi_14'))}"
                    f" | ADX(14) {num(latest.get('adx'))}"
                    f" | +DI {num(latest.get('plus_di'))} / -DI {num(latest.get('minus_di'))}",
                    f"- MACD {num(latest.get('macd'))}"
                    f" signal {num(latest.get('macd_signal'))}"
                    f" hist {num(latest.get('macd_hist'))}",
                    f"- Bollinger: upper {num(latest.get('bb_upper'))}"
                    f" lower {num(latest.get('bb_lower'))}"
                    f" %B {num(latest.get('bb_percent_b'), 3)}"
                    f" width {num(latest.get('bb_width'), 4)}",
                    f"- Supertrend {num(latest.get('supertrend'))}"
                    f" (direction {'LONG' if (latest.get('supertrend_dir') or 0) > 0 else 'SHORT'})",
                    f"- VWAP {num(latest.get('vwap'))}"
                    f" | ATR(14) {num(latest.get('atr_14'))} ({num(latest.get('atr_pct'))}% of price)",
                    f"- Volume vs 20-period avg: {num(latest.get('volume_ratio_20'))}x",
                    f"- Swing supports: {supports or 'none nearby'}",
                    f"- Swing resistances: {resistances or 'none nearby'}",
                ]
            )
        )

    daily = bundle.prices.get("1d")
    if daily is not None and len(daily) > 1:
        prev = daily.iloc[-2]
        pivots = ind.classic_pivots(float(prev["high"]), float(prev["low"]), float(prev["close"]))
        cpr = ind.cpr(float(prev["high"]), float(prev["low"]), float(prev["close"]))
        blocks.append(
            "\n### Levels from the previous session\n"
            f"- Pivot {num(pivots['pivot'])} | R1 {num(pivots['r1'])} R2 {num(pivots['r2'])}"
            f" | S1 {num(pivots['s1'])} S2 {num(pivots['s2'])}\n"
            f"- CPR: {num(cpr['cpr_bottom'])} - {num(cpr['cpr_top'])}"
            f" (width {num(cpr['cpr_width_pct'])}% - narrow CPR often precedes a trending day)"
        )

    if bundle.benchmark is not None and daily is not None:
        rs = market_data.relative_strength(daily, bundle.benchmark, 60)
        vol = market_data.realised_volatility_pct(daily)
        dd = market_data.drawdown_from_high(daily)
        blocks.append(
            "\n### Relative and risk stats\n"
            f"- 60-session relative strength vs Nifty: {pct(rs)}\n"
            f"- Realised volatility (20d, annualised): {num(vol)}%\n"
            f"- Drawdown from 52-week high: {pct(dd)}\n"
            f"- Volatility regime: {ind.volatility_regime(bundle.enriched.get('1d', pd.DataFrame()))}"
        )
    return "\n".join(blocks)


def fundamentals_brief(bundle: MarketBundle) -> str:
    f = bundle.fundamentals
    if not f:
        return "## Fundamentals\n\n(not applicable or unavailable)"

    lines = [
        "## Fundamentals (Ind-AS, consolidated where reported)",
        f"- Market cap: {inr(f.get('market_cap_inr'))}"
        f" | Sector: {f.get('sector') or bundle.ref.sector}"
        f" | Industry: {f.get('industry') or 'n/a'}",
        f"- P/E {num(f.get('pe'))} (fwd {num(f.get('forward_pe'))})"
        f" | P/B {num(f.get('pb'))}"
        f" | EV/EBITDA {num(f.get('ev_to_ebitda'))}"
        f" | Div yield {num(f.get('dividend_yield_pct'))}%",
        f"- ROE {num(f.get('roe_pct'))}%"
        f" | ROCE {num(f.get('roce_pct'))}%"
        f" | Operating margin {num(f.get('operating_margin_pct'))}%",
        f"- Debt/Equity {num(f.get('debt_to_equity'))}x"
        f" | Interest coverage {num(f.get('interest_coverage'))}x"
        f" | Current ratio {num(f.get('current_ratio'))}",
        f"- Net debt {inr(f.get('net_debt_inr'))}"
        f" | OCF {inr(f.get('operating_cash_flow_inr'))}"
        f" | FCF {inr(f.get('free_cash_flow_inr'))}",
        f"- Cash conversion (OCF/PAT): {num(f.get('ocf_to_pat'))}"
        " — sustained below 0.7 is an earnings-quality flag in Indian filings",
        f"- 3Y sales CAGR {num(f.get('sales_growth_3y_cagr_pct'))}%"
        f" | 3Y profit CAGR {num(f.get('profit_growth_3y_cagr_pct'))}%",
        f"- Beta {num(f.get('beta'))} | Book value {num(f.get('book_value'))}",
    ]

    quarters = f.get("quarters") or []
    if quarters:
        lines.append("\n### Recent reported quarters (newest first)")
        for q in quarters[:4]:
            lines.append(
                f"- {q.get('period')}: revenue {inr(q.get('revenue_inr'))}"
                f" ({pct(q.get('revenue_yoy_pct'))} YoY),"
                f" PAT {inr(q.get('pat_inr'))} ({pct(q.get('pat_yoy_pct'))} YoY),"
                f" EBITDA margin {num(q.get('ebitda_margin_pct'))}%"
            )

    lines.append("\n### Shareholding")
    lines.append(
        f"- Promoter/insider holding (proxy): {num(f.get('held_by_insiders_pct'))}%"
        f" | Institutions: {num(f.get('held_by_institutions_pct'))}%"
    )
    lines.append(
        "- Note: the free feed reports an insider-holding proxy, not the SEBI-filed "
        "promoter category, and carries no pledge data. Treat promoter/pledge figures "
        "as unverified unless a filing confirms them."
    )

    if bundle.peers:
        lines.append("\n### Sector peers")
        lines.append(
            "| Symbol | P/E | P/B | ROE% | D/E | Mkt cap |\n|---|---|---|---|---|---|"
        )
        for p in bundle.peers:
            lines.append(
                f"| {p.symbol} | {num(p.pe)} | {num(p.pb)} | {num(p.roe_pct)} "
                f"| {num(p.debt_to_equity)} | {inr(p.market_cap_inr)} |"
            )
        pes = [p.pe for p in bundle.peers if p.pe]
        if pes and f.get("pe"):
            median = sorted(pes)[len(pes) // 2]
            delta = (f["pe"] - median) / median * 100
            lines.append(
                f"\n- Trades at a {delta:+.1f}% P/E {'premium' if delta > 0 else 'discount'} "
                f"to the peer median of {median:.1f}x."
            )

    lines.append(f"\n- Data sources: {', '.join(f.get('sources', [])) or 'none'}")
    return "\n".join(lines)


def news_brief(bundle: MarketBundle) -> str:
    lines = ["## News flow"]
    if bundle.company_news:
        lines.append(f"\n### Company-specific ({len(bundle.company_news)} items)")
        lines.extend(f"- {item}" for item in bundle.company_news[:12])
    else:
        lines.append("\n### Company-specific\n- (no headlines matched this scrip this week)")

    if bundle.announcements:
        lines.append("\n### Exchange filings")
        lines.extend(
            f"- {a.get('date')}: {a.get('subject')}" for a in bundle.announcements[:8]
        )

    lines.append("\n### Broader Indian market headlines")
    lines.extend(f"- {item}" for item in bundle.news_items[:12])
    return "\n".join(lines)


def macro_brief(bundle: MarketBundle) -> str:
    lines = ["## Macro and flows"]

    if bundle.macro_indicators:
        lines.append("\n### Global and cross-asset cues")
        for m in bundle.macro_indicators:
            lines.append(
                f"- **{m.name}**: {num(m.value)} ({pct(m.change_pct)})"
                + (f" — {m.implication_for_india}" if m.implication_for_india else "")
            )

    if bundle.india_indicators:
        lines.append("\n### India gauges")
        for m in bundle.india_indicators:
            lines.append(
                f"- **{m.name}**: {num(m.value)} ({m.direction}) — {m.implication_for_india}"
            )

    flows = bundle.flows
    lines.append("\n### FII / DII cash-segment flows")
    lines.append(
        f"- Date: {flows.date or 'n/a'}"
        f" | FII net: {num(flows.fii_net_inr_cr)} cr"
        f" | DII net: {num(flows.dii_net_inr_cr)} cr"
    )
    if flows.fii_5d_net_inr_cr is not None:
        lines.append(
            f"- 5-session: FII {num(flows.fii_5d_net_inr_cr)} cr,"
            f" DII {num(flows.dii_5d_net_inr_cr)} cr"
        )
    lines.append(f"- Read: {flows.interpretation or 'unavailable'}")

    if bundle.upcoming_events:
        lines.append("\n### Event calendar")
        lines.append(bullet_list(bundle.upcoming_events, limit=12))
    return "\n".join(lines)


def sentiment_brief(bundle: MarketBundle) -> str:
    from tradingagents_in.dataflows.sentiment import aggregate, buzz_level

    lines = ["## Sentiment"]
    if not bundle.sentiment_sources:
        return "## Sentiment\n\n(no sentiment sources returned data - treat sentiment as unknown)"

    score = aggregate(bundle.sentiment_sources)
    lines.append(
        f"- Aggregate score: {score:+.3f} on [-1, 1]"
        f" | Buzz: {buzz_level(bundle.sentiment_sources)}"
    )
    lines.append("\n### By source")
    for src in bundle.sentiment_sources:
        lines.append(
            f"- **{src.source}**: {src.score:+.3f} across {src.volume} items"
            + (f" — {src.notes}" if src.notes else "")
        )
    if bundle.company_news:
        lines.append("\n### Headlines behind the score")
        lines.extend(f"- {item.title} ({item.source})" for item in bundle.company_news[:10])
    return "\n".join(lines)


def microstructure_brief(bundle: MarketBundle) -> str:
    m = bundle.micro
    lines = [
        "## Tradability and market microstructure",
        f"- Market phase: {m.market_phase}",
        f"- Circuit band: {num(m.circuit_band_pct)}%"
        + (
            f" (upper {num(m.upper_circuit)}, lower {num(m.lower_circuit)})"
            if m.upper_circuit
            else " — F&O scrip, no fixed daily band"
        ),
        f"- Circuit locked right now: {'YES' if m.is_circuit_locked else 'no'}",
        f"- Surveillance status: {m.surveillance.value}",
        f"- In F&O ban period: {'YES' if m.in_fno_ban else 'no'}",
        f"- Avg daily turnover (20d): {inr(m.avg_daily_turnover_inr)}",
        f"- Avg daily volume (20d): {num(m.avg_daily_volume, 0)} shares",
        f"- Expiry day: {'YES' if m.is_expiry_day else 'no'}"
        f" | Sessions to weekly expiry: {m.days_to_expiry}",
    ]
    if m.index_circuit_note:
        lines.append(f"- Index note: {m.index_circuit_note}")

    if bundle.option_chain:
        chain = bundle.option_chain
        lines.append("\n### Option chain")
        lines.append(
            f"- Expiry {chain.expiry} | spot {num(chain.spot)}"
            f" | ATM strike {num(chain.atm_strike())}"
        )
        lines.append(
            f"- PCR (OI): {num(chain.pcr_oi(), 3)} | Max pain: {num(chain.max_pain())}"
        )
    return "\n".join(lines)


def corporate_actions_brief(bundle: MarketBundle) -> str:
    lines = ["## Corporate actions", summarise(bundle.corporate_actions)]
    warning = adjustment_warning(bundle.corporate_actions)
    if warning:
        lines.append(f"\n**Adjustment warning:** {warning}")
    return "\n".join(lines)


def full_briefing(bundle: MarketBundle) -> str:
    """The complete data pack. Analysts get slices; researchers get all of it."""
    return "\n\n".join(
        [
            header(bundle),
            technical_brief(bundle),
            fundamentals_brief(bundle),
            news_brief(bundle),
            macro_brief(bundle),
            sentiment_brief(bundle),
            corporate_actions_brief(bundle),
            microstructure_brief(bundle),
            data_quality_brief(bundle),
        ]
    )


def data_quality_brief(bundle: MarketBundle) -> str:
    return "\n".join(
        [
            "## Data quality",
            f"- Sources retrieved: {', '.join(bundle.sources_used) or 'none'}",
            f"- Sources that failed: {', '.join(bundle.sources_failed) or 'none'}",
            f"- Gather time: {bundle.elapsed_seconds}s",
            "- Anything not listed as retrieved is genuinely unknown. Say so rather than "
            "estimating it, and lower your confidence accordingly.",
        ]
    )


def analyst_reports_brief(state: dict[str, Any]) -> str:
    """The four analyst reports, as the research team sees them."""
    keys = [
        ("fundamentals_report", "Fundamentals Analyst"),
        ("sentiment_report", "Sentiment Analyst"),
        ("news_macro_report", "News & Macro Analyst"),
        ("technical_report", "Technical Analyst"),
    ]
    blocks: list[str] = []
    for key, label in keys:
        report = state.get(key)
        if report is None:
            blocks.append(f"### {label}\n\n(did not run)")
            continue
        blocks.append(f"### {label}\n\n{report.to_markdown()}")
    return "\n\n".join(blocks)
