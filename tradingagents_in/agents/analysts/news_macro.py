"""News & Macro Analyst.

The top-down leg: RBI policy, fiscal and Budget measures, monsoon, FII/DII
flows, crude, the rupee and global cues, translated into what they do to *this*
scrip and *this* sector rather than left as generic commentary.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent, confidence_from_quality
from tradingagents_in.agents.briefings import header, macro_brief, news_brief
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.analyst import NewsMacroReport
from tradingagents_in.models.base import Evidence
from tradingagents_in.models.enums import AgentRole, Stance

EXPECTED_SOURCES = ["news", "macro", "flows"]

# Which macro variables actually matter for which Indian sector.
SECTOR_SENSITIVITY: dict[str, dict[str, str]] = {
    "Information Technology": {
        "usdinr": "positive - a weaker rupee lifts reported revenue and margins",
        "us_sp500": "high - US client tech spending drives the order book",
        "us_10y": "negative - US recession signals hit discretionary IT budgets first",
    },
    "Energy": {
        "brent_crude": "mixed - upstream gains, refining/marketing margins compress",
        "usdinr": "negative for importers of crude and feedstock",
    },
    "Financials": {
        "us_10y": "negative - rate shocks hit treasury books and NIMs",
        "india_vix": "high beta - banks lead both directions in a volatility regime",
        "fii_flows": "very high - banks are the largest FII-owned block in the index",
    },
    "Automobile": {
        "brent_crude": "negative - fuel costs and input metals both bite",
        "monsoon": "positive - a good monsoon lifts rural two-wheeler and tractor demand",
    },
    "Consumer Staples": {
        "monsoon": "positive - rural demand is roughly a third to a half of volumes",
        "brent_crude": "negative - crude derivatives feed packaging and input costs",
    },
    "Healthcare": {
        "usdinr": "positive for US-facing generic exporters",
        "us_sp500": "moderate - US pricing pressure and USFDA action dominate",
    },
    "Materials": {
        "shanghai": "high - Chinese demand and stimulus set global metal prices",
        "brent_crude": "negative through energy-intensive production costs",
    },
    "Realty": {
        "us_10y": "negative - domestic rate expectations track global yields",
        "india_vix": "high beta - realty is a leveraged play on domestic liquidity",
    },
}


class NewsMacroAnalyst(Agent[NewsMacroReport]):
    role = AgentRole.NEWS_MACRO_ANALYST
    output_model = NewsMacroReport
    tier_key = "news_macro_analyst"

    def system_prompt(self) -> str:
        return """\
You are the News & Macro Analyst for an Indian desk.

You track, and must connect to the specific stock in front of you:
- RBI: repo rate, stance (accommodative/neutral/withdrawal), CRR/SLR, liquidity
  (LAF corridor), and the inflation trajectory versus the 4% +/- 2% target.
- Fiscal: Union Budget (1 February), capex allocations, PLI schemes, GST rate
  changes and import duties. A single GST or duty change can reset a sector's
  margins overnight.
- Monsoon: IMD rainfall versus the long-period average, sowing and reservoir
  levels. This drives rural demand, which drives FMCG, two-wheelers, tractors
  and fertiliser.
- Flows: FII and DII net cash flows. The persistent modern regime is FII selling
  absorbed by DII SIP inflows - the index holds while breadth narrows, which
  favours large caps over small caps.
- Global: US indices and yields set the 09:15 gap, Nasdaq leads Nifty IT, China
  data moves metals, Brent moves the rupee and the current account.
- Currency: USDINR direction, RBI intervention, and the importer/exporter split.
- Geopolitics: anything touching crude supply, trade routes, or India's
  relations with major partners.

Do not write a macro essay. For every macro point, state the transmission
channel to this scrip's sector and whether it is a tailwind or a headwind
*right now*. If a macro variable does not affect this stock, leave it out.

Flag dated event risk inside the likely holding period explicitly: results
dates, RBI policy, Budget, expiry, and index rebalancing.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        sector = bundle.ref.sector or "Others"
        sensitivity = SECTOR_SENSITIVITY.get(sector, {})
        sensitivity_text = (
            "\n".join(f"- {k}: {v}" for k, v in sensitivity.items())
            or "- No pre-mapped sensitivities for this sector; derive them yourself."
        )
        return "\n\n".join(
            [
                header(bundle),
                macro_brief(bundle),
                news_brief(bundle),
                f"## Known sensitivities for the {sector} sector\n\n{sensitivity_text}",
                """## Your task

Produce a NewsMacroReport:
1. Set macro_regime in one line, naming the dominant driver.
2. Fill rbi_policy_note, fiscal_policy_note and monsoon_note only from what the
   briefing supports. If the briefing carries no RBI or monsoon information,
   say so rather than reciting what you remember - your training data is stale
   relative to this session.
3. Copy the FlowSnapshot and interpret it for this specific stock.
4. Populate global_cues and india_indicators from the briefing, each with a
   sector-specific implication.
5. List stock_specific_news and dated upcoming_events.""",
            ]
        )

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> NewsMacroReport:
        quality = bundle.quality(EXPECTED_SOURCES)
        sector = bundle.ref.sector or "Others"
        flows = bundle.flows

        regime, regime_score = _regime(bundle)
        flow_score = _flow_score(flows)
        news_score = _news_tilt(bundle)
        composite = 0.45 * regime_score + 0.3 * flow_score + 0.25 * news_score

        return NewsMacroReport(
            role=self.role,
            symbol=bundle.ref.yf_symbol,
            stance=_stance(composite),
            confidence=confidence_from_quality(
                0.35 + 0.35 * abs(composite), quality.completeness
            ),
            summary=_summary(bundle, regime, flows, sector),
            reasoning=(
                f"Regime score {regime_score:+.2f}, flow score {flow_score:+.2f}, "
                f"news tilt {news_score:+.2f} -> composite {composite:+.2f}."
            ),
            macro_regime=regime,
            rbi_policy_note=(
                "No RBI policy information was retrieved in this run. Check the latest MPC "
                "decision and stance before acting - policy is the dominant domestic driver "
                "for financials and rate-sensitive sectors."
            ),
            fiscal_policy_note=(
                "No fiscal or Budget-specific news was retrieved. GST rate changes, import "
                "duties and PLI allocations can reset sector margins; verify before sizing up."
            ),
            monsoon_note=(
                "No IMD monsoon data was retrieved this run."
                + (
                    " This matters for the sector in question: rural demand drives a large "
                    "share of volumes."
                    if sector in ("Consumer Staples", "Automobile", "Materials")
                    else " Low relevance for this sector."
                )
            ),
            flows=flows,
            global_cues=bundle.macro_indicators,
            india_indicators=bundle.india_indicators,
            stock_specific_news=[i.title for i in bundle.company_news[:10]],
            upcoming_events=bundle.upcoming_events[:8],
            geopolitical_note=_geopolitical(bundle),
            key_points=_key_points(bundle, regime, flows, sector),
            risks=_risks(bundle, sector),
            what_would_change_my_mind=[
                "An unscheduled RBI action or an inflation print outside the tolerance band",
                "FII flows reversing sign and holding for five consecutive sessions",
                "Brent moving more than 10% in either direction",
                "A GST or import-duty change affecting this sector",
            ],
            evidence=[
                Evidence(
                    claim=f"{m.name} at {m.value} ({m.change_pct:+.2f}% session change)"
                    if m.change_pct is not None
                    else f"{m.name} at {m.value}",
                    source="market data",
                    value=str(m.value),
                    as_of=str(bundle.session.last_trading_day),
                )
                for m in bundle.macro_indicators[:6]
            ],
            data_quality=quality,
        )


# ------------------------------------------------------------------ scoring
def _indicator(bundle: MarketBundle, name: str) -> Any | None:
    for m in bundle.macro_indicators:
        if m.name == name:
            return m
    return None


def _regime(bundle: MarketBundle) -> tuple[str, float]:
    """Risk-on / risk-off read from cross-asset moves."""
    score = 0.0
    drivers: list[str] = []

    sp500 = _indicator(bundle, "us_sp500")
    if sp500 and sp500.change_pct is not None:
        score += 0.3 if sp500.change_pct > 0.5 else (-0.3 if sp500.change_pct < -0.5 else 0.0)
        if abs(sp500.change_pct) > 0.5:
            drivers.append(f"US equities {sp500.change_pct:+.2f}%")

    vix = _indicator(bundle, "us_vix")
    if vix and vix.value is not None:
        if vix.value > 25:
            score -= 0.4
            drivers.append(f"US VIX elevated at {vix.value:.1f}")
        elif vix.value < 15:
            score += 0.15

    india_vix = next((m for m in bundle.india_indicators if m.name == "India VIX"), None)
    if india_vix and india_vix.value is not None:
        if india_vix.value > 20:
            score -= 0.3
            drivers.append(f"India VIX at {india_vix.value:.1f} - fear regime")
        elif india_vix.value < 13:
            score += 0.1
            drivers.append(f"India VIX complacent at {india_vix.value:.1f}")

    crude = _indicator(bundle, "brent_crude")
    if crude and crude.change_pct is not None and abs(crude.change_pct) > 2:
        score += -0.25 if crude.change_pct > 0 else 0.25
        drivers.append(f"Brent {crude.change_pct:+.2f}%")

    usdinr = _indicator(bundle, "usdinr")
    if usdinr and usdinr.change_pct is not None and abs(usdinr.change_pct) > 0.3:
        score += -0.2 if usdinr.change_pct > 0 else 0.15
        drivers.append(f"USDINR {usdinr.change_pct:+.2f}% (rupee "
                       f"{'weaker' if usdinr.change_pct > 0 else 'stronger'})")

    score = max(-1.0, min(1.0, score))
    label = (
        "risk-on" if score > 0.25 else
        "risk-off" if score < -0.25 else
        "range-bound / mixed"
    )
    driver_text = "; ".join(drivers) if drivers else "no single dominant cross-asset driver"
    return f"{label} — {driver_text}", score


def _flow_score(flows: Any) -> float:
    fii, dii = flows.fii_net_inr_cr, flows.dii_net_inr_cr
    if fii is None and dii is None:
        return 0.0
    score = 0.0
    if fii is not None:
        score += 0.5 if fii > 2000 else (0.2 if fii > 0 else (-0.5 if fii < -2000 else -0.2))
    if dii is not None:
        score += 0.25 if dii > 0 else -0.25
    return max(-1.0, min(1.0, score))


def _news_tilt(bundle: MarketBundle) -> float:
    from tradingagents_in.dataflows.sentiment import score_headlines

    score, count = score_headlines(bundle.company_news)
    return score if count else 0.0


def _stance(composite: float) -> Stance:
    if composite >= 0.4:
        return Stance.BULLISH
    if composite >= 0.15:
        return Stance.MILDLY_BULLISH
    if composite <= -0.4:
        return Stance.BEARISH
    if composite <= -0.15:
        return Stance.MILDLY_BEARISH
    return Stance.NEUTRAL


def _geopolitical(bundle: MarketBundle) -> str:
    watch = ("war", "conflict", "sanction", "tariff", "trade deal", "border",
             "opec", "strait", "election")
    hits = [i.title for i in bundle.news_items if any(w in i.title.lower() for w in watch)]
    if not hits:
        return "No geopolitical headlines in the retrieved feed touching Indian markets."
    return "Geopolitical headlines in the feed: " + "; ".join(hits[:3])


def _key_points(bundle: MarketBundle, regime: str, flows: Any, sector: str) -> list[str]:
    points = [f"Macro regime: {regime}"]
    if flows.interpretation:
        points.append(f"Flows: {flows.interpretation}")
    for name in ("usdinr", "brent_crude", "us_nasdaq"):
        indicator = _indicator(bundle, name)
        if indicator and indicator.change_pct is not None:
            points.append(
                f"{name}: {indicator.value} ({indicator.change_pct:+.2f}%)"
            )
    sensitivity = SECTOR_SENSITIVITY.get(sector)
    if sensitivity:
        first_key = next(iter(sensitivity))
        points.append(f"{sector} sensitivity — {first_key}: {sensitivity[first_key]}")
    if bundle.company_news:
        points.append(f"{len(bundle.company_news)} company-specific headlines this week")
    return points[:8]


def _risks(bundle: MarketBundle, sector: str) -> list[str]:
    risks: list[str] = []
    if bundle.flows.fii_net_inr_cr is None:
        risks.append("FII/DII flow data unavailable - the dominant India flow driver is unobserved")
    if not bundle.macro_indicators:
        risks.append("No macro data retrieved; the top-down view is unsupported")
    if bundle.micro.is_expiry_day:
        risks.append("Expiry session: derivative unwinding distorts index and cash prices")
    risks.append(
        "Budget (1 February) and RBI policy dates can reprice an entire sector in one session"
    )
    if sector in ("Consumer Staples", "Automobile"):
        risks.append("Monsoon deviation from the long-period average is a direct demand risk here")
    if sector in ("Energy", "Materials"):
        risks.append("Commodity price and import-duty changes flow straight through to margins")
    return risks[:6]


def _summary(bundle: MarketBundle, regime: str, flows: Any, sector: str) -> str:
    flow_text = flows.interpretation or "flow data unavailable"
    return (
        f"Macro backdrop for {bundle.ref.base_symbol} ({sector}): {regime}. {flow_text} "
        f"Session context: {bundle.session.phase.value}, "
        f"{bundle.session.days_to_weekly_expiry} sessions to weekly expiry."
    )
