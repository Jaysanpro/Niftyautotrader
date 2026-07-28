"""Fundamentals Analyst - Ind-AS flavoured.

Judges the business, not the chart: earnings quality, balance-sheet strength,
promoter behaviour and valuation against Indian sector peers. Deliberately
sceptical about the things that go wrong in Indian mid caps - promoter pledging,
receivable-driven "profits", related-party leakage and auditor churn.
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent, confidence_from_quality
from tradingagents_in.agents.briefings import corporate_actions_brief, fundamentals_brief, header
from tradingagents_in.dataflows.fundamentals import build_quarters, build_shareholding
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.analyst import FundamentalsReport
from tradingagents_in.models.base import Evidence
from tradingagents_in.models.enums import AgentRole, Stance

EXPECTED_SOURCES = ["fundamentals", "peers", "corporate_actions"]

# Thresholds that mark a genuinely stretched Indian balance sheet.
DE_WARNING = 1.5
DE_SEVERE = 2.5
INTEREST_COVER_WARNING = 3.0
OCF_PAT_WARNING = 0.7


class FundamentalsAnalyst(Agent[FundamentalsReport]):
    role = AgentRole.FUNDAMENTALS_ANALYST
    output_model = FundamentalsReport
    tier_key = "fundamentals_analyst"

    def system_prompt(self) -> str:
        return """\
You are the Fundamentals Analyst on an Indian equity desk.

You read Indian filings under Ind-AS. That means:
- Consolidated numbers are the default view; standalone figures hide subsidiary
  leverage. Say which basis you are using.
- "Other income" and "exceptional items" routinely flatter Indian PAT. Strip
  them mentally before calling a quarter good.
- Cash conversion is the lie detector. Sustained OCF/PAT below 0.7 with rising
  receivables means the profit is on paper.
- Promoter holding and pledged shares matter more here than insider ownership
  does in Western markets. Falling promoter stake or rising pledge is a serious
  warning regardless of how good the P&L looks.
- ROCE is the primary quality metric Indian investors screen on; ROE flattered
  by leverage is not quality.
- Check for auditor resignation, CFO churn, related-party transactions,
  frequent equity dilution (QIP/preferential), and SEBI or tax actions.

On valuation: compare to the peer set and to the company's own history, and
quote the multiple. "Expensive" without a number is not a view. An Indian
quality large cap sustaining a premium multiple for a decade is not a short
just because the P/E is high.

Be explicit where data is missing. The free data feed does not carry SEBI-filed
promoter or pledge figures - if they are absent, say the governance picture is
unverified rather than assuming it is clean.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        return "\n\n".join(
            [
                header(bundle),
                fundamentals_brief(bundle),
                corporate_actions_brief(bundle),
                """## Your task

Produce a FundamentalsReport:
1. Copy the ratio fields from the briefing. Leave anything absent as null.
2. Write a valuation_verdict that names the multiple and the comparison
   ("22x forward vs a peer median of 18x and a 5-year mean of 19x").
3. List earnings_quality_notes based on cash conversion, margin trend and the
   quality of recent quarters.
4. List governance_flags. If the data to assess governance is missing, say so
   explicitly as a flag rather than leaving the list empty.
5. Set a stance and state what would change your mind - name the specific
   metric and the level.""",
            ]
        )

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> FundamentalsReport:
        f = bundle.fundamentals
        quality = bundle.quality(EXPECTED_SOURCES)

        if not f:
            return FundamentalsReport(
                role=self.role,
                symbol=bundle.ref.yf_symbol,
                stance=Stance.NEUTRAL,
                confidence=0.1,
                summary=(
                    f"No fundamental data could be retrieved for {bundle.ref.base_symbol}. "
                    "No fundamental view is possible; treat this leg as unknown."
                ),
                risks=["Fundamental data entirely unavailable"],
                data_quality=quality,
            )

        quality_score, quality_notes = _earnings_quality(f)
        governance = _governance_flags(f, bundle)
        valuation_verdict, valuation_score = _valuation(f, bundle)
        growth_score = _growth_score(f)

        composite = 0.4 * valuation_score + 0.35 * quality_score + 0.25 * growth_score
        stance = _stance_from(composite)

        return FundamentalsReport(
            role=self.role,
            symbol=bundle.ref.yf_symbol,
            stance=stance,
            confidence=confidence_from_quality(
                0.4 + 0.4 * abs(composite), quality.completeness, penalties=len(governance)
            ),
            summary=_summary(bundle, f, valuation_verdict, stance),
            reasoning=(
                f"Valuation score {valuation_score:+.2f}, earnings quality {quality_score:+.2f}, "
                f"growth {growth_score:+.2f} -> composite {composite:+.2f}."
            ),
            market_cap_inr=f.get("market_cap_inr"),
            pe=f.get("pe"),
            forward_pe=f.get("forward_pe"),
            pb=f.get("pb"),
            ev_to_ebitda=f.get("ev_to_ebitda"),
            dividend_yield_pct=f.get("dividend_yield_pct"),
            roe_pct=f.get("roe_pct"),
            roce_pct=f.get("roce_pct"),
            debt_to_equity=f.get("debt_to_equity"),
            interest_coverage=f.get("interest_coverage"),
            current_ratio=f.get("current_ratio"),
            operating_cash_flow_inr=f.get("operating_cash_flow_inr"),
            free_cash_flow_inr=f.get("free_cash_flow_inr"),
            ocf_to_pat=f.get("ocf_to_pat"),
            sales_growth_3y_cagr_pct=f.get("sales_growth_3y_cagr_pct"),
            profit_growth_3y_cagr_pct=f.get("profit_growth_3y_cagr_pct"),
            shareholding=build_shareholding(f),
            recent_quarters=build_quarters(f),
            peers=bundle.peers,
            valuation_verdict=valuation_verdict,
            earnings_quality_notes=quality_notes,
            governance_flags=governance,
            key_points=_key_points(f, bundle),
            risks=_risks(f, bundle),
            what_would_change_my_mind=_change_my_mind(f),
            evidence=_evidence(f, bundle),
            data_quality=quality,
        )


# ------------------------------------------------------------------ scoring
def _valuation(f: dict[str, Any], bundle: MarketBundle) -> tuple[str, float]:
    """Verdict text plus a score in [-1, 1] where positive means attractive."""
    pe = f.get("pe")
    if pe is None or pe <= 0:
        return (
            "P/E unavailable or meaningless (loss-making or missing data); valuation "
            "cannot be assessed on earnings multiples.",
            0.0,
        )

    peer_pes = sorted(p.pe for p in bundle.peers if p.pe and p.pe > 0)
    if not peer_pes:
        # No peer set: fall back to broad absolute bands for Indian large caps.
        if pe < 15:
            return f"Trading at {pe:.1f}x trailing earnings, absolutely cheap for an Indian large cap.", 0.5
        if pe > 60:
            return f"Trading at {pe:.1f}x trailing earnings - a demanding multiple with no peer set to anchor it.", -0.5
        return f"Trading at {pe:.1f}x trailing earnings; no peer set retrieved, so this is unanchored.", 0.0

    median = peer_pes[len(peer_pes) // 2]
    delta_pct = (pe - median) / median * 100
    if delta_pct <= -25:
        score, label = 0.7, "a material discount"
    elif delta_pct <= -10:
        score, label = 0.35, "a modest discount"
    elif delta_pct < 10:
        score, label = 0.0, "broadly in line"
    elif delta_pct < 30:
        score, label = -0.35, "a modest premium"
    else:
        score, label = -0.7, "a steep premium"

    # A premium is earned when returns on capital justify it.
    roce = f.get("roce_pct")
    if roce and roce > 20 and score < 0:
        score += 0.25
        label += f", though ROCE of {roce:.1f}% partly justifies it"

    return (
        f"At {pe:.1f}x trailing earnings versus a peer median of {median:.1f}x, "
        f"{bundle.ref.base_symbol} trades at {label} ({delta_pct:+.1f}%).",
        max(-1.0, min(1.0, score)),
    )


def _earnings_quality(f: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []

    ocf_pat = f.get("ocf_to_pat")
    if ocf_pat is not None:
        if ocf_pat >= 1.0:
            score += 0.4
            notes.append(f"Cash conversion healthy: OCF/PAT at {ocf_pat:.2f}x")
        elif ocf_pat >= OCF_PAT_WARNING:
            score += 0.1
            notes.append(f"Cash conversion adequate at {ocf_pat:.2f}x")
        else:
            score -= 0.5
            notes.append(
                f"Weak cash conversion: OCF/PAT at {ocf_pat:.2f}x, below the 0.7 threshold. "
                "Check receivables and inventory build before trusting reported profit."
            )
    else:
        notes.append("Cash conversion could not be computed - earnings quality unverified")

    de = f.get("debt_to_equity")
    if de is not None:
        if de <= 0.3:
            score += 0.3
            notes.append(f"Balance sheet is effectively net-cash or lightly geared (D/E {de:.2f}x)")
        elif de >= DE_SEVERE:
            score -= 0.5
            notes.append(f"Heavily leveraged at D/E {de:.2f}x - refinancing risk is real")
        elif de >= DE_WARNING:
            score -= 0.25
            notes.append(f"Elevated leverage at D/E {de:.2f}x")

    cover = f.get("interest_coverage")
    if cover is not None and cover < INTEREST_COVER_WARNING:
        score -= 0.35
        notes.append(
            f"Interest coverage of only {cover:.1f}x - earnings barely service the debt"
        )

    roce = f.get("roce_pct")
    if roce is not None:
        if roce >= 20:
            score += 0.35
            notes.append(f"High return on capital employed at {roce:.1f}%")
        elif roce < 10:
            score -= 0.25
            notes.append(f"Sub-par ROCE at {roce:.1f}% - capital is not earning its cost")

    fcf = f.get("free_cash_flow_inr")
    if fcf is not None and fcf < 0:
        score -= 0.2
        notes.append("Negative free cash flow - the business is consuming capital")

    return max(-1.0, min(1.0, score)), notes[:6]


def _growth_score(f: dict[str, Any]) -> float:
    score = 0.0
    sales = f.get("sales_growth_3y_cagr_pct")
    profit = f.get("profit_growth_3y_cagr_pct")
    if sales is not None:
        score += 0.4 if sales > 15 else (0.15 if sales > 8 else (-0.3 if sales < 0 else 0.0))
    if profit is not None:
        score += 0.5 if profit > 20 else (0.2 if profit > 10 else (-0.4 if profit < 0 else 0.0))
    latest_growth = f.get("revenue_growth_yoy_pct")
    if latest_growth is not None:
        score += 0.2 if latest_growth > 12 else (-0.25 if latest_growth < 0 else 0.0)
    return max(-1.0, min(1.0, score))


def _governance_flags(f: dict[str, Any], bundle: MarketBundle) -> list[str]:
    flags: list[str] = []
    promoter = f.get("held_by_insiders_pct")
    if promoter is None:
        flags.append(
            "Promoter holding and pledge data not available from the free feed - the "
            "governance picture is unverified, not clean"
        )
    elif promoter < 25:
        flags.append(
            f"Low promoter/insider holding at {promoter:.1f}% - weaker alignment and a "
            "wider takeover/dilution surface"
        )

    if f.get("promoter_pledge_pct"):
        flags.append(f"Promoter pledge at {f['promoter_pledge_pct']:.1f}% of holding")

    from tradingagents_in.dataflows.corporate_actions import recent_price_adjusting

    recent = recent_price_adjusting(bundle.corporate_actions, within_days=180)
    if recent:
        flags.append(
            "Recent price-adjusting corporate action(s): "
            + "; ".join(str(a) for a in recent[:2])
        )

    # Headline-level governance scan.
    watch = (
        "sebi", "auditor", "resign", "pledge", "raid", "probe", "fraud",
        "nclt", "insolvency", "qip", "penalty",
    )
    for item in bundle.company_news[:15]:
        lowered = item.title.lower()
        if any(term in lowered for term in watch):
            flags.append(f"Headline flag: {item.title}")
    return flags[:6]


def _stance_from(composite: float) -> Stance:
    if composite >= 0.45:
        return Stance.BULLISH
    if composite >= 0.15:
        return Stance.MILDLY_BULLISH
    if composite <= -0.45:
        return Stance.BEARISH
    if composite <= -0.15:
        return Stance.MILDLY_BEARISH
    return Stance.NEUTRAL


def _key_points(f: dict[str, Any], bundle: MarketBundle) -> list[str]:
    points: list[str] = []
    if f.get("pe"):
        pb = f.get("pb")
        points.append(f"P/E {f['pe']:.1f}x" + (f", P/B {pb:.1f}x" if pb else ""))
    if f.get("roce_pct"):
        roe = f.get("roe_pct")
        points.append(f"ROCE {f['roce_pct']:.1f}%" + (f", ROE {roe:.1f}%" if roe else ""))
    if f.get("debt_to_equity") is not None:
        points.append(f"Debt/Equity {f['debt_to_equity']:.2f}x")
    if f.get("ocf_to_pat") is not None:
        points.append(f"Cash conversion OCF/PAT {f['ocf_to_pat']:.2f}x")
    if f.get("sales_growth_3y_cagr_pct") is not None:
        profit_cagr = f.get("profit_growth_3y_cagr_pct")
        points.append(
            f"3-year sales CAGR {f['sales_growth_3y_cagr_pct']:.1f}%"
            + (f", profit CAGR {profit_cagr:.1f}%" if profit_cagr is not None else "")
        )
    quarters = f.get("quarters") or []
    if quarters and quarters[0].get("revenue_yoy_pct") is not None:
        points.append(
            f"Latest quarter {quarters[0]['period']}: revenue "
            f"{quarters[0]['revenue_yoy_pct']:+.1f}% YoY"
        )
    return points[:8]


def _risks(f: dict[str, Any], bundle: MarketBundle) -> list[str]:
    risks: list[str] = []
    de = f.get("debt_to_equity")
    if de and de >= DE_WARNING:
        risks.append(f"Leverage at {de:.2f}x D/E amplifies any earnings miss")
    if (f.get("ocf_to_pat") or 1.0) < OCF_PAT_WARNING:
        risks.append("Reported profit is not converting to cash")
    if f.get("interest_coverage") and f["interest_coverage"] < INTEREST_COVER_WARNING:
        risks.append("Thin interest coverage leaves little room for a rate or margin shock")
    if not bundle.peers:
        risks.append("No peer set retrieved, so the valuation call is weakly anchored")
    if f.get("forward_pe") and f.get("pe") and f["forward_pe"] > f["pe"]:
        risks.append(
            "Forward P/E above trailing implies the street expects earnings to fall"
        )
    risks.append(
        "GST rate changes, sector-specific levies and Union Budget measures can reset "
        "margins in a single announcement"
    )
    return risks[:6]


def _change_my_mind(f: dict[str, Any]) -> list[str]:
    out = ["Two consecutive quarters of OCF/PAT below 0.7 with rising receivables"]
    if f.get("debt_to_equity") is not None:
        out.append(f"Debt/Equity moving above {max(1.0, (f['debt_to_equity'] or 0) + 0.5):.1f}x")
    out.append("Any reduction in promoter holding or a fresh pledge disclosure")
    out.append("A quarter where revenue growth turns negative year-on-year")
    return out[:4]


def _evidence(f: dict[str, Any], bundle: MarketBundle) -> list[Evidence]:
    sources = ", ".join(f.get("sources", [])) or "yfinance"
    evidence: list[Evidence] = []
    for field, label in (
        ("pe", "Trailing P/E"),
        ("roce_pct", "ROCE"),
        ("debt_to_equity", "Debt to Equity"),
        ("ocf_to_pat", "OCF/PAT cash conversion"),
    ):
        if f.get(field) is not None:
            evidence.append(
                Evidence(
                    claim=f"{label} for {bundle.ref.base_symbol}",
                    source=sources,
                    value=f"{f[field]:.2f}",
                    as_of=str(bundle.session.last_trading_day),
                )
            )
    return evidence


def _summary(
    bundle: MarketBundle, f: dict[str, Any], verdict: str, stance: Stance
) -> str:
    name = bundle.ref.name or bundle.ref.base_symbol
    quality_bits: list[str] = []
    if f.get("roce_pct"):
        quality_bits.append(f"ROCE {f['roce_pct']:.1f}%")
    if f.get("debt_to_equity") is not None:
        quality_bits.append(f"D/E {f['debt_to_equity']:.2f}x")
    if f.get("ocf_to_pat") is not None:
        quality_bits.append(f"OCF/PAT {f['ocf_to_pat']:.2f}x")
    quality_text = ", ".join(quality_bits) or "limited ratio coverage"
    return (
        f"{name}: {verdict} Business quality reads as {quality_text}. "
        f"Fundamental stance: {stance.value}."
    )
