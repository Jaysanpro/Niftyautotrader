"""Sentiment Analyst.

Reads the Indian financial press and retail social venues, then asks the only
question that matters: is this crowd early, or is it the exit liquidity?
"""

from __future__ import annotations

from typing import Any

from tradingagents_in.agents.base import Agent, confidence_from_quality
from tradingagents_in.agents.briefings import header, sentiment_brief
from tradingagents_in.dataflows.sentiment import aggregate, buzz_level, contrarian_flag
from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.models.analyst import SentimentReport
from tradingagents_in.models.base import Evidence
from tradingagents_in.models.enums import AgentRole, Stance

EXPECTED_SOURCES = ["news", "sentiment"]


class SentimentAnalyst(Agent[SentimentReport]):
    role = AgentRole.SENTIMENT_ANALYST
    output_model = SentimentReport
    tier_key = "sentiment_analyst"

    def system_prompt(self) -> str:
        return """\
You are the Sentiment Analyst covering Indian markets.

Sources you weigh, in descending order of reliability: Economic Times,
Moneycontrol, Business Standard, LiveMint and BusinessLine reporting; exchange
filings; then retail social venues - X/Twitter cashtags and #Nifty/#Sensex
chatter, r/IndiaInvestments and r/IndianStockMarket, StockTwits India.

What you are actually measuring:
- DIRECTION: is coverage improving or deteriorating versus last week?
- INTENSITY: is the chatter quiet, normal, elevated or frenzied?
- COMPOSITION: is enthusiasm concentrated in retail (tips channels, cashtag
  spam, "multibagger" language) or in institutional commentary (brokerage
  upgrades, block deals, management access notes)? These mean opposite things.

Indian retail sentiment is strongly reflexive and momentum-chasing. Treat
frenzied retail enthusiasm on an already-extended move as a warning, not
confirmation. Equally, uniform gloom after a capitulation move is often near
exhaustion - but sentiment alone never times an entry, price has to confirm.

Watch for coordinated promotion. Sudden coordinated positivity on an illiquid
small or mid cap, especially one under ASM/GSM surveillance, is an operator
signature and should be called out as such.

Be honest about coverage. If only headlines were retrieved and no social data,
say the read is press-only and lower your confidence.
"""

    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        return "\n\n".join(
            [
                header(bundle),
                sentiment_brief(bundle),
                """## Your task

Produce a SentimentReport:
1. Set sentiment_score in [-1, 1] and keep it close to the computed aggregate
   unless the headlines clearly justify otherwise - explain any deviation.
2. Fill source_breakdown from the briefing.
3. Write retail_vs_institutional: where the enthusiasm sits and why it matters.
4. Set contrarian_signal only if crowding genuinely argues for fading; leave it
   null otherwise. Do not manufacture a contrarian take.
5. List notable_headlines verbatim from the briefing.""",
            ]
        )

    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> SentimentReport:
        quality = bundle.quality(EXPECTED_SOURCES)
        sources = bundle.sentiment_sources

        if not sources:
            return SentimentReport(
                role=self.role,
                symbol=bundle.ref.yf_symbol,
                stance=Stance.NEUTRAL,
                confidence=0.12,
                summary=(
                    "No sentiment source returned data for this scrip. Sentiment is unknown, "
                    "not neutral - do not read the absence of chatter as calm."
                ),
                sentiment_score=0.0,
                buzz_level="quiet",
                risks=["No sentiment coverage retrieved"],
                data_quality=quality,
            )

        score = aggregate(sources)
        buzz = buzz_level(sources)
        change_pct = bundle.quote.get("change_pct")
        contrarian = contrarian_flag(score, buzz, _recent_move_pct(bundle))

        stance = _stance(score, contrarian is not None)
        headlines = [f"{i.title} ({i.source})" for i in bundle.company_news[:8]]
        press = next((s for s in sources if s.source == "Indian financial press"), None)
        social = [s for s in sources if s.source != "Indian financial press"]

        return SentimentReport(
            role=self.role,
            symbol=bundle.ref.yf_symbol,
            stance=stance,
            confidence=confidence_from_quality(
                0.35 + 0.35 * abs(score), quality.completeness,
                penalties=0 if social else 1,
            ),
            summary=_summary(bundle, score, buzz, contrarian, len(headlines)),
            reasoning=(
                f"Aggregate {score:+.3f} across {len(sources)} source(s); buzz {buzz}. "
                f"Session move {change_pct if change_pct is not None else 'n/a'}%."
            ),
            sentiment_score=score,
            buzz_level=buzz,
            retail_vs_institutional=_composition(press, social),
            source_breakdown=sources,
            notable_headlines=headlines,
            contrarian_signal=contrarian,
            key_points=_key_points(sources, score, buzz),
            risks=_risks(bundle, social, buzz),
            what_would_change_my_mind=[
                "A brokerage downgrade or upgrade that shifts the institutional tone",
                "Company-specific headlines reversing direction for three consecutive days",
                "A jump in chatter volume without any corresponding news - promotion signature",
            ],
            evidence=[
                Evidence(
                    claim=f"{s.source} sentiment {s.score:+.2f} across {s.volume} items",
                    source=s.source,
                    value=f"{s.score:+.2f}",
                    as_of=str(bundle.session.last_trading_day),
                )
                for s in sources
            ],
            data_quality=quality,
        )


def _recent_move_pct(bundle: MarketBundle) -> float | None:
    """Ten-session move, which is the window retail sentiment actually reacts to."""
    daily = bundle.prices.get("1d")
    if daily is None or len(daily) < 11:
        return bundle.quote.get("change_pct")
    recent = daily["close"]
    return round(float((recent.iloc[-1] / recent.iloc[-11] - 1) * 100), 2)


def _stance(score: float, has_contrarian: bool) -> Stance:
    # A live contrarian flag caps the stance: crowded enthusiasm is not a buy.
    if has_contrarian:
        return Stance.NEUTRAL if abs(score) < 0.7 else (
            Stance.MILDLY_BEARISH if score > 0 else Stance.MILDLY_BULLISH
        )
    if score >= 0.4:
        return Stance.BULLISH
    if score >= 0.15:
        return Stance.MILDLY_BULLISH
    if score <= -0.4:
        return Stance.BEARISH
    if score <= -0.15:
        return Stance.MILDLY_BEARISH
    return Stance.NEUTRAL


def _composition(press: Any, social: list[Any]) -> str:
    if not social:
        return (
            "Press coverage only - no social data was retrieved, so retail positioning "
            "cannot be assessed. Treat the read as institutional/media tone alone."
        )
    social_score = sum(s.score for s in social) / len(social)
    press_score = press.score if press else 0.0
    gap = social_score - press_score
    if gap > 0.3:
        return (
            f"Retail chatter ({social_score:+.2f}) is materially more bullish than press "
            f"coverage ({press_score:+.2f}). Retail-led enthusiasm ahead of the reporting "
            "is a crowding warning rather than confirmation."
        )
    if gap < -0.3:
        return (
            f"Retail ({social_score:+.2f}) is more negative than the press "
            f"({press_score:+.2f}) - retail capitulation while coverage stays constructive."
        )
    return (
        f"Retail ({social_score:+.2f}) and press ({press_score:+.2f}) are broadly aligned; "
        "no meaningful divergence between the two."
    )


def _key_points(sources: list[Any], score: float, buzz: str) -> list[str]:
    points = [f"Aggregate sentiment {score:+.2f} on a [-1, 1] scale, buzz level {buzz}"]
    points.extend(
        f"{s.source}: {s.score:+.2f} across {s.volume} items" for s in sources
    )
    return points[:8]


def _risks(bundle: MarketBundle, social: list[Any], buzz: str) -> list[str]:
    risks: list[str] = []
    if not social:
        risks.append("No social sentiment data - the retail side of positioning is unobserved")
    if buzz == "frenzied":
        risks.append("Frenzied chatter: crowded positioning raises reversal risk")
    if bundle.micro.surveillance.value.startswith(("ASM", "GSM")):
        risks.append(
            f"Scrip is under {bundle.micro.surveillance.value} surveillance - promotional "
            "sentiment on a surveilled scrip is a serious warning"
        )
    risks.append(
        "Sentiment is a coincident indicator: it confirms moves, it does not lead them"
    )
    return risks[:5]


def _summary(
    bundle: MarketBundle, score: float, buzz: str, contrarian: str | None, headline_count: int
) -> str:
    tone = (
        "clearly positive" if score > 0.35 else
        "mildly positive" if score > 0.12 else
        "clearly negative" if score < -0.35 else
        "mildly negative" if score < -0.12 else
        "broadly neutral"
    )
    base = (
        f"Sentiment on {bundle.ref.base_symbol} is {tone} ({score:+.2f}) with {buzz} "
        f"chatter across {headline_count} company headlines."
    )
    return f"{base} {contrarian}" if contrarian else base
