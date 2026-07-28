"""The analyst team."""

from tradingagents_in.agents.analysts.fundamentals import FundamentalsAnalyst
from tradingagents_in.agents.analysts.news_macro import NewsMacroAnalyst
from tradingagents_in.agents.analysts.sentiment import SentimentAnalyst
from tradingagents_in.agents.analysts.technical import TechnicalAnalyst

ANALYSTS = {
    "fundamentals": FundamentalsAnalyst,
    "sentiment": SentimentAnalyst,
    "news_macro": NewsMacroAnalyst,
    "technical": TechnicalAnalyst,
}

__all__ = [
    "ANALYSTS",
    "FundamentalsAnalyst",
    "NewsMacroAnalyst",
    "SentimentAnalyst",
    "TechnicalAnalyst",
]
