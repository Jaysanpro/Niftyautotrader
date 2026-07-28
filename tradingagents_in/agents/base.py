"""Agent base class and the shared India-market system prompt."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from tradingagents_in.dataflows.toolkit import MarketBundle
from tradingagents_in.llm.structured import call_structured
from tradingagents_in.models.enums import AgentRole
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# The house rules. Every agent inherits these; role prompts append to them.
HOUSE_RULES = """\
You are part of IndianTradingAgents, a research desk covering NSE and BSE listed
equities and Indian index derivatives. You are not a hype machine and you are not
a compliance bot: you are a working analyst on an Indian desk.

Non-negotiable rules:

1. NUMBERS COME FROM THE BRIEFING. Never invent a price, ratio, holding
   percentage, flow figure or date. If the briefing does not contain it, say
   "not available" and lower your confidence. A confident wrong number is the
   single worst thing you can produce.
2. ALL VALUES IN INR. Prices in rupees, market caps and flows in crore/lakh
   where natural. All times are Asia/Kolkata (IST). The session is 09:15-15:30.
3. INDIA MECHANICS ARE PART OF THE ANALYSIS, not decoration. Circuit filters,
   ASM/GSM surveillance, F&O ban periods, weekly/monthly expiry, delivery vs
   intraday, T+1 settlement, SEBI position limits, STT and stamp duty, promoter
   pledging, and FII/DII flow regimes all change whether an idea is tradable.
4. BE FALSIFIABLE. Every claim needs a number, a source or an observable event
   behind it. "Strong momentum" is not analysis; "RSI 68 with price 4% above a
   rising EMA20 on 1.8x average volume" is.
5. STATE WHAT WOULD CHANGE YOUR MIND. If you cannot name the observation that
   would flip your view, you do not have a view, you have a preference.
6. RESPECT UNCERTAINTY. Confidence is a probability, not enthusiasm. Data gaps,
   a stale calendar, missing promoter data or a failed source must pull it down.

You must respond with a single valid object matching the requested schema.
"""


class Agent(ABC, Generic[T]):
    """One node in the graph: builds a deterministic draft, then refines it."""

    role: AgentRole
    output_model: type[T]
    tier_key: str  # key into config["agent_model_tier"]

    def __init__(self, config: dict[str, Any]):
        self.config = config

    # ------------------------------------------------------------ contract
    @abstractmethod
    def system_prompt(self) -> str:
        """Role-specific instructions, appended to HOUSE_RULES."""

    @abstractmethod
    def user_prompt(self, bundle: MarketBundle, state: dict[str, Any]) -> str:
        """The briefing and the task for this specific call."""

    @abstractmethod
    def deterministic_draft(self, bundle: MarketBundle, state: dict[str, Any]) -> T:
        """A fully-populated output computed from data alone, with no LLM.

        This is the offline mode's answer and the safety net when the model
        fails. It must be genuinely usable, not a placeholder.
        """

    # ------------------------------------------------------------- running
    @property
    def tier(self) -> str:
        return str(self.config.get("agent_model_tier", {}).get(self.tier_key, "quick"))

    def run(self, bundle: MarketBundle, state: dict[str, Any] | None = None) -> T:
        state = state or {}
        started = time.time()

        draft = self.deterministic_draft(bundle, state)
        result, meta = call_structured(
            config=self.config,
            schema=self.output_model,
            system_prompt=f"{HOUSE_RULES}\n\n{self.system_prompt()}",
            user_prompt=self.user_prompt(bundle, state),
            tier=self.tier,
            fallback=draft,
            max_attempts=int(self.config.get("llm_max_retries", 3)),
        )

        # Stamp provenance so a decision log can be audited later.
        if hasattr(result, "model_used"):
            result.model_used = meta.model if meta.used_llm else "deterministic"
        if hasattr(result, "latency_ms"):
            result.latency_ms = int((time.time() - started) * 1000)
        if hasattr(result, "role"):
            result.role = self.role
        if meta.fell_back:
            log.info("%s fell back to the deterministic draft", self.role.value)
        return result

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} role={self.role.value} tier={self.tier}>"


def confidence_from_quality(base: float, completeness: float, penalties: int = 0) -> float:
    """Blend a base conviction with data completeness and known problems."""
    value = base * (0.55 + 0.45 * completeness) - 0.05 * penalties
    return round(max(0.05, min(0.95, value)), 3)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
