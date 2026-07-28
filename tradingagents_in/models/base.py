"""Shared model primitives."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tradingagents_in.models.enums import AgentRole, Exchange
from tradingagents_in.utils.timeutils import now_ist


class ITAModel(BaseModel):
    """Base for every contract in the framework.

    ``extra="ignore"`` matters: LLMs occasionally emit an extra key and we would
    rather keep the decision than fail the run on a stray field.
    """

    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=False,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    def to_markdown(self) -> str:
        """Human/LLM-readable rendering. Subclasses override for nicer output."""
        return _dict_to_markdown(self.model_dump(mode="json"))


class Evidence(ITAModel):
    """One citable fact behind a claim. Agents must attach these, not vibes."""

    claim: str = Field(..., description="The specific fact being asserted")
    source: str = Field(..., description="Where it came from, e.g. 'NSE', 'Screener.in', 'Mint'")
    value: str | None = Field(None, description="Numeric or textual value, if any")
    url: str | None = None
    as_of: str | None = Field(None, description="Date/period the fact refers to")


class DataQuality(ITAModel):
    """What the agent actually managed to fetch. Drives confidence haircuts."""

    sources_used: list[str] = Field(default_factory=list)
    sources_failed: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    staleness_days: float | None = None
    completeness: float = Field(
        1.0, ge=0.0, le=1.0, description="0-1 fraction of the expected inputs that were available"
    )

    @property
    def is_degraded(self) -> bool:
        return self.completeness < 0.6 or bool(self.sources_failed)


class SymbolRef(ITAModel):
    """A resolved Indian instrument reference."""

    base_symbol: str = Field(..., description="Exchange symbol without suffix, e.g. RELIANCE")
    yf_symbol: str = Field(..., description="yfinance symbol, e.g. RELIANCE.NS")
    exchange: Exchange = Exchange.NSE
    name: str | None = None
    sector: str | None = None
    isin: str | None = None
    is_index: bool = False
    fno_eligible: bool = False
    lot_size: int | None = None

    @field_validator("base_symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.yf_symbol


class AgentOutput(ITAModel):
    """Common envelope every agent decision carries."""

    role: AgentRole
    symbol: str
    generated_at: datetime = Field(default_factory=now_ist)
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Self-assessed confidence in this output"
    )
    summary: str = Field(..., description="2-4 sentence verdict a PM could read on its own")
    reasoning: str = Field("", description="Full chain of argument")
    evidence: list[Evidence] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    model_used: str | None = None
    latency_ms: int | None = None

    def effective_confidence(self) -> float:
        """Confidence after a haircut for missing data.

        An agent that only saw half its inputs should not speak with full
        conviction, whatever it claims about itself.
        """
        return round(self.confidence * (0.5 + 0.5 * self.data_quality.completeness), 4)


def _dict_to_markdown(payload: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value in (None, [], {}, ""):
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}- **{label}**:")
                lines.append(_dict_to_markdown(value, indent + 1))
            else:
                lines.append(f"{pad}- **{label}**: {value}")
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                lines.append(_dict_to_markdown(item, indent))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{payload}")
    return "\n".join(x for x in lines if x.strip())
