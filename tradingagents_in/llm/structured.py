"""Structured-output calling with retries and a deterministic fallback.

Every agent asks for a Pydantic object, never free text. Two failure modes are
handled explicitly:

  * the model returns unparseable or schema-violating JSON -> retry with the
    validation error appended, then fall back to the deterministic draft;
  * the provider is down, rate-limited or not configured -> use the
    deterministic draft directly.

The deterministic draft is not a stub. It is a fully-populated object computed
from the market data, so a run always produces a defensible answer and the
LLM's job is to *improve* on a baseline rather than invent from nothing.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tradingagents_in.llm.provider import LLMUnavailable, build_chat_model, provider_is_offline
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _get_model(config: dict[str, Any], tier: str) -> Any:
    key = (
        str(config.get("llm_provider")),
        tier,
        str(config.get("deep_think_model" if tier == "deep" else "quick_think_model")),
    )
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_chat_model(config, tier)
    return _MODEL_CACHE[key]


def reset_model_cache() -> None:
    _MODEL_CACHE.clear()


class StructuredCallResult(BaseModel):
    """Bookkeeping about how an agent's output was actually produced."""

    used_llm: bool = False
    model: str | None = None
    attempts: int = 0
    latency_ms: int = 0
    fell_back: bool = False
    error: str | None = None


def call_structured(
    config: dict[str, Any],
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    tier: str = "quick",
    fallback: T | None = None,
    max_attempts: int = 3,
) -> tuple[T, StructuredCallResult]:
    """Ask the model for one `schema` instance.

    `fallback` is the deterministic draft. It is returned unchanged when no LLM
    is configured, and used as the safety net when every attempt fails.
    """
    meta = StructuredCallResult()

    if provider_is_offline(config):
        if fallback is None:
            raise LLMUnavailable(
                "offline provider selected but this agent has no deterministic fallback"
            )
        return fallback, meta

    started = time.time()
    try:
        model = _get_model(config, tier)
    except LLMUnavailable as exc:
        log.warning("LLM unavailable (%s); using deterministic output", exc)
        meta.error, meta.fell_back = str(exc), True
        if fallback is None:
            raise
        return fallback, meta

    from tradingagents_in.llm.provider import resolve_model  # noqa: PLC0415

    meta.model = resolve_model(config, tier)
    structured = model.with_structured_output(schema)

    messages: list[tuple[str, str]] = [("system", system_prompt), ("human", user_prompt)]
    if fallback is not None:
        # Anchoring the model on the computed baseline is what keeps numbers
        # honest: it edits real figures instead of hallucinating fresh ones.
        messages.insert(
            1,
            (
                "human",
                "A deterministic baseline has already been computed from the raw data. "
                "Use it as your factual anchor: keep any number you cannot independently "
                "justify, and change a field only when the evidence in the briefing "
                "supports a different value.\n\n```json\n"
                + json.dumps(fallback.model_dump(mode="json"), indent=2, default=str)[:6000]
                + "\n```",
            ),
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        meta.attempts = attempt
        try:
            result = structured.invoke(messages)
            if isinstance(result, dict):
                result = schema.model_validate(result)
            meta.used_llm = True
            meta.latency_ms = int((time.time() - started) * 1000)
            return result, meta  # type: ignore[return-value]
        except ValidationError as exc:
            last_error = exc
            log.debug("schema validation failed (attempt %d): %s", attempt, exc)
            messages.append(
                (
                    "human",
                    "Your previous response did not satisfy the schema. Fix exactly these "
                    f"problems and return the full object again:\n{exc}",
                )
            )
        except Exception as exc:
            last_error = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 8))

    meta.latency_ms = int((time.time() - started) * 1000)
    meta.error = str(last_error) if last_error else "unknown failure"
    meta.fell_back = True
    if fallback is None:
        raise LLMUnavailable(f"structured call failed after {max_attempts} attempts: {last_error}")
    log.warning("falling back to deterministic output for %s", schema.__name__)
    return fallback, meta


def call_text(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    tier: str = "quick",
    fallback: str = "",
) -> str:
    """Plain-text completion. Used only for narrative summaries, never decisions."""
    if provider_is_offline(config):
        return fallback
    try:
        model = _get_model(config, tier)
        response = model.invoke([("system", system_prompt), ("human", user_prompt)])
        return getattr(response, "content", str(response)) or fallback
    except Exception as exc:
        log.warning("text call failed: %s", exc)
        return fallback
