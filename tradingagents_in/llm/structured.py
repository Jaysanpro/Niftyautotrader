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
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tradingagents_in.llm.provider import (
    LLMUnavailable,
    build_chat_model,
    provider_is_offline,
    resolve_model,
)
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

# Which "no usable model" reasons have already been reported.
_REPORTED_UNAVAILABLE: set[str] = set()
_REPORT_LOCK = threading.Lock()


# Set once a provider rejects our credentials. An invalid key is not a
# transient fault, so every later call in this process would fail the same
# way; the latch turns thousands of doomed requests into one.
_CREDENTIALS_REJECTED: dict[str, str] = {}


def rejected_credentials(provider: str) -> str | None:
    """Why this provider's key was rejected in this process, if it was.

    A key that is present but wrong is invisible to a configuration check -
    the variable is set, so everything reports healthy while every call
    comes back 401. This is the only place that knows otherwise, so
    anything describing what the deployment is doing has to ask it.
    """
    return _CREDENTIALS_REJECTED.get(str(provider).lower())


def _status_of(exc: Exception) -> int | None:
    """HTTP status behind an SDK exception, wherever it hung it."""
    for owner in (exc, getattr(exc, "response", None), getattr(exc, "body", None)):
        if owner is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(owner, attr, None)
            if isinstance(value, int):
                return value
    return None


def is_auth_failure(exc: Exception) -> bool:
    """Did the provider reject our credentials rather than our request?

    Worth its own test because the answer decides whether retrying is
    sensible or just expensive: a 401 will be a 401 on every attempt.
    """
    if _status_of(exc) in (401, 403):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "authentication_error",
            "invalid x-api-key",
            "invalid_api_key",
            "incorrect api key",
            "permission_error",
            "unauthorized",
        )
    )


def is_retryable(exc: Exception) -> bool:
    """Could this same call plausibly succeed if tried again?

    Rate limits, timeouts, overloads and 5xx are worth another attempt.
    Bad credentials and a malformed request are not: retrying them burns
    time and quota to arrive at the same error. With a bad key, a NIFTY50
    screen was making 1,920 doomed calls and sleeping 64 minutes in backoff
    to reach the deterministic answer it would have reached at once.
    """
    if is_auth_failure(exc):
        return False
    status = _status_of(exc)
    if status in (400, 404, 422):
        return False
    if status == 429 or (status is not None and status >= 500):
        return True
    text = str(exc).lower()
    if "invalid_request_error" in text or "model_not_found" in text:
        return False
    return True


def _warn_unavailable_once(reason: str, key: str | None = None) -> None:
    """Report an unusable provider once per process, then stay quiet.

    A missing or rejected key is a fact about the configuration, not an
    event: it is true identically for every agent on every run. Warning per
    call meant ten lines a run, and about 640 in one NIFTY50 screen - and
    because a screen forwards only warnings to the browser, those lines
    buried the per-name progress they were streamed alongside.

    ``key`` is what makes two reports the same report, and it has to be
    given separately because the message is not stable: a provider's
    rejection carries a fresh request_id every time, so keying on the text
    deduplicates nothing. Four still got through even so, because LangGraph
    runs the four analysts on parallel threads and they all failed before
    any of them had recorded it.

    A *different* failure still gets its own warning.
    """
    with _REPORT_LOCK:
        marker = key or reason
        first = marker not in _REPORTED_UNAVAILABLE
        _REPORTED_UNAVAILABLE.add(marker)
    if first:
        log.warning(
            "LLM unavailable (%s); using deterministic output. Every agent "
            "will fall back for the rest of this process, and this is logged "
            "once rather than per call.",
            reason,
        )
    else:
        log.debug("LLM unavailable (%s); using deterministic output", reason)


def _get_model(config: dict[str, Any], tier: str) -> Any:
    # The resolved id, not the configured one: an unset tier now means "the
    # provider's default", and two providers' defaults must not share a key.
    try:
        model = resolve_model(config, tier)
    except LLMUnavailable:
        model = str(config.get("deep_think_model" if tier == "deep" else "quick_think_model"))
    key = (str(config.get("llm_provider")), tier, model)
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

    provider = str(config.get("llm_provider", "")).lower()
    rejected = _CREDENTIALS_REJECTED.get(provider)
    if rejected is not None:
        # The key was already rejected once in this process. Every agent on
        # every name would get the same 401, so there is nothing to learn
        # from asking again - just latency and quota.
        log.debug("%s credentials already rejected (%s); using the draft", provider, rejected)
        meta.error, meta.fell_back = rejected, True
        if fallback is None:
            raise LLMUnavailable(rejected)
        return fallback, meta

    started = time.time()
    try:
        model = _get_model(config, tier)
    except LLMUnavailable as exc:
        _warn_unavailable_once(str(exc))
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
            if is_auth_failure(exc):
                # Latch the whole provider, not just this call: the key is
                # wrong for every agent and every name that follows.
                reason = (
                    f"{provider} rejected the API key "
                    f"({str(exc).strip()[:160]})"
                )
                _CREDENTIALS_REJECTED[provider] = reason
                _warn_unavailable_once(reason, key=f"{provider}:credentials-rejected")
                break
            if not is_retryable(exc):
                log.warning("LLM call failed, not retryable: %s", exc)
                break
            log.warning("LLM call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 8))

    meta.latency_ms = int((time.time() - started) * 1000)
    meta.error = str(last_error) if last_error else "unknown failure"
    meta.fell_back = True
    if fallback is None:
        raise LLMUnavailable(f"structured call failed after {max_attempts} attempts: {last_error}")
    log.info("falling back to deterministic output for %s", schema.__name__)
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
    provider = str(config.get("llm_provider", "")).lower()
    if provider in _CREDENTIALS_REJECTED:
        return fallback
    try:
        model = _get_model(config, tier)
        response = model.invoke([("system", system_prompt), ("human", user_prompt)])
        return getattr(response, "content", str(response)) or fallback
    except Exception as exc:
        # Same latch as the structured path: a rejected key is rejected for
        # every later call too, and narration is not worth a retry storm.
        if is_auth_failure(exc):
            reason = f"{provider} rejected the API key ({str(exc).strip()[:160]})"
            _CREDENTIALS_REJECTED[provider] = reason
            _warn_unavailable_once(reason, key=f"{provider}:credentials-rejected")
        else:
            log.warning("text call failed: %s", exc)
        return fallback
