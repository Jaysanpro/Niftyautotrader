"""Provider construction, with the sampling-temperature rule pinned.

Current Anthropic models reject `temperature` with a 400 rather than ignoring
it, so a configured temperature turns a model upgrade into an outage. These
tests build the chat model with the SDK stubbed out and assert on the kwargs
that would have reached it - the failure is in what gets *sent*, so that is
what has to be checked.
"""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

from tradingagents_in.config.default_config import build_config
from tradingagents_in.llm.provider import (
    PROVIDERS,
    LLMUnavailable,
    anthropic_accepts_temperature,
    build_chat_model,
)

# ------------------------------------------------------- the rule itself


@pytest.mark.parametrize("model", [
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
])
def test_legacy_models_still_take_temperature(model):
    assert anthropic_accepts_temperature(model)


@pytest.mark.parametrize("model", [
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5-2",
])
def test_current_models_do_not_take_temperature(model):
    assert not anthropic_accepts_temperature(model)


def test_an_unknown_model_omits_temperature():
    """The safe direction: a model released after this list was written.

    Omitting costs a less random answer. Sending costs every request.
    """
    assert not anthropic_accepts_temperature("claude-something-9")
    assert not anthropic_accepts_temperature("")


def test_the_check_is_case_and_whitespace_insensitive():
    assert anthropic_accepts_temperature("  Claude-Sonnet-4-5  ")
    assert not anthropic_accepts_temperature("  Claude-Sonnet-5-5  ")


# --------------------------------------------------- what reaches the SDK


@pytest.fixture
def captured_anthropic(monkeypatch):
    """Stub langchain_anthropic and capture the kwargs ChatAnthropic got."""
    seen: dict = {}

    class _ChatAnthropic:
        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    module = types.ModuleType("langchain_anthropic")
    module.ChatAnthropic = _ChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return seen


def _cfg(**overrides):
    base = {"llm_provider": "anthropic"}
    base.update(overrides)
    return build_config(base)


def test_a_current_model_is_built_without_temperature(captured_anthropic):
    build_chat_model(_cfg(deep_think_model="claude-sonnet-5-5", llm_temperature=0.2), "deep")
    assert "temperature" not in captured_anthropic, (
        "temperature reached a model that rejects it with a 400"
    )
    assert captured_anthropic["model"] == "claude-sonnet-5-5"


def test_a_legacy_model_still_gets_its_temperature(captured_anthropic):
    build_chat_model(_cfg(deep_think_model="claude-sonnet-4-5", llm_temperature=0.4), "deep")
    assert captured_anthropic["temperature"] == pytest.approx(0.4)


def test_the_shipped_defaults_are_built_without_temperature(captured_anthropic):
    """The regression that matters: defaults plus a stock config must work.

    ITA_LLM_TEMPERATURE has a default, so this is the path a user who sets
    nothing at all takes.
    """
    for tier in ("deep", "quick"):
        build_chat_model(_cfg(), tier)
        assert "temperature" not in captured_anthropic, (
            f"the default {tier} model would 400 on temperature"
        )


def test_max_tokens_and_retries_still_reach_the_model(captured_anthropic):
    build_chat_model(
        _cfg(deep_think_model="claude-sonnet-5-5", llm_max_tokens=8192, llm_max_retries=5),
        "deep",
    )
    assert captured_anthropic["max_tokens"] == 8192
    assert captured_anthropic["max_retries"] == 5


def test_max_tokens_falls_back_to_a_usable_default(captured_anthropic):
    build_chat_model(_cfg(deep_think_model="claude-sonnet-5-5"), "deep")
    assert captured_anthropic["max_tokens"] > 0


# ------------------------------------------------------------- defaults


def test_the_anthropic_defaults_are_current_generation():
    spec = PROVIDERS["anthropic"]
    for model in (spec.default_deep, spec.default_quick):
        assert not anthropic_accepts_temperature(model), (
            f"shipped default {model} is a generation that predates the "
            "temperature removal; it should be a current model"
        )


# --------------------------------------------------------- missing key


def test_a_missing_key_names_the_variable_and_the_way_out(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable) as excinfo:
        build_chat_model(_cfg(), "deep")
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "offline" in message


def test_offline_has_no_chat_model():
    with pytest.raises(LLMUnavailable):
        build_chat_model(build_config({"llm_provider": "offline"}), "deep")


def test_an_unknown_provider_is_refused():
    with pytest.raises(LLMUnavailable):
        build_chat_model({"llm_provider": "definitely-not-a-provider"}, "deep")


# ------------------------------------------------------- a provider with no key


def test_missing_api_key_names_the_variable(monkeypatch):
    """A provider named without its key is the dangerous configuration.

    The service answers every request - each agent falls back to its
    deterministic draft, which is the design - so nothing looks broken while
    no model is ever asked anything. Anything reporting what a deployment is
    doing has to detect this rather than trust ``llm_provider``.
    """
    from tradingagents_in.llm.provider import missing_api_key

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert missing_api_key({"llm_provider": "anthropic"}) == "ANTHROPIC_API_KEY"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-real")
    assert missing_api_key({"llm_provider": "anthropic"}) is None


def test_the_offline_engine_needs_no_key(monkeypatch):
    from tradingagents_in.llm.provider import missing_api_key

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for provider in ("offline", "none", ""):
        assert missing_api_key({"llm_provider": provider}) is None


def test_an_unknown_provider_reports_no_missing_key():
    """Unknown providers fail elsewhere, with their own message."""
    from tradingagents_in.llm.provider import missing_api_key

    assert missing_api_key({"llm_provider": "not-a-provider"}) is None


def test_an_unusable_provider_is_reported_once_not_per_call(monkeypatch, caplog):
    """A missing key is a fact about the configuration, not an event.

    It is true identically for every agent on every run. Warning per call
    meant ten lines a run and ~640 in one NIFTY50 screen - and since a screen
    forwards only warnings to the browser, they buried the per-name progress
    they were streamed alongside.
    """
    import logging

    from tradingagents_in.llm import structured

    monkeypatch.setattr(structured, "_REPORTED_UNAVAILABLE", set())
    with caplog.at_level(logging.WARNING, logger="tradingagents_in.llm.structured"):
        for _ in range(12):
            structured._warn_unavailable_once("ANTHROPIC_API_KEY is not set")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


def test_a_different_failure_still_gets_its_own_warning(monkeypatch, caplog):
    """Silencing repeats must not silence a new problem."""
    import logging

    from tradingagents_in.llm import structured

    monkeypatch.setattr(structured, "_REPORTED_UNAVAILABLE", set())
    with caplog.at_level(logging.WARNING, logger="tradingagents_in.llm.structured"):
        structured._warn_unavailable_once("ANTHROPIC_API_KEY is not set")
        structured._warn_unavailable_once("ANTHROPIC_API_KEY is not set")
        structured._warn_unavailable_once("langchain_anthropic is not installed")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


# ------------------------------------------------ keys that arrive mangled


def test_a_quoted_key_is_unwrapped(monkeypatch):
    """Dashboards store a value as typed, quotes included.

    A field filled in as "sk-ant-..." arrives with the quotes as part of the
    secret, reaches the API as an invalid key, and comes back as a 401 that
    looks nothing like a quoting mistake.
    """
    from tradingagents_in.llm.provider import read_api_key

    for wrapped in ('"sk-ant-abc"', "'sk-ant-abc'", "  sk-ant-abc  ", '  "sk-ant-abc" '):
        monkeypatch.setenv("ITA_TEST_KEY", wrapped)
        assert read_api_key("ITA_TEST_KEY") == "sk-ant-abc", wrapped


def test_a_single_stray_quote_is_left_alone(monkeypatch):
    """Only a matching pair is stripped.

    A quote on one side is a key someone typed oddly, not a wrapped one, and
    silently altering it would hide the real fault.
    """
    from tradingagents_in.llm.provider import read_api_key

    monkeypatch.setenv("ITA_TEST_KEY", '"sk-ant-abc')
    assert read_api_key("ITA_TEST_KEY") == '"sk-ant-abc'


def test_an_empty_key_reads_as_absent(monkeypatch):
    from tradingagents_in.llm.provider import read_api_key

    monkeypatch.setenv("ITA_TEST_KEY", '""')
    assert read_api_key("ITA_TEST_KEY") is None
    monkeypatch.setenv("ITA_TEST_KEY", "   ")
    assert read_api_key("ITA_TEST_KEY") is None


# -------------------------------------------------------- what to retry


def _err(message: str, status: int | None = None) -> Exception:
    exc = Exception(message)
    if status is not None:
        exc.status_code = status
    return exc


def test_credentials_errors_are_not_retried():
    """A 401 will be a 401 on every attempt.

    Retrying it burns time and quota to reach the same answer. With a bad
    key a NIFTY50 screen made 1,920 doomed calls and slept 64 minutes in
    backoff before producing the deterministic result it had all along.
    """
    from tradingagents_in.llm.structured import is_auth_failure, is_retryable

    for exc in (
        _err("Error code: 401 - {'type': 'authentication_error'}", 401),
        _err("invalid x-api-key", 401),
        _err("permission_error", 403),
        _err("Unauthorized"),
    ):
        assert is_auth_failure(exc), exc
        assert not is_retryable(exc), exc


def test_transient_failures_are_still_retried():
    """Silencing the hopeless cases must not silence the recoverable ones."""
    from tradingagents_in.llm.structured import is_auth_failure, is_retryable

    for exc in (
        _err("rate_limit_error", 429),
        _err("internal server error", 500),
        _err("overloaded_error", 529),
        _err("Request timed out"),
        _err("Connection reset by peer"),
    ):
        assert not is_auth_failure(exc), exc
        assert is_retryable(exc), exc


def test_a_malformed_request_is_not_retried():
    from tradingagents_in.llm.structured import is_retryable

    assert not is_retryable(_err("invalid_request_error", 400))
    assert not is_retryable(_err("model_not_found", 404))


def test_a_rejected_key_stops_every_later_call(monkeypatch):
    """One 401 latches the provider for the process.

    Without the latch each agent on each name still makes its own doomed
    call: 640 of them in one NIFTY50 screen, for nothing.
    """
    from tradingagents_in.llm import structured

    monkeypatch.setattr(structured, "_CREDENTIALS_REJECTED", {})
    monkeypatch.setattr(structured, "_REPORTED_UNAVAILABLE", set())

    attempts = {"n": 0}

    class Rejecting:
        status_code = 401

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            attempts["n"] += 1
            raise _err("Error code: 401 - invalid x-api-key", 401)

    monkeypatch.setattr(structured, "_get_model", lambda cfg, tier: Rejecting())

    class Draft(BaseModel):
        value: int = 1

    cfg = {"llm_provider": "anthropic"}
    for _ in range(25):
        result, meta = structured.call_structured(
            cfg, Draft, "system", "user", fallback=Draft()
        )
        assert result.value == 1, "the deterministic draft must still come back"
        assert meta.fell_back

    assert attempts["n"] == 1, f"expected one doomed call, made {attempts['n']}"
