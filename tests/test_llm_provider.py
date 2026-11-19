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
