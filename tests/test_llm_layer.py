"""The structured-output layer, exercised with a fake chat model.

The rest of the suite runs offline, which never touches this code path. These
tests cover what actually happens when a provider is configured: schema
validation, the retry-with-the-error loop, and the fall back to the
deterministic draft when the model cannot be salvaged.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from tradingagents_in.llm import structured
from tradingagents_in.llm.provider import (
    PROVIDERS,
    LLMUnavailable,
    available_providers,
    describe,
    provider_is_offline,
    resolve_model,
)


class Answer(BaseModel):
    verdict: str
    score: float = Field(ge=0.0, le=1.0)


DRAFT = Answer(verdict="deterministic baseline", score=0.4)
LLM_CONFIG = {
    "llm_provider": "openai",
    "deep_think_model": "test-deep",
    "quick_think_model": "test-quick",
    "llm_max_retries": 3,
}


class FakeStructured:
    """Stands in for `model.with_structured_output(schema)`."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        response = self.responses.pop(0) if self.responses else self.responses
        if isinstance(response, Exception):
            raise response
        return response


class FakeModel:
    def __init__(self, responses: list):
        self.structured = FakeStructured(responses)

    def with_structured_output(self, _schema):
        return self.structured

    def invoke(self, _messages):
        class Response:
            content = "plain text answer"

        return Response()


@pytest.fixture(autouse=True)
def _clear_cache():
    structured.reset_model_cache()
    yield
    structured.reset_model_cache()


def install(monkeypatch, model: FakeModel) -> FakeModel:
    monkeypatch.setattr(structured, "_get_model", lambda config, tier: model)
    return model


class TestStructuredCalls:
    def test_valid_response_is_returned(self, monkeypatch):
        install(monkeypatch, FakeModel([Answer(verdict="from the model", score=0.9)]))

        result, meta = structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="sys",
            user_prompt="user", fallback=DRAFT,
        )
        assert result.verdict == "from the model"
        assert meta.used_llm and not meta.fell_back and meta.attempts == 1
        assert meta.model == "test-quick"

    def test_the_draft_is_sent_as_a_factual_anchor(self, monkeypatch):
        model = install(monkeypatch, FakeModel([Answer(verdict="ok", score=0.5)]))

        structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="sys",
            user_prompt="user", fallback=DRAFT,
        )
        sent = "".join(text for _role, text in model.structured.calls[0])
        assert "deterministic baseline" in sent
        assert "factual anchor" in sent

    def test_a_dict_response_is_validated_into_the_schema(self, monkeypatch):
        install(monkeypatch, FakeModel([{"verdict": "dict form", "score": 0.3}]))

        result, meta = structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="s", user_prompt="u", fallback=DRAFT,
        )
        assert isinstance(result, Answer) and result.verdict == "dict form"
        assert meta.used_llm

    def test_schema_violation_is_retried_with_the_error(self, monkeypatch):
        bad = ValidationError.from_exception_data("Answer", [])
        model = install(
            monkeypatch, FakeModel([bad, Answer(verdict="second attempt", score=0.6)])
        )

        result, meta = structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="s", user_prompt="u", fallback=DRAFT,
        )
        assert result.verdict == "second attempt"
        assert meta.attempts == 2
        # The retry must actually tell the model what was wrong.
        retry_prompt = "".join(text for _role, text in model.structured.calls[1])
        assert "did not satisfy the schema" in retry_prompt

    def test_exhausted_retries_fall_back_to_the_draft(self, monkeypatch):
        install(monkeypatch, FakeModel([RuntimeError("boom")] * 3))
        monkeypatch.setattr(structured.time, "sleep", lambda _s: None)

        result, meta = structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="s", user_prompt="u",
            fallback=DRAFT, max_attempts=3,
        )
        assert result is DRAFT
        assert meta.fell_back and not meta.used_llm and meta.error

    def test_provider_failure_falls_back_without_calling(self, monkeypatch):
        def unavailable(_config, _tier):
            raise LLMUnavailable("no key")

        monkeypatch.setattr(structured, "_get_model", unavailable)

        result, meta = structured.call_structured(
            config=LLM_CONFIG, schema=Answer, system_prompt="s", user_prompt="u", fallback=DRAFT,
        )
        assert result is DRAFT and meta.fell_back

    def test_no_fallback_and_no_provider_raises(self, monkeypatch):
        def unavailable(_config, _tier):
            raise LLMUnavailable("no key")

        monkeypatch.setattr(structured, "_get_model", unavailable)
        with pytest.raises(LLMUnavailable):
            structured.call_structured(
                config=LLM_CONFIG, schema=Answer, system_prompt="s", user_prompt="u",
            )

    def test_offline_returns_the_draft_without_a_model(self):
        result, meta = structured.call_structured(
            config={"llm_provider": "offline"}, schema=Answer,
            system_prompt="s", user_prompt="u", fallback=DRAFT,
        )
        assert result is DRAFT
        assert not meta.used_llm and not meta.fell_back

    def test_offline_without_a_draft_is_an_error(self):
        with pytest.raises(LLMUnavailable):
            structured.call_structured(
                config={"llm_provider": "offline"}, schema=Answer,
                system_prompt="s", user_prompt="u",
            )

    def test_text_calls_fall_back_quietly(self, monkeypatch):
        assert structured.call_text(
            config={"llm_provider": "offline"}, system_prompt="s",
            user_prompt="u", fallback="baseline",
        ) == "baseline"


class TestProviderRegistry:
    def test_every_provider_declares_defaults(self):
        for name, spec in PROVIDERS.items():
            if name == "offline":
                continue
            assert spec.default_deep and spec.default_quick, name

    def test_the_documented_providers_are_all_present(self):
        expected = {
            "openai", "anthropic", "google", "xai", "deepseek",
            "qwen", "groq", "ollama", "openrouter", "offline",
        }
        assert expected <= set(PROVIDERS)

    def test_tiers_resolve_to_the_configured_models(self):
        assert resolve_model(LLM_CONFIG, "deep") == "test-deep"
        assert resolve_model(LLM_CONFIG, "quick") == "test-quick"

    def test_tiers_fall_back_to_provider_defaults(self):
        config = {"llm_provider": "anthropic", "deep_think_model": None,
                  "quick_think_model": None}
        assert resolve_model(config, "deep") == PROVIDERS["anthropic"].default_deep

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(LLMUnavailable):
            resolve_model({"llm_provider": "nonesuch"}, "deep")

    def test_offline_is_detected(self):
        assert provider_is_offline({"llm_provider": "offline"})
        assert not provider_is_offline({"llm_provider": "openai"})

    def test_offline_and_ollama_need_no_credentials(self):
        ready = available_providers()
        assert "offline" in ready and "ollama" in ready

    def test_describe_is_readable(self):
        assert "offline" in describe({"llm_provider": "offline"})
        assert "test-deep" in describe(LLM_CONFIG)
