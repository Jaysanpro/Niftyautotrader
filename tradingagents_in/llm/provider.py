"""Multi-provider LLM factory.

Supported: openai, anthropic, google, xai, deepseek, qwen, groq, ollama,
openrouter, and ``offline`` (no LLM at all - the deterministic engine runs
instead, which is what the test suite and the no-key demo path use).

Providers that expose an OpenAI-compatible endpoint are routed through
``langchain-openai`` with a base_url override, so adding another one is a
single table entry rather than a new integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    env_key: str | None
    base_url: str | None = None
    openai_compatible: bool = True
    default_deep: str = ""
    default_quick: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        env_key="OPENAI_API_KEY",
        default_deep="gpt-4.1",
        default_quick="gpt-4.1-mini",
    ),
    "anthropic": ProviderSpec(
        env_key="ANTHROPIC_API_KEY",
        openai_compatible=False,
        default_deep="claude-sonnet-4-5",
        default_quick="claude-haiku-4-5",
    ),
    "google": ProviderSpec(
        env_key="GOOGLE_API_KEY",
        openai_compatible=False,
        default_deep="gemini-2.5-pro",
        default_quick="gemini-2.5-flash",
    ),
    "xai": ProviderSpec(
        env_key="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_deep="grok-4",
        default_quick="grok-3-mini",
    ),
    "deepseek": ProviderSpec(
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        default_deep="deepseek-reasoner",
        default_quick="deepseek-chat",
    ),
    "qwen": ProviderSpec(
        env_key="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_deep="qwen-max",
        default_quick="qwen-plus",
    ),
    "groq": ProviderSpec(
        env_key="GROQ_API_KEY",
        openai_compatible=False,
        default_deep="llama-3.3-70b-versatile",
        default_quick="llama-3.1-8b-instant",
    ),
    "openrouter": ProviderSpec(
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_deep="anthropic/claude-sonnet-4.5",
        default_quick="anthropic/claude-haiku-4.5",
    ),
    "ollama": ProviderSpec(
        env_key=None,  # local, no key
        base_url="http://localhost:11434",
        openai_compatible=False,
        default_deep="qwen2.5:32b",
        default_quick="qwen2.5:7b",
    ),
    "offline": ProviderSpec(env_key=None, openai_compatible=False),
}


class LLMUnavailable(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


def provider_is_offline(config: dict[str, Any]) -> bool:
    return str(config.get("llm_provider", "")).lower() in ("offline", "none", "")


def resolve_model(config: dict[str, Any], tier: str) -> str:
    """Model id for a tier ('deep' or 'quick'), falling back to provider defaults."""
    key = "deep_think_model" if tier == "deep" else "quick_think_model"
    configured = config.get(key)
    if configured:
        return str(configured)
    spec = PROVIDERS.get(str(config.get("llm_provider", "")).lower())
    if spec is None:
        raise LLMUnavailable(f"unknown provider {config.get('llm_provider')!r}")
    return spec.default_deep if tier == "deep" else spec.default_quick


def build_chat_model(config: dict[str, Any], tier: str = "quick") -> Any:
    """Construct a LangChain chat model for the configured provider and tier."""
    provider = str(config.get("llm_provider", "offline")).lower()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMUnavailable(f"unknown llm_provider {provider!r}")
    if provider == "offline":
        raise LLMUnavailable("offline provider has no chat model")

    if spec.env_key and not os.getenv(spec.env_key):
        raise LLMUnavailable(
            f"{spec.env_key} is not set. Export it, or set ITA_LLM_PROVIDER=offline "
            "to run the deterministic engine."
        )

    model = resolve_model(config, tier)
    common: dict[str, Any] = {
        "temperature": float(config.get("llm_temperature", 0.2)),
        "timeout": config.get("llm_timeout_seconds", 120),
        "max_retries": int(config.get("llm_max_retries", 3)),
    }
    max_tokens = config.get("llm_max_tokens")

    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            return ChatAnthropic(model=model, max_tokens=max_tokens or 4096, **common)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=common["temperature"],
                max_output_tokens=max_tokens,
                timeout=common["timeout"],
            )

        if provider == "groq":
            from langchain_groq import ChatGroq  # noqa: PLC0415

            return ChatGroq(model=model, max_tokens=max_tokens, **common)

        if provider == "ollama":
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            return ChatOllama(
                model=model,
                temperature=common["temperature"],
                base_url=config.get("llm_base_url") or os.getenv("OLLAMA_BASE_URL")
                or spec.base_url,
            )

        # Everything else speaks the OpenAI wire format.
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(
            model=model,
            base_url=config.get("llm_base_url") or spec.base_url,
            api_key=os.getenv(spec.env_key) if spec.env_key else "not-needed",
            max_tokens=max_tokens,
            **common,
        )
    except ImportError as exc:
        raise LLMUnavailable(
            f"LangChain integration for {provider!r} is not installed: {exc}. "
            "Install with `pip install 'tradingagents-in[llm]'`."
        ) from exc


def describe(config: dict[str, Any]) -> str:
    provider = str(config.get("llm_provider", "offline")).lower()
    if provider == "offline":
        return "offline (deterministic engine, no LLM calls)"
    return (
        f"{provider}: deep={resolve_model(config, 'deep')}, "
        f"quick={resolve_model(config, 'quick')}"
    )


def available_providers() -> list[str]:
    """Providers whose credentials are actually present in the environment."""
    ready = ["offline"]
    for name, spec in PROVIDERS.items():
        if name == "offline":
            continue
        if spec.env_key is None or os.getenv(spec.env_key):
            ready.append(name)
    return ready
