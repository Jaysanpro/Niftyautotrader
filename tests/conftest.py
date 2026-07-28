"""Shared fixtures.

Every test runs against the synthetic data provider with the offline LLM
engine, so the suite is deterministic, needs no API key and makes no network
call. That is a hard requirement: a test that depends on what Reliance did this
morning is not a test.
"""

from __future__ import annotations

import pytest

from tradingagents_in.config.default_config import build_config
from tradingagents_in.dataflows.synthetic import build_synthetic_bundle
from tradingagents_in.models.risk import PortfolioSnapshot

BASE_OVERRIDES = {
    "llm_provider": "offline",
    "data_provider": "synthetic",
    "online_tools": False,
    "enable_memory": False,
    "log_level": "ERROR",
    "synthetic_seed": 20260101,
    "technical_timeframes": ["1d", "1h"],
}


@pytest.fixture
def config() -> dict:
    return build_config(dict(BASE_OVERRIDES))


@pytest.fixture
def config_factory():
    def make(**overrides) -> dict:
        return build_config({**BASE_OVERRIDES, **overrides})

    return make


@pytest.fixture
def bundle(config):
    return build_synthetic_bundle("RELIANCE.NS", config)


@pytest.fixture
def index_bundle(config):
    return build_synthetic_bundle("NIFTY", config)


@pytest.fixture
def flat_book(config) -> PortfolioSnapshot:
    capital = float(config["capital_inr"])
    return PortfolioSnapshot(capital_inr=capital, cash_inr=capital)


@pytest.fixture
def analysed(config):
    """A completed run, shared by the tests that assert on the final decision."""
    from tradingagents_in.graph.runner import IndianTradingAgents

    app = IndianTradingAgents(config=config, use_langgraph=False)
    decision = app.analyse("RELIANCE.NS", save=False)
    return app, decision
