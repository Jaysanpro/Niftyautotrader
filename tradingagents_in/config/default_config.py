"""DEFAULT_CONFIG and the .env overlay.

Precedence, lowest to highest:

    DEFAULT_CONFIG  <  environment variables (ITA_*)  <  explicit overrides

Every key is flat and JSON-serialisable so a run's exact configuration can be
persisted alongside its decision.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

try:  # optional dependency, only used to hydrate os.environ from a .env file
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass


DEFAULT_CONFIG: dict[str, Any] = {
    # ------------------------------------------------------------------ core
    "project_name": "IndianTradingAgents",
    "timezone": "Asia/Kolkata",
    "currency": "INR",
    "results_dir": "results",
    "cache_dir": ".cache/tradingagents_in",
    "cache_ttl_seconds": 900,
    # ------------------------------------------------------------------- llm
    "llm_provider": "anthropic",
    "llm_base_url": None,
    "deep_think_model": "claude-sonnet-4-5",
    "quick_think_model": "claude-haiku-4-5",
    "llm_temperature": 0.2,
    "llm_max_tokens": 4096,
    "llm_timeout_seconds": 120,
    "llm_max_retries": 3,
    # Per-agent model tier. "deep" -> deep_think_model, "quick" -> quick_think_model.
    "agent_model_tier": {
        "fundamentals_analyst": "quick",
        "sentiment_analyst": "quick",
        "news_macro_analyst": "quick",
        "technical_analyst": "quick",
        "bull_researcher": "deep",
        "bear_researcher": "deep",
        "research_manager": "deep",
        "trader": "deep",
        "risk_manager": "deep",
        "portfolio_manager": "deep",
    },
    # -------------------------------------------------------------- research
    "selected_analysts": [
        "fundamentals",
        "sentiment",
        "news_macro",
        "technical",
    ],
    "max_debate_rounds": 2,
    "max_risk_rounds": 1,
    "online_tools": True,
    "lookback_days": 400,
    "technical_timeframes": ["1d", "1h", "15m"],
    "peer_count": 5,
    # ------------------------------------------------------- capital & risk
    "profile": "retail",  # retail | institutional
    "capital_inr": 1_000_000.0,
    "risk_per_trade_pct": 1.0,
    "max_position_pct": 10.0,
    "max_sector_pct": 25.0,
    "max_portfolio_heat_pct": 6.0,
    "max_drawdown_pct": 15.0,
    "min_reward_risk": 1.5,
    "max_open_positions": 12,
    # India instrument permissions
    "allow_fno": True,
    "allow_overnight_fno": True,
    "allow_short_cash": False,  # cash-segment shorts must be squared off intraday
    "allow_expiry_day_entry": False,
    "avoid_fno_ban_period": True,
    "avoid_asm_gsm": True,
    "min_avg_daily_turnover_inr": 5_00_00_000.0,  # Rs 5 crore
    # Costs (used for net R:R sanity checks). Percentages of turnover.
    "costs": {
        "brokerage_pct_delivery": 0.0,
        "brokerage_pct_intraday": 0.03,
        "stt_pct_delivery_buy": 0.1,
        "stt_pct_delivery_sell": 0.1,
        "stt_pct_intraday_sell": 0.025,
        "stt_pct_futures_sell": 0.02,
        "stt_pct_options_sell_premium": 0.1,
        "exchange_txn_pct": 0.00297,
        "sebi_turnover_pct": 0.0001,
        "stamp_duty_pct_buy_delivery": 0.015,
        "gst_pct_on_charges": 18.0,
    },
    # -------------------------------------------------------------- data
    # yfinance | angelone | zerodha | synthetic.
    # "synthetic" is a deterministic, network-free tape used by the test
    # suite and the reproducible demo. It is a seeded random walk, not a
    # market simulator - never validate a strategy against it.
    "data_provider": "yfinance",
    "synthetic_seed": None,  # None derives a stable seed from the symbol
    "enable_screener_scrape": False,
    "news_lookback_days": 7,
    "max_news_items": 40,
    "benchmark_symbol": "^NSEI",
    # ------------------------------------------------------- persistence
    "memory_file": "trading_memory.md",
    "enable_memory": True,
    "memory_recall_count": 5,
    "enable_checkpoints": False,
    "checkpoint_db": "checkpoints/graph.sqlite",
    # ---------------------------------------------------------------- misc
    "log_level": "INFO",
    "debug_prompts": False,
}


# Mapping of ITA_* env var -> (config key, caster)
_ENV_MAP: dict[str, tuple[str, Any]] = {
    "ITA_LLM_PROVIDER": ("llm_provider", str),
    "ITA_LLM_BASE_URL": ("llm_base_url", str),
    "ITA_DEEP_THINK_MODEL": ("deep_think_model", str),
    "ITA_QUICK_THINK_MODEL": ("quick_think_model", str),
    "ITA_LLM_TEMPERATURE": ("llm_temperature", float),
    "ITA_LLM_MAX_TOKENS": ("llm_max_tokens", int),
    "ITA_MAX_DEBATE_ROUNDS": ("max_debate_rounds", int),
    "ITA_MAX_RISK_ROUNDS": ("max_risk_rounds", int),
    "ITA_ONLINE_TOOLS": ("online_tools", bool),
    "ITA_LOOKBACK_DAYS": ("lookback_days", int),
    "ITA_CAPITAL_INR": ("capital_inr", float),
    "ITA_PROFILE": ("profile", str),
    "ITA_RISK_PER_TRADE_PCT": ("risk_per_trade_pct", float),
    "ITA_MAX_POSITION_PCT": ("max_position_pct", float),
    "ITA_MAX_SECTOR_PCT": ("max_sector_pct", float),
    "ITA_MAX_PORTFOLIO_HEAT_PCT": ("max_portfolio_heat_pct", float),
    "ITA_MAX_DRAWDOWN_PCT": ("max_drawdown_pct", float),
    "ITA_ALLOW_FNO": ("allow_fno", bool),
    "ITA_ALLOW_OVERNIGHT_FNO": ("allow_overnight_fno", bool),
    "ITA_ALLOW_SHORT_CASH": ("allow_short_cash", bool),
    "ITA_DATA_PROVIDER": ("data_provider", str),
    "ITA_SYNTHETIC_SEED": ("synthetic_seed", int),
    "ITA_ENABLE_SCREENER_SCRAPE": ("enable_screener_scrape", bool),
    "ITA_CACHE_DIR": ("cache_dir", str),
    "ITA_CACHE_TTL_SECONDS": ("cache_ttl_seconds", int),
    "ITA_RESULTS_DIR": ("results_dir", str),
    "ITA_MEMORY_FILE": ("memory_file", str),
    "ITA_CHECKPOINT_DB": ("checkpoint_db", str),
    "ITA_ENABLE_CHECKPOINTS": ("enable_checkpoints", bool),
    "ITA_LOG_LEVEL": ("log_level", str),
}

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _cast(raw: str, caster: Any) -> Any:
    if caster is bool:
        return raw.strip().lower() in _TRUTHY
    return caster(raw)


# Institutional desks run tighter single-name limits but carry more names and
# tolerate a deeper aggregate drawdown than a retail book.
_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "retail": {},
    "institutional": {
        "risk_per_trade_pct": 0.5,
        "max_position_pct": 5.0,
        "max_sector_pct": 20.0,
        "max_portfolio_heat_pct": 4.0,
        "max_open_positions": 40,
        "min_avg_daily_turnover_inr": 25_00_00_000.0,
    },
}


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a config dict: DEFAULT_CONFIG <- env <- profile <- overrides."""
    cfg = deepcopy(DEFAULT_CONFIG)

    for env_key, (cfg_key, caster) in _ENV_MAP.items():
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        try:
            cfg[cfg_key] = _cast(raw, caster)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for {env_key}: {raw!r}") from None

    # Profile defaults apply before explicit overrides so a caller can still
    # widen an institutional limit deliberately.
    profile_key = str((overrides or {}).get("profile") or cfg["profile"]).lower()
    cfg.update(deepcopy(_PROFILE_OVERRIDES.get(profile_key, {})))
    cfg["profile"] = profile_key

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                merged = deepcopy(cfg[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    if cfg["capital_inr"] <= 0:
        raise ValueError("capital_inr must be positive")
    if not 0 < cfg["risk_per_trade_pct"] <= 10:
        raise ValueError("risk_per_trade_pct must be in (0, 10]")
    if cfg["max_position_pct"] > 100 or cfg["max_position_pct"] <= 0:
        raise ValueError("max_position_pct must be in (0, 100]")
    if cfg["max_debate_rounds"] < 1:
        raise ValueError("max_debate_rounds must be >= 1")
    if cfg["profile"] not in _PROFILE_OVERRIDES:
        raise ValueError(f"unknown profile {cfg['profile']!r}")
    unknown = set(cfg["selected_analysts"]) - {
        "fundamentals",
        "sentiment",
        "news_macro",
        "technical",
    }
    if unknown:
        raise ValueError(f"unknown analysts: {sorted(unknown)}")
