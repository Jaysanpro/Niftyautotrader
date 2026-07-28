"""Symbol resolution for NSE (.NS) and BSE (.BO).

Users type ``RELIANCE``, ``reliance.ns``, ``NSE:RELIANCE``, ``^NSEI`` or
``NIFTY``. Everything downstream wants one canonical :class:`SymbolRef`.
"""

from __future__ import annotations

import re

from tradingagents_in.config.universe import (
    INDICES,
    is_fno_eligible,
    lot_size,
    sector_for,
)
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.models.enums import Exchange

_NS_SUFFIX = ".NS"
_BO_SUFFIX = ".BO"

# Common shorthands people actually type for the indices.
_INDEX_ALIASES: dict[str, str] = {
    "NIFTY": "NIFTY50",
    "NIFTY50": "NIFTY50",
    "NIFTY 50": "NIFTY50",
    "^NSEI": "NIFTY50",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "^NSEBANK": "BANKNIFTY",
    "SENSEX": "SENSEX",
    "^BSESN": "SENSEX",
    "BSESENSEX": "SENSEX",
    "FINNIFTY": "NIFTYFINSERVICE",
    "NIFTYIT": "NIFTYIT",
    "NIFTY IT": "NIFTYIT",
    "NIFTYPHARMA": "NIFTYPHARMA",
    "NIFTY PHARMA": "NIFTYPHARMA",
    "NIFTYMIDCAP": "NIFTYMIDCAP",
    "MIDCAP": "NIFTYMIDCAP",
    "INDIAVIX": "INDIAVIX",
    "VIX": "INDIAVIX",
}

# BSE scrip codes are numeric; yfinance still wants the .BO alpha symbol, so we
# only use this to detect that the user meant BSE.
_NUMERIC = re.compile(r"^\d{6}$")


class SymbolError(ValueError):
    """Raised when a user-supplied symbol cannot be resolved."""


def normalise(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip()).upper()


def split_exchange_prefix(text: str) -> tuple[str, Exchange | None]:
    """Handle TradingView-style ``NSE:RELIANCE`` / ``BSE:500325``."""
    if ":" in text:
        prefix, rest = text.split(":", 1)
        prefix = prefix.strip().upper()
        if prefix in ("NSE", "NSEI"):
            return rest.strip(), Exchange.NSE
        if prefix in ("BSE", "BOM"):
            return rest.strip(), Exchange.BSE
        return rest.strip(), None
    return text, None


def base_symbol(symbol: str) -> str:
    """Strip a .NS/.BO suffix and any exchange prefix."""
    text, _ = split_exchange_prefix(normalise(symbol))
    if text.endswith(_NS_SUFFIX) or text.endswith(_BO_SUFFIX):
        text = text[:-3]
    return text


def to_yf(symbol: str, exchange: Exchange = Exchange.NSE) -> str:
    """Canonical yfinance symbol for a bare base symbol."""
    text = normalise(symbol)
    if text.startswith("^") or text.endswith((_NS_SUFFIX, _BO_SUFFIX)):
        return text
    return text + (_BO_SUFFIX if exchange == Exchange.BSE else _NS_SUFFIX)


def resolve(symbol: str, default_exchange: Exchange = Exchange.NSE) -> SymbolRef:
    """Resolve any user input to a :class:`SymbolRef`.

    >>> resolve("reliance").yf_symbol
    'RELIANCE.NS'
    >>> resolve("TCS.BO").exchange
    <Exchange.BSE: 'BSE'>
    >>> resolve("nifty").is_index
    True
    """
    if not symbol or not symbol.strip():
        raise SymbolError("empty symbol")

    text, prefix_exchange = split_exchange_prefix(normalise(symbol))
    exchange = prefix_exchange or default_exchange

    # 1. Index?
    alias = _INDEX_ALIASES.get(text) or _INDEX_ALIASES.get(text.replace(" ", ""))
    if alias:
        meta = INDICES[alias]
        return SymbolRef(
            base_symbol=alias,
            yf_symbol=meta["yf"],
            exchange=Exchange(meta["exchange"]),
            name=meta["label"],
            sector="Index",
            is_index=True,
            fno_eligible=alias in ("NIFTY50", "BANKNIFTY", "SENSEX", "NIFTYFINSERVICE"),
            lot_size=lot_size({"NIFTY50": "NIFTY", "BANKNIFTY": "BANKNIFTY",
                               "SENSEX": "SENSEX"}.get(alias, alias)),
        )
    if text.startswith("^"):
        for key, meta in INDICES.items():
            if meta["yf"] == text:
                return SymbolRef(
                    base_symbol=key, yf_symbol=text, exchange=Exchange(meta["exchange"]),
                    name=meta["label"], sector="Index", is_index=True,
                )
        return SymbolRef(base_symbol=text.lstrip("^"), yf_symbol=text, is_index=True, sector="Index")

    # 2. Explicit suffix wins over the default exchange.
    if text.endswith(_NS_SUFFIX):
        exchange, text = Exchange.NSE, text[:-3]
    elif text.endswith(_BO_SUFFIX):
        exchange, text = Exchange.BSE, text[:-3]
    elif _NUMERIC.match(text):
        # Six-digit BSE scrip code. We cannot map it to an alpha symbol offline;
        # yfinance accepts <code>.BO for most listed scrips.
        exchange = Exchange.BSE

    if not text:
        raise SymbolError(f"could not parse symbol {symbol!r}")

    return SymbolRef(
        base_symbol=text,
        yf_symbol=text + (_BO_SUFFIX if exchange == Exchange.BSE else _NS_SUFFIX),
        exchange=exchange,
        sector=sector_for(text),
        fno_eligible=is_fno_eligible(text),
        lot_size=lot_size(text),
    )


def sibling_symbol(ref: SymbolRef) -> str | None:
    """The same company on the other exchange, for cross-venue sanity checks."""
    if ref.is_index:
        return None
    other = _BO_SUFFIX if ref.exchange == Exchange.NSE else _NS_SUFFIX
    return ref.base_symbol + other


def resolve_many(symbols: list[str], default_exchange: Exchange = Exchange.NSE) -> list[SymbolRef]:
    resolved: list[SymbolRef] = []
    for sym in symbols:
        try:
            resolved.append(resolve(sym, default_exchange))
        except SymbolError:
            continue
    return resolved
