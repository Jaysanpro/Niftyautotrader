"""The Indian market universe: indices, sectors, index constituents, lot sizes.

Everything here is *reference data with a shelf life*. Index constituents get
reshuffled twice a year and F&O lot sizes are revised by the exchanges, so each
table carries an `AS_OF` date and the loaders prefer a user-supplied override
file (``ITA_UNIVERSE_FILE``) when one exists.
"""

from __future__ import annotations

import json
import os
from typing import Any

AS_OF = "2025-09-30"

# --------------------------------------------------------------------------
# Indices
# --------------------------------------------------------------------------
# yfinance symbols for the benchmarks an Indian desk actually watches.
INDICES: dict[str, dict[str, str]] = {
    "NIFTY50": {"yf": "^NSEI", "exchange": "NSE", "label": "Nifty 50"},
    "BANKNIFTY": {"yf": "^NSEBANK", "exchange": "NSE", "label": "Nifty Bank"},
    "SENSEX": {"yf": "^BSESN", "exchange": "BSE", "label": "S&P BSE Sensex"},
    "NIFTYNEXT50": {"yf": "^NSMIDCP", "exchange": "NSE", "label": "Nifty Next 50"},
    "NIFTYMIDCAP": {"yf": "NIFTY_MIDCAP_100.NS", "exchange": "NSE", "label": "Nifty Midcap 100"},
    "NIFTYSMALLCAP": {"yf": "NIFTYSMLCAP250.NS", "exchange": "NSE", "label": "Nifty Smallcap 250"},
    "NIFTYIT": {"yf": "^CNXIT", "exchange": "NSE", "label": "Nifty IT"},
    "NIFTYPHARMA": {"yf": "^CNXPHARMA", "exchange": "NSE", "label": "Nifty Pharma"},
    "NIFTYAUTO": {"yf": "^CNXAUTO", "exchange": "NSE", "label": "Nifty Auto"},
    "NIFTYFMCG": {"yf": "^CNXFMCG", "exchange": "NSE", "label": "Nifty FMCG"},
    "NIFTYMETAL": {"yf": "^CNXMETAL", "exchange": "NSE", "label": "Nifty Metal"},
    "NIFTYENERGY": {"yf": "^CNXENERGY", "exchange": "NSE", "label": "Nifty Energy"},
    "NIFTYREALTY": {"yf": "^CNXREALTY", "exchange": "NSE", "label": "Nifty Realty"},
    "NIFTYPSUBANK": {"yf": "^CNXPSUBANK", "exchange": "NSE", "label": "Nifty PSU Bank"},
    "NIFTYFINSERVICE": {"yf": "NIFTY_FIN_SERVICE.NS", "exchange": "NSE", "label": "Nifty Fin Services"},
    "INDIAVIX": {"yf": "^INDIAVIX", "exchange": "NSE", "label": "India VIX"},
}

DEFAULT_INDEX = "NIFTY50"

# Index-level circuit breakers (SEBI): a move in Nifty 50 *or* Sensex of these
# magnitudes halts the whole market. Halt length depends on time of day.
INDEX_CIRCUIT_BANDS_PCT = (10.0, 15.0, 20.0)

# --------------------------------------------------------------------------
# Macro instruments the News & Macro analyst tracks
# --------------------------------------------------------------------------
MACRO_SYMBOLS: dict[str, str] = {
    "usdinr": "INR=X",
    "brent_crude": "BZ=F",
    "wti_crude": "CL=F",
    "gold": "GC=F",
    "us_sp500": "^GSPC",
    "us_nasdaq": "^IXIC",
    "us_vix": "^VIX",
    "us_10y": "^TNX",
    "dollar_index": "DX-Y.NYB",
    "hangseng": "^HSI",
    "shanghai": "000001.SS",
    "nikkei": "^N225",
    "india_vix": "^INDIAVIX",
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex": "^BSESN",
}

# --------------------------------------------------------------------------
# Sectors (NIC-flavoured buckets used for concentration limits)
# --------------------------------------------------------------------------
SECTORS = (
    "Financials",
    "Information Technology",
    "Energy",
    "Consumer Staples",
    "Consumer Discretionary",
    "Automobile",
    "Healthcare",
    "Materials",
    "Industrials",
    "Utilities",
    "Telecom",
    "Realty",
    "Others",
)

# --------------------------------------------------------------------------
# Nifty 50 constituents (NSE base symbols, no suffix).
# --------------------------------------------------------------------------
NIFTY_50: tuple[str, ...] = (
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
)

BANK_NIFTY: tuple[str, ...] = (
    "AXISBANK", "AUBANK", "BANKBARODA", "CANBK", "FEDERALBNK",
    "HDFCBANK", "ICICIBANK", "IDFCFIRSTB", "INDUSINDBK", "KOTAKBANK",
    "PNB", "SBIN",
)

NIFTY_IT: tuple[str, ...] = (
    "COFORGE", "HCLTECH", "INFY", "LTIM", "MPHASIS",
    "OFSS", "PERSISTENT", "TCS", "TECHM", "WIPRO",
)

NIFTY_PHARMA: tuple[str, ...] = (
    "ABBOTINDIA", "ALKEM", "AUROPHARMA", "BIOCON", "CIPLA",
    "DIVISLAB", "DRREDDY", "GLAND", "GRANULES", "IPCALAB",
    "LAURUSLABS", "LUPIN", "MANKIND", "NATCOPHARM", "SUNPHARMA",
    "TORNTPHARM", "ZYDUSLIFE",
)

INDEX_CONSTITUENTS: dict[str, tuple[str, ...]] = {
    "NIFTY50": NIFTY_50,
    "BANKNIFTY": BANK_NIFTY,
    "NIFTYIT": NIFTY_IT,
    "NIFTYPHARMA": NIFTY_PHARMA,
}

# --------------------------------------------------------------------------
# Sector map for the liquid large caps. Anything absent falls back to the
# provider-reported sector, then to "Others".
# --------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Financials
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "AXISBANK": "Financials", "KOTAKBANK": "Financials", "INDUSINDBK": "Financials",
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "SHRIRAMFIN": "Financials",
    "HDFCLIFE": "Financials", "SBILIFE": "Financials", "JIOFIN": "Financials",
    "BANKBARODA": "Financials", "PNB": "Financials", "CANBK": "Financials",
    "AUBANK": "Financials", "FEDERALBNK": "Financials", "IDFCFIRSTB": "Financials",
    # IT
    "TCS": "Information Technology", "INFY": "Information Technology",
    "HCLTECH": "Information Technology", "WIPRO": "Information Technology",
    "TECHM": "Information Technology", "LTIM": "Information Technology",
    "COFORGE": "Information Technology", "PERSISTENT": "Information Technology",
    "MPHASIS": "Information Technology", "OFSS": "Information Technology",
    # Energy / utilities
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy",
    "COALINDIA": "Energy", "GAIL": "Energy",
    "NTPC": "Utilities", "POWERGRID": "Utilities", "TATAPOWER": "Utilities",
    # Consumer
    "HINDUNILVR": "Consumer Staples", "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples", "BRITANNIA": "Consumer Staples",
    "TATACONSUM": "Consumer Staples", "DABUR": "Consumer Staples",
    "TITAN": "Consumer Discretionary", "TRENT": "Consumer Discretionary",
    "ETERNAL": "Consumer Discretionary", "DMART": "Consumer Discretionary",
    "ASIANPAINT": "Consumer Discretionary",
    # Autos
    "MARUTI": "Automobile", "M&M": "Automobile", "TATAMOTORS": "Automobile",
    "BAJAJ-AUTO": "Automobile", "HEROMOTOCO": "Automobile", "EICHERMOT": "Automobile",
    "TVSMOTOR": "Automobile",
    # Healthcare
    "SUNPHARMA": "Healthcare", "CIPLA": "Healthcare", "DRREDDY": "Healthcare",
    "DIVISLAB": "Healthcare", "APOLLOHOSP": "Healthcare", "LUPIN": "Healthcare",
    "TORNTPHARM": "Healthcare", "ZYDUSLIFE": "Healthcare", "BIOCON": "Healthcare",
    "ALKEM": "Healthcare", "MANKIND": "Healthcare", "AUROPHARMA": "Healthcare",
    # Materials / industrials
    "TATASTEEL": "Materials", "JSWSTEEL": "Materials", "HINDALCO": "Materials",
    "GRASIM": "Materials", "ULTRACEMCO": "Materials", "SHREECEM": "Materials",
    "AMBUJACEM": "Materials", "VEDL": "Materials",
    "LT": "Industrials", "SIEMENS": "Industrials", "ABB": "Industrials",
    "BEL": "Industrials", "ADANIENT": "Industrials", "ADANIPORTS": "Industrials",
    "HAL": "Industrials", "CUMMINSIND": "Industrials",
    # Telecom / realty
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "INDUSTOWER": "Telecom",
    "DLF": "Realty", "GODREJPROP": "Realty", "OBEROIRLTY": "Realty",
}

# --------------------------------------------------------------------------
# F&O reference data.
# --------------------------------------------------------------------------
# Derivatives lot sizes. Exchanges revise these; treat as a starting point and
# override via the universe file for live trading.
LOT_SIZES: dict[str, int] = {
    # Indices
    "NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65, "MIDCPNIFTY": 140,
    "NIFTYNXT50": 25, "SENSEX": 20, "BANKEX": 30,
    # Single stocks (subset of the F&O universe)
    "RELIANCE": 500, "TCS": 175, "INFY": 400, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 750, "AXISBANK": 625, "KOTAKBANK": 400,
    "ITC": 1600, "LT": 150, "BHARTIARTL": 475, "HINDUNILVR": 300,
    "MARUTI": 50, "TATAMOTORS": 800, "TATASTEEL": 5500, "JSWSTEEL": 675,
    "SUNPHARMA": 350, "CIPLA": 375, "DRREDDY": 625, "WIPRO": 3000,
    "HCLTECH": 350, "TECHM": 600, "ONGC": 3850, "NTPC": 1500,
    "POWERGRID": 1900, "COALINDIA": 1350, "ADANIENT": 300, "ADANIPORTS": 475,
    "ASIANPAINT": 250, "TITAN": 175, "BAJFINANCE": 750, "BAJAJFINSV": 500,
    "ULTRACEMCO": 50, "GRASIM": 250, "HINDALCO": 1400, "INDUSINDBK": 700,
    "M&M": 200, "NESTLEIND": 250, "BRITANNIA": 125, "EICHERMOT": 100,
    "HEROMOTOCO": 150, "BAJAJ-AUTO": 75, "DIVISLAB": 100, "APOLLOHOSP": 125,
    "TRENT": 100, "BEL": 2850, "SHRIRAMFIN": 825, "HDFCLIFE": 1100,
    "SBILIFE": 375, "TATACONSUM": 550, "JIOFIN": 2350, "ETERNAL": 2425,
}

# Strike step used when the agent proposes option strikes.
STRIKE_STEPS: dict[str, float] = {
    "NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 50.0,
    "MIDCPNIFTY": 25.0, "SENSEX": 100.0, "BANKEX": 100.0,
}

# Weekly-expiry weekday per exchange (Monday=0). SEBI limits each exchange to a
# single weekly expiry; NSE settled on Thursday and BSE on Tuesday.
WEEKLY_EXPIRY_WEEKDAY: dict[str, int] = {"NSE": 3, "BSE": 1}


def index_symbol(name: str) -> str | None:
    """Return the yfinance symbol for an index alias, or None."""
    return (INDICES.get(name.upper().replace(" ", "")) or {}).get("yf")


def sector_for(base_symbol: str, fallback: str | None = None) -> str:
    """Sector bucket for an NSE base symbol."""
    hit = SECTOR_MAP.get(base_symbol.upper())
    if hit:
        return hit
    if fallback:
        for sector in SECTORS:
            if sector.lower() in fallback.lower():
                return sector
    return "Others"


# Index *names* and their derivatives *underlying* codes differ. The index is
# quoted as NIFTY 50 but the contract trades as NIFTY, and every derivatives
# lookup - lot size, strike ladder, option chain - keys off the contract code.
DERIVATIVE_UNDERLYING: dict[str, str] = {
    "NIFTY50": "NIFTY",
    "NIFTY": "NIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYFINSERVICE": "FINNIFTY",
    "FINNIFTY": "FINNIFTY",
    "NIFTYMIDCAP": "MIDCPNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "SENSEX": "SENSEX",
    "BANKEX": "BANKEX",
}


def derivatives_underlying(base_symbol: str) -> str:
    """Contract code for a symbol, e.g. NIFTY50 -> NIFTY. Stocks pass through."""
    sym = base_symbol.upper()
    return DERIVATIVE_UNDERLYING.get(sym, sym)


def lot_size(base_symbol: str) -> int | None:
    """F&O lot size for a base symbol, or None if it is not in our table."""
    return LOT_SIZES.get(derivatives_underlying(base_symbol))


def strike_step(base_symbol: str, spot: float | None = None) -> float:
    """Strike interval to use when proposing option legs."""
    sym = derivatives_underlying(base_symbol)
    if sym in STRIKE_STEPS:
        return STRIKE_STEPS[sym]
    if spot is None:
        return 10.0
    # Stock options roughly follow a 2.5% - 5% ladder; snap to a clean step.
    for threshold, step in ((100, 2.5), (500, 5.0), (1000, 10.0), (5000, 20.0), (10_000, 50.0)):
        if spot < threshold:
            return step
    return 100.0


def is_fno_eligible(base_symbol: str) -> bool:
    return derivatives_underlying(base_symbol) in LOT_SIZES


def load_universe_overrides() -> dict[str, Any]:
    """Merge a user-maintained universe JSON file, if configured.

    Expected shape::

        {"lot_sizes": {"RELIANCE": 500}, "sector_map": {"XYZ": "Materials"}}
    """
    path = os.getenv("ITA_UNIVERSE_FILE")
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    LOT_SIZES.update({k.upper(): int(v) for k, v in payload.get("lot_sizes", {}).items()})
    SECTOR_MAP.update({k.upper(): str(v) for k, v in payload.get("sector_map", {}).items()})
    STRIKE_STEPS.update({k.upper(): float(v) for k, v in payload.get("strike_steps", {}).items()})
    return payload


load_universe_overrides()
