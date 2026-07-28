"""Fundamental data for Indian listed companies.

Sources, in order of preference:
  1. yfinance statements - free, covers both .NS and .BO, reasonably reliable
     for large caps; note it reports *consolidated* figures for most Indian names.
  2. Screener.in - opt-in HTML parse (``enable_screener_scrape``). Screener's
     ratio page is the de-facto reference for Indian retail research, but the
     scrape is off by default: check their terms before enabling it.

Everything is Ind-AS flavoured: we care about consolidated vs standalone,
promoter holding and pledge, and cash conversion (OCF/PAT), because those are
the numbers that actually separate good and bad Indian balance sheets.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from tradingagents_in.dataflows.cache import get_cache
from tradingagents_in.dataflows.http import get_text
from tradingagents_in.models.analyst import (
    PeerMetric,
    QuarterlyResult,
    ShareholdingPattern,
)
from tradingagents_in.models.base import SymbolRef
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

SCREENER_URL = "https://www.screener.in/company/{symbol}/consolidated/"

# yfinance statement row labels -> our field names.
_INCOME_ROWS = {
    "Total Revenue": "revenue",
    "Operating Revenue": "revenue",
    "EBITDA": "ebitda",
    "Net Income": "pat",
    "Net Income Common Stockholders": "pat",
    "Basic EPS": "eps",
    "Diluted EPS": "eps",
}


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return None if (np.isnan(out) or np.isinf(out)) else out
    except (TypeError, ValueError):
        return None


def _period_label(period: Any) -> str:
    """Column headers arrive as Timestamps or plain strings depending on feed."""
    to_date = getattr(period, "date", None)
    return str(to_date()) if callable(to_date) else str(period)


def _pick(frame: pd.DataFrame | None, labels: list[str], column: int = 0) -> float | None:
    """First matching row label in a yfinance statement frame."""
    if frame is None or frame.empty or column >= frame.shape[1]:
        return None
    for label in labels:
        if label in frame.index:
            return _f(frame.loc[label].iloc[column])
    return None


# ---------------------------------------------------------------- yfinance
def fetch_yf_fundamentals(
    ref: SymbolRef, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Ratios, statements and shareholding from yfinance."""
    cache = get_cache(config)
    hit = cache.get("fundamentals_yf", {"symbol": ref.yf_symbol})
    if hit is not None:
        return hit

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return {"error": "yfinance not installed", "sources": []}

    out: dict[str, Any] = {"sources": ["yfinance"], "symbol": ref.yf_symbol}
    try:
        ticker = yf.Ticker(ref.yf_symbol)
        info: dict[str, Any] = {}
        try:
            info = ticker.get_info() or {}
        except Exception as exc:
            log.debug("get_info failed for %s: %s", ref.yf_symbol, exc)

        out.update(
            {
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap_inr": _f(info.get("marketCap")),
                "pe": _f(info.get("trailingPE")),
                "forward_pe": _f(info.get("forwardPE")),
                "pb": _f(info.get("priceToBook")),
                "ev_to_ebitda": _f(info.get("enterpriseToEbitda")),
                "dividend_yield_pct": _pct(info.get("dividendYield")),
                "roe_pct": _pct(info.get("returnOnEquity")),
                "roa_pct": _pct(info.get("returnOnAssets")),
                "profit_margin_pct": _pct(info.get("profitMargins")),
                "operating_margin_pct": _pct(info.get("operatingMargins")),
                "debt_to_equity": _ratio_de(info.get("debtToEquity")),
                "current_ratio": _f(info.get("currentRatio")),
                "revenue_growth_yoy_pct": _pct(info.get("revenueGrowth")),
                "earnings_growth_yoy_pct": _pct(info.get("earningsGrowth")),
                "book_value": _f(info.get("bookValue")),
                "beta": _f(info.get("beta")),
                "held_by_insiders_pct": _pct(info.get("heldPercentInsiders")),
                "held_by_institutions_pct": _pct(info.get("heldPercentInstitutions")),
                "shares_outstanding": _f(info.get("sharesOutstanding")),
                "total_debt_inr": _f(info.get("totalDebt")),
                "total_cash_inr": _f(info.get("totalCash")),
                "free_cash_flow_inr": _f(info.get("freeCashflow")),
                "operating_cash_flow_inr": _f(info.get("operatingCashflow")),
                "ebitda_inr": _f(info.get("ebitda")),
                "currency": info.get("financialCurrency", "INR"),
            }
        )

        out["quarters"] = _quarterly_from_yf(ticker)
        out.update(_derived_from_statements(ticker))
    except Exception as exc:
        log.warning("yfinance fundamentals failed for %s: %s", ref.yf_symbol, exc)
        out["error"] = str(exc)

    # Cash conversion: sustained OCF/PAT below ~0.7 is the classic Indian
    # earnings-quality red flag (receivables build, promoter-linked entities).
    ocf, pat_proxy = out.get("operating_cash_flow_inr"), out.get("net_income_inr")
    if ocf and pat_proxy:
        out["ocf_to_pat"] = round(ocf / pat_proxy, 3) if pat_proxy else None

    cache.set("fundamentals_yf", {"symbol": ref.yf_symbol}, out)
    return out


def _quarterly_from_yf(ticker: Any) -> list[dict[str, Any]]:
    """Last few reported quarters, newest first."""
    try:
        income = ticker.quarterly_income_stmt
    except Exception:
        return []
    if income is None or not isinstance(income, pd.DataFrame) or income.empty:
        return []

    quarters: list[dict[str, Any]] = []
    for col_idx, period in enumerate(income.columns[:6]):
        revenue = _pick(income, ["Total Revenue", "Operating Revenue"], col_idx)
        ebitda = _pick(income, ["EBITDA", "Normalized EBITDA"], col_idx)
        pat = _pick(income, ["Net Income", "Net Income Common Stockholders"], col_idx)
        entry = {
            "period": _period_label(period),
            "revenue_inr": revenue,
            "ebitda_inr": ebitda,
            "pat_inr": pat,
            "eps": _pick(income, ["Basic EPS", "Diluted EPS"], col_idx),
            "ebitda_margin_pct": round(ebitda / revenue * 100, 2)
            if (ebitda and revenue) else None,
        }
        quarters.append(entry)

    # Year-on-year uses the same quarter of the prior year (index +4), which is
    # the comparison Indian companies actually report against.
    for i, q in enumerate(quarters):
        prior = quarters[i + 4] if i + 4 < len(quarters) else None
        if prior:
            q["revenue_yoy_pct"] = _growth(q["revenue_inr"], prior["revenue_inr"])
            q["pat_yoy_pct"] = _growth(q["pat_inr"], prior["pat_inr"])
    return quarters


def _derived_from_statements(ticker: Any) -> dict[str, Any]:
    """ROCE, interest coverage and multi-year CAGRs from annual statements."""
    out: dict[str, Any] = {}
    try:
        income = ticker.income_stmt
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow
    except Exception:
        return out

    ebit = _pick(income, ["EBIT", "Operating Income"], 0)
    interest = _pick(income, ["Interest Expense", "Interest Expense Non Operating"], 0)
    net_income = _pick(income, ["Net Income", "Net Income Common Stockholders"], 0)
    equity = _pick(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"], 0)
    total_debt = _pick(balance, ["Total Debt", "Long Term Debt"], 0)
    cash = _pick(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], 0)

    out["net_income_inr"] = net_income
    out["ebit_inr"] = ebit
    out["shareholders_equity_inr"] = equity

    # ROCE = EBIT / (equity + debt). The Indian screening convention.
    if ebit is not None and equity:
        capital_employed = equity + (total_debt or 0.0)
        if capital_employed:
            out["roce_pct"] = round(ebit / capital_employed * 100, 2)
    if ebit is not None and interest:
        out["interest_coverage"] = round(ebit / abs(interest), 2)
    if total_debt is not None and cash is not None:
        out["net_debt_inr"] = total_debt - cash

    if isinstance(income, pd.DataFrame) and income.shape[1] >= 4:
        out["sales_growth_3y_cagr_pct"] = _cagr(
            _pick(income, ["Total Revenue", "Operating Revenue"], 0),
            _pick(income, ["Total Revenue", "Operating Revenue"], 3),
            3,
        )
        out["profit_growth_3y_cagr_pct"] = _cagr(
            _pick(income, ["Net Income", "Net Income Common Stockholders"], 0),
            _pick(income, ["Net Income", "Net Income Common Stockholders"], 3),
            3,
        )

    ocf = _pick(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)
    capex = _pick(cashflow, ["Capital Expenditure"], 0)
    if ocf is not None:
        out["operating_cash_flow_inr"] = ocf
        if capex is not None:
            out["free_cash_flow_inr"] = ocf + capex  # capex is reported negative
    return out


# --------------------------------------------------------------- Screener.in
_SCREENER_RATIO = re.compile(
    r'<li[^>]*>\s*<span class="name">\s*([^<]+?)\s*</span>.*?<span class="value">(.*?)</span>',
    re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")


def fetch_screener(
    ref: SymbolRef, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Best-effort Screener.in ratio scrape. Disabled unless explicitly enabled.

    Screener publishes the ratios Indian analysts quote (ROCE, promoter
    holding, pledge). This parser is intentionally forgiving: any failure
    degrades to an empty dict and the analyst notes the missing source.
    """
    cfg = config or {}
    if not cfg.get("enable_screener_scrape"):
        return {}
    if ref.is_index:
        return {}

    cache = get_cache(config)
    key = {"symbol": ref.base_symbol}
    hit = cache.get("screener", key)
    if hit is not None:
        return hit

    url = SCREENER_URL.format(symbol=ref.base_symbol)
    html = get_text(url, timeout=15)
    if not html:
        return {}

    ratios: dict[str, float] = {}
    for name, value in _SCREENER_RATIO.findall(html):
        clean_name = _TAG.sub("", name).strip().lower()
        clean_value = _TAG.sub("", value).replace(",", "").replace("₹", "").replace("%", "")
        clean_value = clean_value.replace("Cr.", "").strip()
        try:
            ratios[clean_name] = float(clean_value.split()[0])
        except (ValueError, IndexError):
            continue

    out = {
        "sources": ["screener.in"],
        "url": url,
        "market_cap_inr_cr": ratios.get("market cap"),
        "pe": ratios.get("stock p/e"),
        "book_value": ratios.get("book value"),
        "dividend_yield_pct": ratios.get("dividend yield"),
        "roce_pct": ratios.get("roce"),
        "roe_pct": ratios.get("roe"),
        "face_value": ratios.get("face value"),
        "high_52w": ratios.get("high / low"),
        "raw_ratios": ratios,
    }
    cache.set("screener", key, out)
    return out


# ------------------------------------------------------------------- peers
def fetch_peers(
    ref: SymbolRef, limit: int = 5, config: dict[str, Any] | None = None
) -> list[PeerMetric]:
    """Sector peers from our universe map, with headline ratios each."""
    from tradingagents_in.config.universe import SECTOR_MAP
    from tradingagents_in.dataflows.symbols import resolve

    sector = ref.sector or SECTOR_MAP.get(ref.base_symbol)
    if not sector or sector == "Others":
        return []

    candidates = [
        sym for sym, sec in SECTOR_MAP.items() if sec == sector and sym != ref.base_symbol
    ][: limit * 2]

    peers: list[PeerMetric] = []
    for sym in candidates:
        if len(peers) >= limit:
            break
        try:
            peer_ref = resolve(sym, ref.exchange)
            data = fetch_yf_fundamentals(peer_ref, config)
            if data.get("error") and data.get("pe") is None:
                continue
            peers.append(
                PeerMetric(
                    symbol=sym,
                    name=data.get("name"),
                    pe=data.get("pe"),
                    pb=data.get("pb"),
                    roe_pct=data.get("roe_pct"),
                    debt_to_equity=data.get("debt_to_equity"),
                    market_cap_inr=data.get("market_cap_inr"),
                    revenue_growth_yoy_pct=data.get("revenue_growth_yoy_pct"),
                )
            )
        except Exception as exc:  # a bad peer must not sink the run
            log.debug("peer fetch failed for %s: %s", sym, exc)
    return peers


# ------------------------------------------------------------ shareholding
def build_shareholding(payload: dict[str, Any]) -> ShareholdingPattern:
    """Map whatever we have into the Indian shareholding view.

    yfinance's ``heldPercentInsiders`` is the closest free proxy for promoter
    holding. It is not identical to the SEBI-filed promoter category, so it is
    reported as a proxy and the analyst is told to treat it as indicative.
    """
    return ShareholdingPattern(
        promoter_pct=payload.get("promoter_pct") or payload.get("held_by_insiders_pct"),
        promoter_pledge_pct=payload.get("promoter_pledge_pct"),
        fii_pct=payload.get("fii_pct"),
        dii_pct=payload.get("dii_pct"),
        public_pct=payload.get("public_pct"),
        promoter_change_qoq_pct=payload.get("promoter_change_qoq_pct"),
        fii_change_qoq_pct=payload.get("fii_change_qoq_pct"),
    )


def build_quarters(payload: dict[str, Any]) -> list[QuarterlyResult]:
    out: list[QuarterlyResult] = []
    for q in payload.get("quarters", [])[:6]:
        try:
            out.append(QuarterlyResult(**q))
        except Exception:
            continue
    return out


# ----------------------------------------------------------------- helpers
def _pct(value: Any) -> float | None:
    """yfinance returns ratios as fractions; Indian reporting quotes percent."""
    out = _f(value)
    return None if out is None else round(out * 100, 2)


def _ratio_de(value: Any) -> float | None:
    """yfinance reports debtToEquity in percent (e.g. 45.2 means 0.45x)."""
    out = _f(value)
    return None if out is None else round(out / 100, 3)


def _growth(current: float | None, prior: float | None) -> float | None:
    if current is None or not prior:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def _cagr(latest: float | None, oldest: float | None, years: int) -> float | None:
    if latest is None or not oldest or oldest <= 0 or latest <= 0 or years <= 0:
        return None
    return round(((latest / oldest) ** (1 / years) - 1) * 100, 2)


def merge_sources(*payloads: dict[str, Any]) -> dict[str, Any]:
    """Merge fundamentals dicts left-to-right; later non-null values win."""
    merged: dict[str, Any] = {"sources": []}
    for payload in payloads:
        if not payload:
            continue
        merged["sources"].extend(payload.get("sources", []))
        for key, value in payload.items():
            if key == "sources":
                continue
            if value is not None and value != []:
                merged[key] = value
    merged["sources"] = sorted(set(merged["sources"]))
    return merged
