"""Rank a sector or index and pick what the book can actually hold.

``analyse_many`` already walks a list and accumulates positions, but it does so
in the order it was given: a mediocre name analysed first consumes capital and
heat that a better one later in the alphabet then cannot have. For a screener
that is the wrong answer, and it is wrong silently.

So this runs in two passes:

1. **Merit.** Every constituent is analysed against a flat book, so each is
   judged on its own and nothing is rejected merely for arriving late. The
   results are ranked.
2. **Fit.** The best names are re-run in rank order with the book
   accumulating, so sector caps and the heat budget bind against the
   candidates worth keeping.

The second pass is a real re-run, not arithmetic over the first: sizing against
a loaded book is the risk manager's job, and a screener that reports a size the
desk would not actually approve is worse than one that reports nothing. It is
bounded to a shortlist, because names past the book's capacity cannot be
selected whatever their rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tradingagents_in.config.universe import (
    INDEX_CONSTITUENTS,
    SECTORS,
    sector_for,
)
from tradingagents_in.models.enums import Action
from tradingagents_in.utils.logging import get_logger

log = get_logger(__name__)

RANK_KEYS = ("reward_risk", "conviction", "confidence", "size")

# Every name the universe tables know about, used to resolve a sector.
ALL_CONSTITUENTS: tuple[str, ...] = tuple(
    sorted({n for names in INDEX_CONSTITUENTS.values() for n in names})
)


# The desk writes rejection reasons as sentences. The screen needs a label.
_REASON_LABELS = {
    "approved size is zero": "no capacity left",
    "insufficient_cash": "cash exhausted",
    "portfolio_heat": "heat budget full",
    "sector_concentration": "sector cap",
    "reward_risk_below_minimum": "R:R too thin",
}


def _short_reason(raw: Any) -> str:
    text = str(raw).strip()
    low = text.lower()
    for needle, label in _REASON_LABELS.items():
        if low.startswith(needle):
            return label
    return text.split(":")[0][:22]


@dataclass
class Candidate:
    """One name's standing in the screen."""

    symbol: str
    sector: str = "Others"
    action: str = "HOLD"
    approved_standalone: bool = False
    confidence: float = 0.0
    conviction: float | None = None
    reward_risk: float | None = None
    size_pct: float = 0.0
    risk_inr: float = 0.0
    product_type: str = ""
    time_horizon: str = ""

    # Filled by the portfolio pass.
    selected: bool = False
    final_size_pct: float = 0.0
    final_risk_inr: float = 0.0
    blocked_by: str = ""
    error: str = ""

    def rank_value(self, key: str) -> float:
        """Sort key. Missing values sort last rather than crashing the screen."""
        if key == "reward_risk":
            return self.reward_risk if self.reward_risk is not None else -1.0
        if key == "conviction":
            return self.conviction if self.conviction is not None else -1.0
        if key == "size":
            return self.size_pct
        return self.confidence


@dataclass
class ScreenResult:
    universe: str
    as_of: str | None = None
    rank_by: str = "reward_risk"
    scanned: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def shortlist(self) -> list[Candidate]:
        """Names the desk would take on their own, best first."""
        return [c for c in self.candidates if c.approved_standalone]

    @property
    def selected(self) -> list[Candidate]:
        """Names that also fit the book once the caps bind."""
        return [c for c in self.candidates if c.selected]

    def summary(self) -> dict[str, Any]:
        sel = self.selected
        return {
            "universe": self.universe,
            "as_of": self.as_of,
            "rank_by": self.rank_by,
            "scanned": self.scanned,
            "passed_standalone": len(self.shortlist),
            "selected": len(sel),
            "delivery_pct": round(
                sum(c.final_size_pct for c in sel if c.product_type == "CNC"), 2
            ),
            "intraday_pct": round(
                sum(c.final_size_pct for c in sel if c.product_type != "CNC"), 2
            ),
            "book_risk_inr": round(sum(c.final_risk_inr for c in sel), 2),
            "sectors": sorted({c.sector for c in sel}),
            "errors": self.errors[:5],
        }


# ------------------------------------------------------------------ universe


def resolve_universe(name: str) -> list[str]:
    """Turn an index name, a sector name or a symbol list into NSE symbols.

    Raises ValueError rather than silently screening nothing, because an empty
    screen and a misspelled sector look identical in the output.
    """
    raw = (name or "").strip()
    if not raw:
        raise ValueError("No universe given.")

    key = raw.upper().replace(" ", "").replace("_", "")
    if key in INDEX_CONSTITUENTS:
        return [f"{s}.NS" for s in INDEX_CONSTITUENTS[key]]

    for sector in SECTORS:
        if sector.lower() == raw.lower():
            members = [s for s in ALL_CONSTITUENTS if sector_for(s) == sector]
            if not members:
                raise ValueError(f"No known constituents in sector {sector!r}.")
            return [f"{s}.NS" for s in members]

    if "," in raw:
        return [s.strip() for s in raw.split(",") if s.strip()]

    # A bare symbol is a legitimate one-name screen, but only when it actually
    # looks like one. Accepting any alphanumeric word here would turn a
    # misspelled sector ("Enrgy") into a one-name screen of a ticker that does
    # not exist - the silent empty result this function exists to prevent.
    bare = raw.upper().split(".")[0]
    if raw.upper().endswith((".NS", ".BO")) or bare in ALL_CONSTITUENTS:
        return [raw]

    raise ValueError(
        f"Unknown universe {raw!r}. Expected an index "
        f"({', '.join(sorted(INDEX_CONSTITUENTS))}), a sector "
        f"({', '.join(SECTORS)}), or a comma-separated symbol list."
    )


# -------------------------------------------------------------------- screen


def _candidate(symbol: str, decision: Any) -> Candidate:
    proposal = getattr(decision, "proposal", None)
    plan = decision.investment_plan or {}
    return Candidate(
        symbol=decision.symbol or symbol,
        sector=decision.sector or sector_for(symbol.split(".")[0]),
        action=str(decision.final_action),
        approved_standalone=bool(decision.approved)
        and decision.final_action != Action.HOLD,
        confidence=float(decision.confidence or 0.0),
        conviction=plan.get("conviction"),
        reward_risk=getattr(proposal, "reward_risk", None) if proposal else None,
        size_pct=float(decision.position_size_pct or 0.0),
        risk_inr=float(decision.capital_at_risk_inr or 0.0),
        product_type=str(getattr(proposal, "product_type", "") or ""),
        time_horizon=str(getattr(proposal, "time_horizon", "") or ""),
    )


def screen(
    app: Any,
    universe: str,
    *,
    as_of: str | date | None = None,
    rank_by: str = "reward_risk",
    shortlist: int | None = None,
    apply_portfolio: bool = True,
) -> ScreenResult:
    """Rank a universe, then fit the best of it into one book.

    ``shortlist`` bounds the second pass; it defaults to the configured
    ``max_open_positions``, since a name past the book's capacity cannot be
    selected however well it ranks.
    """
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {RANK_KEYS}, got {rank_by!r}")

    symbols = resolve_universe(universe)
    result = ScreenResult(
        universe=universe, as_of=str(as_of) if as_of else None, rank_by=rank_by
    )

    # ---- pass 1: merit, each name against a flat book.
    for symbol in symbols:
        try:
            decision = app.analyse(symbol, as_of=as_of, save=False)
        except Exception as exc:  # noqa: BLE001 - one bad name must not stop the screen
            log.warning("screen: %s failed: %s", symbol, exc)
            result.errors.append(f"{symbol}: {exc}")
            result.candidates.append(Candidate(symbol=symbol, error=str(exc)))
            continue
        result.candidates.append(_candidate(symbol, decision))
    result.scanned = len(symbols)

    result.candidates.sort(
        key=lambda c: (c.approved_standalone, c.rank_value(rank_by), c.confidence),
        reverse=True,
    )

    if not apply_portfolio:
        return result

    # ---- pass 2: fit, best first, with the book accumulating.
    cap = shortlist or int(app.config.get("max_open_positions", 12))
    top = [c for c in result.candidates if c.approved_standalone][:cap]
    if not top:
        return result

    by_symbol = {c.symbol: c for c in result.candidates}
    try:
        fitted = app.analyse_many(
            [c.symbol for c in top], save=False, accumulate=True, as_of=as_of
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"portfolio pass: {exc}")
        return result

    for symbol, decision in fitted.items():
        candidate = by_symbol.get(symbol) or by_symbol.get(decision.symbol)
        if candidate is None:
            continue
        if decision.approved and decision.final_action != Action.HOLD:
            candidate.selected = True
            candidate.final_size_pct = float(decision.position_size_pct or 0.0)
            candidate.final_risk_inr = float(decision.capital_at_risk_inr or 0.0)
        else:
            reasons = (decision.portfolio_decision or {}).get("rejection_reasons") or []
            candidate.blocked_by = (
                _short_reason(reasons[0]) if reasons else "book full"
            )
    return result


def render(result: ScreenResult, limit: int = 20) -> str:
    """A compact table, in the CLI's register."""
    s = result.summary()
    lines = [
        f"{s['universe']} · {s['scanned']} scanned · "
        f"{s['passed_standalone']} passed on merit · {s['selected']} fit the book"
        + (f" · as of {s['as_of']}" if s["as_of"] else ""),
        f"delivery {s['delivery_pct']:.2f}% of capital (needs cash) · "
        f"intraday {s['intraday_pct']:.2f}% (leveraged)",
        f"book risk Rs {s['book_risk_inr']:,.2f} · "
        f"sectors {', '.join(s['sectors']) or '—'}",
        "",
        f"{'':2} {'symbol':<12}{'sector':<15}{'act':<5}{'R:R':>6}{'conv':>6}{'size%':>7} {'horizon':<11}note",
    ]
    rank = 0
    for c in result.candidates[:limit]:
        if c.error:
            lines.append(f"{'':2} {c.symbol[:11]:<12}{'':<24}{'ERR':<5}{'':>6}{'':>6}{'':>7} {'':<11}{c.error[:40]}")
            continue
        rank += 1
        mark = "✓" if c.selected else " "
        note = ""
        if c.selected:
            note = f"take {c.final_size_pct:.2f}%"
        elif c.approved_standalone:
            note = (c.blocked_by or "not shortlisted")[:34]
        lines.append(
            f"{mark:<2} {c.symbol[:11]:<12}{c.sector[:14]:<15}{c.action:<5}"
            f"{(c.reward_risk if c.reward_risk is not None else 0):>6.2f}"
            f"{(c.conviction if c.conviction is not None else 0):>6.2f}"
            f"{c.size_pct:>7.2f} {c.time_horizon[:10]:<11}{note}"
        )
    if len(result.candidates) > limit:
        lines.append(f"   ... and {len(result.candidates) - limit} more")
    for err in s["errors"]:
        lines.append(f"error: {err}")
    return "\n".join(lines)
