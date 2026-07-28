"""Persistent decision log and recall.

Two files are written side by side:

* ``trading_memory.md`` - the human record. Append-only, readable in a terminal
  or a PR diff, and the thing a person actually reviews on a Friday evening.
* ``trading_memory.jsonl`` - the machine record, one JSON object per decision,
  used for recall and for scoring past calls once outcomes are known.

Recall is deliberately simple and explainable: same symbol first, then same
sector, then the same market regime, most recent first. There is no embedding
model in the loop, because a desk that cannot say *why* a past decision was
surfaced cannot audit the advice it produced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import fmt_ist, now_ist

log = get_logger(__name__)

_HEADER = """\
# Trading memory - IndianTradingAgents

Append-only decision log. Every run of the graph adds one entry. Outcomes are
filled in later by `record_outcome`, which is what turns a log into a lesson.

All amounts are in INR, all timestamps Asia/Kolkata.

---
"""


@dataclass
class MemoryRecord:
    """One decision, plus its outcome once that is known."""

    run_id: str
    timestamp: str
    symbol: str
    exchange: str = "NSE"
    sector: str | None = None
    action: str = "HOLD"
    approved: bool = False
    confidence: float = 0.0
    position_size_pct: float = 0.0
    capital_at_risk_inr: float = 0.0
    instrument: str = "CASH"
    time_horizon: str = "SWING"
    entry_zone: str | None = None
    stop_loss: float | None = None
    first_target: float | None = None
    price_at_decision: float | None = None
    research_stance: str | None = None
    winning_side: str | None = None
    risk_verdict: str | None = None
    thesis: str = ""
    key_risks: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    market_regime: str | None = None
    # Filled in later
    outcome: str | None = None  # win | loss | scratch | not_taken | open
    realised_pnl_inr: float | None = None
    outcome_notes: str | None = None
    lesson: str | None = None

    def to_markdown(self) -> str:
        lines = [
            f"## {self.timestamp} — {self.symbol} — {self.action}"
            f"{' (approved)' if self.approved else ' (not taken)'}",
            "",
            f"- **Run**: `{self.run_id}`",
            f"- **Sector**: {self.sector or 'unclassified'} | "
            f"**Instrument**: {self.instrument} | **Horizon**: {self.time_horizon}",
            f"- **Price at decision**: {self.price_at_decision}",
            f"- **Size**: {self.position_size_pct:.2f}% of capital | "
            f"**At risk**: Rs {self.capital_at_risk_inr:,.0f} | "
            f"**Confidence**: {self.confidence:.0%}",
        ]
        if self.entry_zone:
            lines.append(
                f"- **Levels**: entry {self.entry_zone}, stop {self.stop_loss}, "
                f"target {self.first_target}"
            )
        lines.append(
            f"- **Research**: {self.research_stance} (debate won by {self.winning_side}) | "
            f"**Risk**: {self.risk_verdict}"
        )
        if self.market_regime:
            lines.append(f"- **Regime**: {self.market_regime}")
        lines += ["", f"**Thesis.** {self.thesis}", ""]
        if self.key_risks:
            lines.append("**Risks flagged.**")
            lines += [f"- {r}" for r in self.key_risks]
            lines.append("")
        if self.blockers:
            lines.append("**Blocked by.**")
            lines += [f"- {b}" for b in self.blockers]
            lines.append("")
        if self.outcome:
            lines += [
                f"**Outcome: {self.outcome.upper()}** "
                f"({f'Rs {self.realised_pnl_inr:,.0f}' if self.realised_pnl_inr is not None else 'P&L not recorded'})",
                "",
            ]
            if self.outcome_notes:
                lines += [self.outcome_notes, ""]
            if self.lesson:
                lines += [f"> **Lesson.** {self.lesson}", ""]
        lines.append("---")
        return "\n".join(lines)

    def one_line(self) -> str:
        status = self.outcome.upper() if self.outcome else ("TAKEN" if self.approved else "PASSED")
        return (
            f"[{self.timestamp[:10]}] {self.symbol} {self.action} "
            f"{self.position_size_pct:.1f}% — {status}"
            + (f" — {self.lesson}" if self.lesson else f" — {self.thesis[:160]}")
        )


class TradingMemory:
    """Append-only decision log with explainable recall."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enable_memory", True))
        self.md_path = Path(config.get("memory_file", "trading_memory.md"))
        self.jsonl_path = self.md_path.with_suffix(".jsonl")
        self.recall_count = int(config.get("memory_recall_count", 5))

    # ------------------------------------------------------------- writing
    def append(self, record: MemoryRecord) -> None:
        if not self.enabled:
            return
        try:
            self.md_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.md_path.exists():
                self.md_path.write_text(_HEADER, encoding="utf-8")
            with self.md_path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + record.to_markdown() + "\n")
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), default=str) + "\n")
        except OSError as exc:  # a full disk must not lose the decision itself
            log.warning("could not write trading memory: %s", exc)

    def record_outcome(
        self,
        run_id: str,
        outcome: str,
        realised_pnl_inr: float | None = None,
        notes: str | None = None,
        lesson: str | None = None,
    ) -> bool:
        """Close the loop on a past decision. Returns False if the run is unknown."""
        records = self.load()
        match = next((r for r in records if r.run_id == run_id), None)
        if match is None:
            return False

        match.outcome = outcome
        match.realised_pnl_inr = realised_pnl_inr
        match.outcome_notes = notes
        match.lesson = lesson

        try:
            with self.jsonl_path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(asdict(rec), default=str) + "\n")
            with self.md_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"\n### Outcome update — {run_id} ({match.symbol})\n\n"
                    f"- **{outcome.upper()}**"
                    + (f", P&L Rs {realised_pnl_inr:,.0f}" if realised_pnl_inr is not None else "")
                    + "\n"
                    + (f"- {notes}\n" if notes else "")
                    + (f"\n> **Lesson.** {lesson}\n" if lesson else "")
                    + "\n---\n"
                )
        except OSError as exc:
            log.warning("could not update trading memory: %s", exc)
            return False
        return True

    # ------------------------------------------------------------- reading
    def load(self) -> list[MemoryRecord]:
        if not self.jsonl_path.exists():
            return []
        records: list[MemoryRecord] = []
        known = set(MemoryRecord.__dataclass_fields__)
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated write must not break recall
            records.append(MemoryRecord(**{k: v for k, v in payload.items() if k in known}))
        return records

    def recall(
        self,
        symbol: str,
        sector: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """Past decisions worth showing this run, most relevant first."""
        limit = limit or self.recall_count
        records = self.load()
        if not records:
            return []

        def tier(rec: MemoryRecord) -> int:
            if rec.symbol.upper() == symbol.upper():
                return 0
            if sector and rec.sector == sector:
                return 1
            if rec.outcome in ("loss", "win"):
                return 2  # any closed trade carries a transferable lesson
            return 3

        # Closest match first; newest first within each tier.
        return sorted(records, key=lambda r: (tier(r), _neg_time(r.timestamp)))[:limit]

    def context(self, symbol: str, sector: str | None = None, limit: int | None = None) -> str:
        """Markdown block injected into researcher, trader and PM prompts."""
        recalled = self.recall(symbol, sector, limit)
        if not recalled:
            return ""

        lines = [
            "These are the desk's own previous decisions, closest match first. Use them to "
            "avoid repeating a mistake, not to anchor on a past view - the market has moved "
            "since.",
            "",
        ]
        for rec in recalled:
            lines.append(f"- {rec.one_line()}")
        closed = [r for r in self.load() if r.outcome in ("win", "loss")]
        if closed:
            wins = sum(1 for r in closed if r.outcome == "win")
            lines += [
                "",
                f"Closed-trade record so far: {wins}/{len(closed)} "
                f"({wins / len(closed):.0%} hit rate across {len(closed)} decisions with a "
                "recorded outcome).",
            ]
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        records = self.load()
        closed = [r for r in records if r.outcome in ("win", "loss", "scratch")]
        wins = [r for r in closed if r.outcome == "win"]
        pnl = [r.realised_pnl_inr for r in closed if r.realised_pnl_inr is not None]
        return {
            "total_decisions": len(records),
            "approved": sum(1 for r in records if r.approved),
            "closed": len(closed),
            "wins": len(wins),
            "hit_rate": round(len(wins) / len(closed), 3) if closed else None,
            "net_pnl_inr": round(sum(pnl), 2) if pnl else None,
            "symbols": sorted({r.symbol for r in records}),
        }


def _neg_time(stamp: str) -> str:
    """Sort key that puts newer timestamps first within a relevance tier."""
    # Inverting a lexicographic ISO timestamp: complement each digit.
    return "".join(chr(ord("9") - (ord(c) - ord("0"))) if c.isdigit() else c for c in stamp)


def build_record(decision: Any, config: dict[str, Any] | None = None) -> MemoryRecord:
    """Construct a memory record from a :class:`FinalDecision`."""
    proposal = decision.trade_proposal or {}
    plan = decision.investment_plan or {}
    risk = decision.risk_assessment or {}
    portfolio = decision.portfolio_decision or {}
    entry = proposal.get("entry_zone") or {}
    targets = proposal.get("targets") or []

    return MemoryRecord(
        run_id=decision.run_id,
        timestamp=fmt_ist(now_ist(), with_tz=False),
        symbol=decision.symbol,
        exchange=decision.exchange,
        sector=decision.sector,
        action=decision.final_action.value
        if hasattr(decision.final_action, "value") else str(decision.final_action),
        approved=decision.approved,
        confidence=decision.confidence,
        position_size_pct=decision.position_size_pct,
        capital_at_risk_inr=decision.capital_at_risk_inr,
        instrument=proposal.get("instrument", "CASH"),
        time_horizon=proposal.get("time_horizon", "SWING"),
        entry_zone=(
            f"{entry.get('low')}-{entry.get('high')}" if entry.get("low") is not None else None
        ),
        stop_loss=proposal.get("stop_loss"),
        first_target=targets[0].get("price") if targets else None,
        price_at_decision=(proposal.get("entry_zone") or {}).get("reference_price"),
        research_stance=plan.get("stance"),
        winning_side=plan.get("winning_side"),
        risk_verdict=risk.get("verdict"),
        thesis=plan.get("verdict") or portfolio.get("summary") or "",
        key_risks=list(decision.risk_notes)[:5],
        blockers=[
            f"{b.get('rule')}: {b.get('detail')}"
            for b in risk.get("breaches", [])
            if b.get("severity") == "BLOCKER"
        ][:5],
        market_regime=(decision.analyst_reports.get("news_macro") or {}).get("macro_regime"),
        outcome=None if decision.approved else "not_taken",
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
