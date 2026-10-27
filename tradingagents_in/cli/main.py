"""``ita`` / ``tradingagents-in`` - the command line interface.

    ita analyse RELIANCE.NS --capital 500000 --depth 3
    ita batch RELIANCE.NS TCS.NS HDFCBANK.NS --provider offline
    ita session
    ita memory --stats
    ita outcome RELIANCE_NS-a1b2c3d4 --result win --pnl 18500

Rendering degrades gracefully: `rich` is a hard dependency and gives the panels
and tables, but every command also prints something usable when piped to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from tradingagents_in import __version__
from tradingagents_in.config.default_config import DEFAULT_CONFIG, build_config
from tradingagents_in.utils.formatting import inr_plain

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAVE_RICH = True
except ImportError:  # pragma: no cover - rich is a core dependency
    console = None
    HAVE_RICH = False


def _print(text: str = "") -> None:
    if HAVE_RICH:
        console.print(text)
    else:
        print(text)


def _rule(title: str) -> None:
    if HAVE_RICH:
        console.rule(f"[bold]{title}")
    else:
        print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ita",
        description="IndianTradingAgents - multi-agent research for NSE & BSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ita analyse RELIANCE.NS\n"
            "  ita analyse NIFTY --horizon intraday --provider offline\n"
            "  ita batch TCS.NS INFY.NS WIPRO.NS --capital 250000\n"
            "  ita session\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"IndianTradingAgents {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", help="LLM provider (openai, anthropic, ..., offline)")
        p.add_argument("--deep-model", help="Model for researchers, trader, risk, PM")
        p.add_argument("--quick-model", help="Model for the analyst team")
        p.add_argument("--capital", type=float, help="Trading capital in INR")
        p.add_argument("--profile", choices=["retail", "institutional"])
        p.add_argument("--risk-per-trade", type=float, help="Risk per trade, %% of capital")
        p.add_argument("--max-position", type=float, help="Max single-name exposure, %%")
        p.add_argument("--depth", type=int, help="Debate rounds (research depth)")
        p.add_argument(
            "--analysts",
            help="Comma-separated: fundamentals,sentiment,news_macro,technical",
        )
        p.add_argument(
            "--as-of",
            help="Decide a past session (YYYY-MM-DD) instead of today, using only "
                 "data available up to that close",
        )
        p.add_argument("--offline", action="store_true", help="No network calls, no LLM")
        p.add_argument("--no-fno", action="store_true", help="Cash segment only")
        p.add_argument("--no-save", action="store_true", help="Do not write results or memory")
        p.add_argument("--sequential", action="store_true", help="Skip LangGraph, run in-process")
        p.add_argument("--json", action="store_true", help="Emit the decision as JSON")
        p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")

    p_analyse = sub.add_parser("analyse", aliases=["analyze", "run"], help="Analyse one symbol")
    p_analyse.add_argument("symbol", help="RELIANCE.NS, TCS.BO, NIFTY, BANKNIFTY, ...")
    p_analyse.add_argument("--report", action="store_true", help="Print the full report")
    add_common(p_analyse)

    p_batch = sub.add_parser("batch", help="Analyse several symbols in sequence")
    p_batch.add_argument("symbols", nargs="+")
    p_batch.add_argument(
        "--independent",
        action="store_true",
        help="Judge each symbol against a flat book instead of accumulating "
             "approved positions across the batch",
    )
    add_common(p_batch)

    p_back = sub.add_parser(
        "backtest", help="Walk the desk across past sessions and score its decisions"
    )
    p_back.add_argument("symbols", nargs="+")
    p_back.add_argument("--start", required=True, help="First session (YYYY-MM-DD)")
    p_back.add_argument("--end", required=True, help="Last session (YYYY-MM-DD)")
    p_back.add_argument(
        "--max-hold-days", type=int, default=10,
        help="Sessions a position is carried before it is closed at the market (default 10)",
    )
    p_back.add_argument(
        "--step-days", type=int, default=1,
        help="Decide every Nth calendar day instead of every session (default 1)",
    )
    p_back.add_argument(
        "--fill-anywhere", action="store_true",
        help="Fill at the next close even when price is outside the entry band. "
             "Off by default: the desk's approval assumes the band, so filling "
             "outside it scores a trade it did not approve",
    )
    add_common(p_back)

    sub.add_parser("session", help="Market session status (IST) and expiry calendar")
    sub.add_parser("providers", help="LLM providers with credentials present")
    sub.add_parser("graph", help="Print the agent topology as a Mermaid diagram")

    p_config = sub.add_parser("config", help="Show the resolved configuration")
    add_common(p_config)

    p_memory = sub.add_parser("memory", help="Inspect the decision log")
    p_memory.add_argument("--stats", action="store_true")
    p_memory.add_argument("--symbol", help="Recall entries relevant to a symbol")
    p_memory.add_argument("--limit", type=int, default=10)

    p_outcome = sub.add_parser("outcome", help="Record how a past decision turned out")
    p_outcome.add_argument("run_id")
    p_outcome.add_argument(
        "--result", required=True, choices=["win", "loss", "scratch", "not_taken", "open"]
    )
    p_outcome.add_argument("--pnl", type=float, help="Realised P&L in INR")
    p_outcome.add_argument("--notes")
    p_outcome.add_argument("--lesson", help="What to do differently next time")

    p_universe = sub.add_parser("universe", help="Indices, sectors and F&O lot sizes")
    p_universe.add_argument("--index", help="List the constituents of an index")

    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI flags onto config keys. Only what was passed is overridden."""
    out: dict[str, Any] = {}
    if getattr(args, "offline", False):
        out["llm_provider"] = "offline"
        out["online_tools"] = False
    if getattr(args, "provider", None):
        out["llm_provider"] = args.provider
    if getattr(args, "deep_model", None):
        out["deep_think_model"] = args.deep_model
    if getattr(args, "quick_model", None):
        out["quick_think_model"] = args.quick_model
    if getattr(args, "capital", None):
        out["capital_inr"] = args.capital
    if getattr(args, "profile", None):
        out["profile"] = args.profile
    if getattr(args, "risk_per_trade", None):
        out["risk_per_trade_pct"] = args.risk_per_trade
    if getattr(args, "max_position", None):
        out["max_position_pct"] = args.max_position
    if getattr(args, "depth", None):
        out["max_debate_rounds"] = args.depth
    if getattr(args, "analysts", None):
        out["selected_analysts"] = [a.strip() for a in args.analysts.split(",") if a.strip()]
    if getattr(args, "no_fno", False):
        out["allow_fno"] = False
    if getattr(args, "as_of", None):
        out["as_of"] = args.as_of
    if getattr(args, "log_level", None):
        out["log_level"] = args.log_level
    return out


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_analyse(args: argparse.Namespace) -> int:
    from tradingagents_in.graph.runner import IndianTradingAgents

    config = build_config(overrides_from_args(args))
    app = IndianTradingAgents(config=config, use_langgraph=not args.sequential)

    _banner(config, app.backend, quiet=args.json)
    decision = app.analyse(args.symbol, save=not args.no_save)

    if args.json:
        print(decision.model_dump_json(indent=2, exclude={"proposal"}))
        return 0 if decision.approved else 1

    render_decision(decision, app)
    if args.report:
        _rule("Full report")
        _print(app.report(decision))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from tradingagents_in.graph.runner import IndianTradingAgents

    config = build_config(overrides_from_args(args))
    app = IndianTradingAgents(config=config, use_langgraph=not args.sequential)
    _banner(config, app.backend, quiet=args.json)

    results = app.analyse_many(
        args.symbols, save=not args.no_save, accumulate=not args.independent
    )

    if args.json:
        print(json.dumps(
            {k: json.loads(v.model_dump_json(exclude={"proposal"})) for k, v in results.items()},
            indent=2,
        ))
        return 0

    _rule("Batch summary")
    if HAVE_RICH:
        table = Table(show_header=True, header_style="bold")
        for column in ("Symbol", "Action", "Status", "Size %", "At risk", "Conf", "Note"):
            table.add_column(column)
        for symbol, decision in results.items():
            table.add_row(
                symbol,
                _action_markup(decision.final_action.value),
                "[green]APPROVED[/]" if decision.approved else "[dim]not taken[/]",
                f"{decision.position_size_pct:.2f}",
                inr_plain(decision.capital_at_risk_inr, 0),
                f"{decision.confidence:.0%}",
                (decision.portfolio_decision.get("summary", "") or "")[:60],
            )
        console.print(table)
    else:
        for symbol, decision in results.items():
            print(f"{symbol}: {decision.headline()}")

    approved = [d for d in results.values() if d.approved]
    total_risk = sum(d.capital_at_risk_inr for d in approved)
    _print(
        f"\n{len(approved)}/{len(results)} approved · total capital at risk "
        f"{inr_plain(total_risk)} "
        f"({total_risk / float(config['capital_inr']) * 100:.2f}% of the book)"
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from tradingagents_in.backtest import render_summary, walk_forward
    from tradingagents_in.graph.runner import IndianTradingAgents

    config = build_config(overrides_from_args(args))
    app = IndianTradingAgents(config=config, use_langgraph=not args.sequential)
    _banner(config, app.backend, quiet=args.json)

    result = walk_forward(
        app, args.symbols, args.start, args.end,
        max_hold_days=args.max_hold_days,
        step_days=args.step_days,
        require_entry_band=not args.fill_anywhere,
    )

    if args.json:
        print(json.dumps(result.summary(), indent=2, default=str))
        return 0

    _rule("backtest")
    _print(render_summary(result))
    if result.taken:
        _rule("trades")
        for t in result.taken[:25]:
            _print(
                f"{t.decided_on} {t.symbol:<12} {t.action:<4} "
                f"entry {t.entry_price:>10,.2f} exit {t.exit_price:>10,.2f} "
                f"{t.exit_reason:<8} R {t.r_multiple if t.r_multiple is not None else 0:>6.2f} "
                f"Rs {t.pnl_inr:>12,.2f}"
            )
        if len(result.taken) > 25:
            _print(f"... and {len(result.taken) - 25} more")
    _print()
    _print("Past decisions scored on past prices. Not a forecast, and not advice.")
    return 0


def cmd_session(_args: argparse.Namespace) -> int:
    from tradingagents_in.dataflows.calendar_in import session_context

    ctx = session_context()
    _rule("NSE session (Asia/Kolkata)")
    _print(ctx.to_markdown())
    status = {
        "REGULAR": "[green]OPEN — continuous session, 09:15-15:30[/]",
        "PRE_OPEN": "[yellow]PRE-OPEN call auction, 09:00-09:15[/]",
        "CLOSING": "[yellow]Post-close session, 15:40-16:00[/]",
        "HOLIDAY": "[red]TRADING HOLIDAY[/]",
        "CLOSED": "[red]CLOSED[/]",
    }.get(ctx.phase.value, ctx.phase.value)
    _print(f"\nStatus: {status}")
    return 0


def cmd_providers(_args: argparse.Namespace) -> int:
    from tradingagents_in.llm.provider import PROVIDERS, available_providers

    ready = set(available_providers())
    _rule("LLM providers")
    if HAVE_RICH:
        table = Table(show_header=True, header_style="bold")
        for column in ("Provider", "Credential", "Ready", "Default deep", "Default quick"):
            table.add_column(column)
        for name, spec in PROVIDERS.items():
            table.add_row(
                name,
                spec.env_key or "(none needed)",
                "[green]yes[/]" if name in ready else "[dim]no[/]",
                spec.default_deep or "-",
                spec.default_quick or "-",
            )
        console.print(table)
    else:
        for name, spec in PROVIDERS.items():
            print(f"{name}: {'ready' if name in ready else 'no key'} ({spec.env_key})")
    _print(
        "\nSet ITA_LLM_PROVIDER and the matching key in .env, or use "
        "`--offline` to run the deterministic engine with no LLM at all."
    )
    return 0


def cmd_graph(_args: argparse.Namespace) -> int:
    from tradingagents_in.graph.setup import mermaid

    print(mermaid())
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = build_config(overrides_from_args(args))
    _rule("Resolved configuration")
    if HAVE_RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Key")
        table.add_column("Value")
        table.add_column("Default", style="dim")
        for key in sorted(config):
            value, default = config[key], DEFAULT_CONFIG.get(key)
            table.add_row(
                key,
                f"[yellow]{value}[/]" if value != default else str(value),
                "" if value == default else str(default),
            )
        console.print(table)
    else:
        print(json.dumps(config, indent=2, default=str))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    from tradingagents_in.memory import TradingMemory

    memory = TradingMemory(build_config({}))
    if args.stats or not args.symbol:
        stats = memory.stats()
        _rule("Decision log")
        if not stats["total_decisions"]:
            _print(f"No decisions recorded yet ({memory.md_path} does not exist).")
            return 0
        for key, value in stats.items():
            _print(f"- **{key.replace('_', ' ')}**: {value}")

    if args.symbol:
        _rule(f"Recall for {args.symbol}")
        for record in memory.recall(args.symbol, limit=args.limit):
            _print(f"- {record.one_line()}")
    return 0


def cmd_outcome(args: argparse.Namespace) -> int:
    from tradingagents_in.memory import TradingMemory

    memory = TradingMemory(build_config({}))
    ok = memory.record_outcome(
        args.run_id, args.result, args.pnl, args.notes, args.lesson
    )
    if ok:
        _print(f"Recorded [green]{args.result}[/] for run `{args.run_id}`.")
        return 0
    _print(f"[red]No decision with run id `{args.run_id}` in {memory.jsonl_path}.[/]")
    return 1


def cmd_universe(args: argparse.Namespace) -> int:
    from tradingagents_in.config import universe as uni

    if args.index:
        key = args.index.upper().replace(" ", "").replace("_", "").replace("-", "")
        members = uni.INDEX_CONSTITUENTS.get(key)
        if not members:
            _print(
                f"[red]Unknown index {args.index!r}.[/] Known: "
                + ", ".join(sorted(uni.INDEX_CONSTITUENTS))
            )
            return 1
        _rule(f"{key} ({len(members)} constituents)")
        for symbol in members:
            _print(
                f"- {symbol:<14} {uni.sector_for(symbol):<24} "
                f"lot {uni.lot_size(symbol) or '-'}"
            )
        return 0

    _rule("Indices")
    for key, meta in uni.INDICES.items():
        _print(f"- {key:<18} {meta['yf']:<10} {meta['label']} ({meta['exchange']})")
    _rule("Sectors")
    _print(", ".join(uni.SECTORS))
    _print(f"\nConstituent lists loaded: {', '.join(sorted(uni.INDEX_CONSTITUENTS))}")
    _print(f"Universe reference date: {uni.AS_OF}")
    return 0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _banner(config: dict[str, Any], backend: str, quiet: bool = False) -> None:
    """Run header. Suppressed under --json so stdout stays machine-readable."""
    if quiet:
        return

    from tradingagents_in.dataflows.calendar_in import session_context
    from tradingagents_in.llm.provider import describe
    from tradingagents_in.utils.timeutils import as_of_moment

    # On a dated run the header must show the session being decided. Printing
    # today's date over a decision made for last March is how a backtested
    # result gets mistaken for a live one.
    ctx = session_context(when=as_of_moment(config))
    body = (
        f"Capital {inr_plain(config['capital_inr'])} · {config['profile']} profile · "
        f"risk {config['risk_per_trade_pct']}%/trade · max position "
        f"{config['max_position_pct']}%\n"
        f"LLM: {describe(config)}\n"
        f"Backend: {backend} · debate rounds {config['max_debate_rounds']} · "
        f"analysts {', '.join(config['selected_analysts'])}\n"
        f"Market: {ctx.phase.value} · {ctx.now:%Y-%m-%d %H:%M} IST · "
        f"weekly expiry {ctx.weekly_expiry}"
        + (f"\nAS OF {config['as_of']} - historical run, not a live decision"
           if config.get("as_of") else "")
    )
    if HAVE_RICH:
        console.print(
            Panel(body, title="IndianTradingAgents", subtitle="NSE · BSE · INR", expand=False)
        )
    else:
        print(body)


def _action_markup(action: str) -> str:
    colour = {"BUY": "green", "SELL": "red", "SHORT": "red", "HOLD": "yellow"}.get(action, "white")
    return f"[{colour}]{action}[/]" if HAVE_RICH else action


def render_decision(decision: Any, app: Any) -> None:
    """The one-screen verdict a person reads after a run."""
    proposal = decision.proposal
    status = "[green]APPROVED[/]" if decision.approved else "[yellow]NOT TAKEN[/]"

    _rule(f"{decision.symbol} — decision")
    _print(
        f"{_action_markup(decision.final_action.value)}  {status}  "
        f"confidence {decision.confidence:.0%}"
    )

    if decision.approved and proposal is not None:
        if HAVE_RICH:
            table = Table(show_header=False, box=None)
            table.add_column(style="bold")
            table.add_column()
            table.add_row("Instrument", f"{proposal.instrument.value} / "
                                        f"{proposal.product_type.value} / "
                                        f"{proposal.time_horizon.value}")
            table.add_row("Entry", str(proposal.entry_zone))
            table.add_row("Stop-loss", f"{proposal.stop_loss:,.2f}")
            if proposal.targets:
                table.add_row(
                    "Targets",
                    ", ".join(f"{t.price:,.2f} ({t.book_pct:.0f}%)" for t in proposal.targets),
                )
            table.add_row("Size", f"{decision.position_size_pct:.2f}% of capital")
            table.add_row("At risk", inr_plain(decision.capital_at_risk_inr))
            if proposal.reward_risk:
                table.add_row("Reward:risk", f"{proposal.reward_risk:.2f} (after charges)")
            if proposal.option_legs:
                table.add_row("Options", proposal.option_strategy or "custom")
                for leg in proposal.option_legs:
                    table.add_row("", str(leg))
            console.print(table)
        else:
            print(proposal.to_markdown())

    pm = decision.portfolio_decision or {}
    if pm.get("portfolio_reasoning"):
        _rule("Portfolio manager")
        _print(pm["portfolio_reasoning"])
    for reason in pm.get("rejection_reasons", [])[:5]:
        _print(f"  [red]x[/] {reason}" if HAVE_RICH else f"  x {reason}")
    for condition in pm.get("conditions", [])[:5]:
        _print(f"  [cyan]•[/] {condition}" if HAVE_RICH else f"  * {condition}")

    if decision.risk_notes:
        _rule("Risk notes")
        for note in decision.risk_notes[:6]:
            _print(f"- {note}")

    plan = decision.investment_plan or {}
    if plan:
        _rule("Research")
        _print(
            f"{plan.get('stance')} · conviction {float(plan.get('conviction', 0)):.0%} · "
            f"debate won by {plan.get('winning_side')}\n{plan.get('verdict', '')}"
        )

    if decision.errors:
        _rule("Errors")
        for err in decision.errors[:5]:
            _print(f"[red]- {err}[/]" if HAVE_RICH else f"- {err}")

    _print(
        f"\n[dim]run {decision.run_id} · {decision.elapsed_seconds or 0:.1f}s · "
        f"results/{decision.symbol}/[/]"
        if HAVE_RICH else f"\nrun {decision.run_id}"
    )
    _print(
        "[dim]Research output, not investment advice. Verify against the exchange and "
        "consult a SEBI-registered adviser.[/]" if HAVE_RICH else
        "Research output, not investment advice."
    )


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "analyse": cmd_analyse, "analyze": cmd_analyse, "run": cmd_analyse,
        "backtest": cmd_backtest,
        "batch": cmd_batch,
        "session": cmd_session,
        "providers": cmd_providers,
        "graph": cmd_graph,
        "config": cmd_config,
        "memory": cmd_memory,
        "outcome": cmd_outcome,
        "universe": cmd_universe,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        _print("\n[yellow]Interrupted.[/]" if HAVE_RICH else "\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not traceback
        _print(f"[red]Error: {type(exc).__name__}: {exc}[/]" if HAVE_RICH
               else f"Error: {type(exc).__name__}: {exc}")
        if str(args.__dict__.get("log_level", "")).upper() == "DEBUG":
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
