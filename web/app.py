"""A thin HTTP front end over ``IndianTradingAgents``.

One page, one endpoint. The analysis itself is unchanged - this module only
builds a runner, hands it a symbol and renders what comes back.

Run locally::

    ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
        uvicorn web.app:app --reload

The defaults are deliberately the offline ones: a deployment with no API key
still answers, deterministically, instead of erroring on the first request.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.screen import RANK_KEYS
from tradingagents_in.utils.logging import get_logger
from tradingagents_in.utils.timeutils import parse_date
from web.runs import registry

log = get_logger(__name__)

# NSE/BSE tickers, optionally with the Yahoo suffix the dataflows expect.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9&._-]{1,24}$")

# Building the graph is not free, so runners are reused per distinct config.
# One lock around the whole dict: creation is rare and contention is not the
# bottleneck here, a full agent run is.
_runners: dict[tuple[float, str], IndianTradingAgents] = {}
_runners_lock = threading.Lock()

# An agent run holds a graph, a tape and the LLM client. Letting an arbitrary
# number run at once is how a 512 MB container gets OOM-killed.
_slots = threading.Semaphore(int(os.getenv("ITA_WEB_CONCURRENCY", "2")))

# The paper book is shared mutable state: one file, read-modify-written by
# every pass. Two passes at once would interleave fills and lose orders, so
# autopilot is serialised on its own lock rather than on the analysis slots.
_BOOK_PATH = Path(os.getenv("ITA_PAPER_BOOK", "results/paper_book.json"))
_book_lock = threading.Lock()

app = FastAPI(
    title="IndianTradingAgents",
    description="Multi-agent LLM trading research for NSE & BSE.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# A browser front end served from somewhere else - a dashboard on Vercel -
# is a cross-origin caller and needs to be named here. Empty by default: the
# bundled page is same-origin, so nothing is opened up unless it is asked for.
# A Vercel deployment that proxies /api through its own domain stays
# same-origin and needs nothing set.
_origins = [o.strip() for o in os.getenv("ITA_WEB_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        # Exact origins only. allow_origins=["*"] would let any page on the
        # internet drive this deployment's analysis budget.
        allow_origins=_origins,
        allow_origin_regex=os.getenv("ITA_WEB_CORS_ORIGIN_REGEX") or None,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )


def _runner(capital_inr: float, profile: str) -> IndianTradingAgents:
    key = (capital_inr, profile)
    with _runners_lock:
        runner = _runners.get(key)
        if runner is None:
            runner = IndianTradingAgents(
                {"capital_inr": capital_inr, "profile": profile}
            )
            _runners[key] = runner
        return runner


class AnalyseRequest(BaseModel):
    symbol: str = Field("RELIANCE.NS", max_length=24)
    capital_inr: float = Field(1_000_000.0, gt=0, le=1e11)
    profile: str = Field("retail")
    as_of: str | None = Field(
        None,
        max_length=10,
        description="Decide a past session (YYYY-MM-DD) using only data up to that close",
    )


class ScreenRequest(BaseModel):
    universe: str = Field("NIFTY50", max_length=400)
    rank_by: str = Field("reward_risk")
    capital_inr: float = Field(1_000_000.0, gt=0, le=1e11)
    profile: str = Field("retail")
    as_of: str | None = Field(None, max_length=10)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """Railway's health check. Must not build a runner or touch the network."""
    return {"status": "ok"}


@app.get("/api/config")
def config_summary() -> dict[str, Any]:
    """What this deployment is actually configured to do."""
    from tradingagents_in.llm.provider import missing_api_key

    cfg = build_config()
    # A provider named without its key answers every request anyway - each
    # agent falls back to its deterministic draft. Reporting that as a live
    # LLM made the console claim reasoning that never happened, so the
    # deployment counts as deterministic unless the key is actually there.
    needs_key = missing_api_key(cfg)
    configured = str(cfg.get("llm_provider", "")).lower()
    return {
        "llm_provider": cfg.get("llm_provider"),
        "data_provider": cfg.get("data_provider"),
        "online_tools": cfg.get("online_tools"),
        "profile": cfg.get("profile"),
        "capital_inr": cfg.get("capital_inr"),
        "max_debate_rounds": cfg.get("max_debate_rounds"),
        "offline": configured in ("offline", "none", "") or needs_key is not None,
        "missing_api_key": needs_key,
        "llm_note": (
            f"{cfg.get('llm_provider')} is configured but {needs_key} is not set, "
            "so every agent is answering from its deterministic draft"
        ) if needs_key else None,
    }


@app.post("/api/analyse")
def analyse(req: AnalyseRequest) -> JSONResponse:
    symbol = req.symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=422, detail=f"Not a usable symbol: {symbol!r}")
    if req.profile not in {"retail", "institutional"}:
        raise HTTPException(status_code=422, detail="profile must be retail or institutional")

    as_of = (req.as_of or "").strip() or None
    if as_of is not None:
        try:
            parse_date(as_of)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"as_of must be YYYY-MM-DD, got {as_of!r}"
            ) from None

    if not _slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="All analysis slots are busy. Try again in a moment.",
        )
    try:
        runner = _runner(req.capital_inr, req.profile)
        # save=False: the container filesystem is ephemeral, and a web caller
        # has not asked for a decision to be written to the audit log.
        decision = runner.analyse(symbol, save=False, as_of=as_of)
        payload = decision.model_dump(mode="json")
        payload["summary_markdown"] = runner.summary(decision)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface the failure, do not 500 blankly
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc
    finally:
        _slots.release()

    return JSONResponse(payload)


# --------------------------------------------------------------- the console


@app.get("/api/team")
def team() -> dict[str, Any]:
    """The desk roster, and which model each seat runs on."""
    cfg = build_config()
    from tradingagents_in.llm.provider import missing_api_key, resolve_model

    def model_for(tier: str) -> str:
        try:
            return resolve_model(cfg, tier) or "deterministic"
        except Exception:  # noqa: BLE001
            return "deterministic"

    # A provider with no key falls back on every call, so every seat really
    # is running deterministically whatever the config names.
    needs_key = missing_api_key(cfg)
    offline = (
        str(cfg.get("llm_provider", "")).lower() == "offline" or needs_key is not None
    )
    selected = set(cfg.get("selected_analysts") or [])
    roster = []
    for role, tier in (cfg.get("agent_model_tier") or {}).items():
        short = role.replace("_analyst", "").replace("_researcher", "")
        roster.append({
            "role": role,
            "name": short.replace("_", " "),
            "tier": tier,
            "model": "deterministic" if offline else model_for(tier),
            # An analyst the config switched off is shown greyed, not hidden:
            # a missing seat is worth seeing.
            "active": short in selected or not role.endswith("_analyst"),
        })
    return {
        "offline": offline,
        "provider": cfg.get("llm_provider"),
        "missing_api_key": needs_key,
        "agents": roster,
    }


@app.get("/api/systems")
def systems() -> dict[str, Any]:
    """Data sources, and whether this deployment can actually reach them."""
    cfg = build_config()
    online = bool(cfg.get("online_tools"))
    provider = str(cfg.get("data_provider", "")).lower()
    synthetic = provider == "synthetic"

    def state(needs_network: bool) -> str:
        if synthetic:
            return "sim"
        return "up" if (online or not needs_network) else "off"

    sources = [
        ("prices", provider or "yfinance", True),
        ("fundamentals", "screener + yfinance", True),
        ("news", "rss", True),
        ("sentiment", "social", True),
        ("macro", "rbi / fii-dii", True),
        ("microstructure", "nse", True),
        ("corp actions", "nse", True),
        ("calendar", "nse holidays", False),
        ("f&o", "lot sizes / expiry", False),
        ("risk rules", "deterministic", False),
        ("costs", "stt / stamp / gst", False),
        ("memory", "decision log", False),
    ]
    return {
        "online": online,
        "data_provider": provider,
        "sources": [
            {"name": n, "detail": d, "state": state(net)} for n, d, net in sources
        ],
    }


class RunRequest(AnalyseRequest):
    pass


@app.post("/api/runs")
def start_run(req: RunRequest) -> dict[str, Any]:
    """Start an analysis and return a handle to watch it."""
    symbol = req.symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=422, detail=f"Not a usable symbol: {symbol!r}")
    as_of = (req.as_of or "").strip() or None
    if as_of is not None:
        try:
            parse_date(as_of)
        except ValueError:
            raise HTTPException(status_code=422, detail="as_of must be YYYY-MM-DD") from None

    if not _slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="All analysis slots are busy.")

    run = registry.create(symbol)

    def work() -> dict[str, Any]:
        runner = _runner(req.capital_inr, req.profile)
        decision = runner.analyse(symbol, save=False, as_of=as_of)
        payload = decision.model_dump(mode="json")
        payload["summary_markdown"] = runner.summary(decision)
        return payload

    def body() -> None:
        try:
            registry.execute(run, work)
        finally:
            _slots.release()

    threading.Thread(target=body, name=f"run-{run.run_id}", daemon=True).start()
    return {"run_id": run.run_id, "symbol": symbol, "status": run.status}


@app.get("/api/runs/{run_id}")
def run_state(run_id: str) -> dict[str, Any]:
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    return {
        "run_id": run.run_id, "symbol": run.symbol, "status": run.status,
        "error": run.error, "decision": run.decision, "events": run.events,
    }


@app.get("/api/runs/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    """Server-sent events: the desk narrating itself, then the decision."""
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")

    def events():
        # Replay what already happened, so a late subscriber sees the whole
        # run. The replayed events are still sitting on the queue, so track
        # the highest sequence number sent and skip anything at or below it -
        # otherwise every event in flight when the browser connected arrives
        # twice, which on a screen means a progress bar that jumps backwards.
        last_seq = 0
        replayed = list(run.events)
        for event in replayed:
            last_seq = max(last_seq, event.get("seq", 0))
            yield f"data: {json.dumps(event, default=str)}\n\n"
        if run.finished_at and (not replayed or replayed[-1].get("done") is True):
            yield "data: {\"type\": \"status\", \"done\": true}\n\n"
            return
        while True:
            try:
                event = run.queue.get(timeout=20)
            except Exception:  # noqa: BLE001 - timeout: keep the connection warm
                yield ": keep-alive\n\n"
                if run.finished_at:
                    return
                continue
            if event.get("seq", 0) <= last_seq:
                continue
            last_seq = event.get("seq", last_seq)
            yield f"data: {json.dumps(event, default=str)}\n\n"
            # ``is True``, not truthiness: "done" is the terminal sentinel, and
            # a payload that carries its own "done" field must not be able to
            # close the stream. A progress count under that key did exactly
            # that - the screen ended after its first name.
            if event.get("done") is True:
                return

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this an intermediary may buffer the stream into silence.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/screen")
def screen_universe(req: ScreenRequest) -> dict[str, Any]:
    """Start a screen and return a handle to watch it.

    This does **not** block until the screen finishes, and that is the point.
    A screen is two passes of full pipeline runs - 64 of them for NIFTY50 -
    and on live data each one also makes about ten network fetches, so it runs
    for minutes. Served synchronously it held one request open that whole time
    with nothing on the wire, which looks identical to a hang in the browser
    and gets cut by any proxy with an idle timeout.

    So it starts like an analysis: a run id now, progress over
    ``/api/runs/{id}/stream``, the result as the last event.
    """
    from tradingagents_in.screen import resolve_universe

    universe = req.universe.strip()
    rank_by = req.rank_by

    # Resolve before taking a slot, so a misspelled universe is a fast 422
    # rather than a run that fails a minute later.
    try:
        symbols = resolve_universe(universe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if rank_by not in RANK_KEYS:
        raise HTTPException(
            status_code=422, detail=f"rank_by must be one of {', '.join(RANK_KEYS)}"
        )

    as_of = (req.as_of or "").strip() or None
    if as_of is not None:
        try:
            parse_date(as_of)
        except ValueError:
            raise HTTPException(status_code=422, detail="as_of must be YYYY-MM-DD") from None

    if not _slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="All analysis slots are busy.")

    run = registry.create(universe.upper(), kind="screen")
    capital = req.capital_inr
    profile = req.profile

    def work() -> dict[str, Any]:
        from tradingagents_in.screen import render as render_screen
        from tradingagents_in.screen import screen as run_screen

        runner = _runner(capital, profile)
        result = run_screen(
            runner, universe, rank_by=rank_by, as_of=as_of,
            on_progress=lambda event: run.publish({"type": "progress", **event}),
        )
        out = result.summary()
        out["candidates"] = [vars(c) for c in result.candidates]
        out["text"] = render_screen(result, limit=40)
        return out

    def body() -> None:
        try:
            registry.execute(run, work, result_type="screen")
        finally:
            _slots.release()

    threading.Thread(target=body, name=f"screen-{run.run_id}", daemon=True).start()
    return {
        "run_id": run.run_id,
        "universe": universe.upper(),
        "status": run.status,
        # The console shows this up front: a screen's cost is knowable before
        # it starts, and "64 runs" explains the wait far better than a spinner.
        "names": len(symbols),
        # From the config, not from a runner: building one costs the graph
        # build, and this response has to come back at once.
        "runs_expected": len(symbols) + min(
            len(symbols), int(build_config().get("max_open_positions", 12))
        ),
    }


class AutopilotRequest(BaseModel):
    universe: str = Field("NIFTY50", max_length=400)
    max_new_positions: int = Field(3, ge=1, le=12)
    square_off: bool = Field(False)
    rank_by: str = Field("reward_risk")
    capital_inr: float = Field(1_000_000.0, gt=0, le=1e11)
    profile: str = Field("retail")
    as_of: str | None = Field(None, max_length=10)


def _load_broker(capital_inr: float):
    """The paper book as it stands, or a fresh one at this capital."""
    from tradingagents_in.paper import PaperBroker

    if _BOOK_PATH.is_file():
        broker = PaperBroker.load(_BOOK_PATH, config=build_config())
    else:
        broker = PaperBroker(
            starting_cash_inr=capital_inr,
            config=build_config(),
            state_path=_BOOK_PATH,
        )
    broker.state_path = _BOOK_PATH
    return broker


def _book_payload(broker) -> dict[str, Any]:
    # Same reasoning as the orders: the model serialises itself, including
    # the computed properties a hand-written mapping would miss.
    positions = [p.to_dict() for p in broker.positions()]
    funds = broker.funds()
    return {
        # Set here, not by the caller: this payload also travels over SSE at
        # the end of an autopilot pass, and a console that only learns
        # "exists" from GET /api/book renders a freshly filled book as empty.
        "exists": True,
        "equity_inr": broker.equity_inr(),
        "cash_inr": funds.cash_inr,
        "starting_cash_inr": funds.starting_cash_inr,
        "realised_pnl_inr": funds.realised_pnl_inr,
        "positions": positions,
        # Both the model's own to_dict and the broker's own open predicate.
        # Hand-rolling either invents fields that do not exist (Order has
        # limit_price, not price) or filters on statuses that do not exist
        # (there is no "OPEN"), and both fail quietly or at the wrong time.
        "open_orders": [o.to_dict() for o in broker.orders(open_only=True)],
        "simulated": True,
    }


@app.get("/api/book")
def book() -> dict[str, Any]:
    """The paper book. Simulated throughout - no order ever leaves this process."""
    if not _BOOK_PATH.is_file():
        return {
            "exists": False, "simulated": True, "positions": [], "open_orders": [],
            "note": "No paper book yet. Run autopilot to start one.",
        }
    with _book_lock:
        return _book_payload(_load_broker(1_000_000.0))


@app.delete("/api/book")
def reset_book() -> dict[str, Any]:
    """Start again from flat. The only destructive call in this API."""
    with _book_lock:
        existed = _BOOK_PATH.is_file()
        if existed:
            _BOOK_PATH.unlink()
    return {"reset": True, "existed": existed}


@app.post("/api/autopilot")
def autopilot(req: AutopilotRequest) -> dict[str, Any]:
    """One autonomous pass: mark the book, screen, enter what it does not hold.

    Deliberately one pass rather than a daemon. The saved book *is* the
    state, so a pass is resumable and auditable, and nothing keeps trading
    because a process was left running.

    Everything here is simulated. No order leaves this process.
    """
    from tradingagents_in.screen import resolve_universe

    universe = req.universe.strip()
    try:
        symbols = resolve_universe(universe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if req.rank_by not in RANK_KEYS:
        raise HTTPException(
            status_code=422, detail=f"rank_by must be one of {', '.join(RANK_KEYS)}"
        )
    as_of = (req.as_of or "").strip() or None
    if as_of is not None:
        try:
            parse_date(as_of)
        except ValueError:
            raise HTTPException(status_code=422, detail="as_of must be YYYY-MM-DD") from None

    if not _slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="All analysis slots are busy.")

    run = registry.create(universe.upper(), kind="screen")

    def work() -> dict[str, Any]:
        from tradingagents_in.autopilot import run_pass

        runner = _runner(req.capital_inr, req.profile)
        # One pass at a time: the book is one file, read-modify-written.
        with _book_lock:
            broker = _load_broker(req.capital_inr)

            # Re-price what is held before anything new is considered, so a
            # stop is judged on this pass rather than on a stale fill price.
            held = list(broker.positions())
            prices: dict[str, float] = {}
            for index, position in enumerate(held, start=1):
                run.publish({
                    "type": "progress", "phase": "mark", "completed": index,
                    "total": len(held), "symbol": position.symbol,
                })
                try:
                    decision = runner.analyse(position.symbol, as_of=as_of, save=False)
                    zone = getattr(decision.proposal, "entry_zone", None)
                    if zone:
                        prices[position.symbol] = round((zone.low + zone.high) / 2, 2)
                except Exception as exc:  # noqa: BLE001 - a stale mark beats no pass
                    log.warning("could not re-price %s: %s", position.symbol, exc)

            result = run_pass(
                runner, broker, universe,
                as_of=as_of,
                prices=prices or None,
                max_new_positions=req.max_new_positions,
                square_off=req.square_off,
                rank_by=req.rank_by,
            )
            broker.save(_BOOK_PATH)
            out = result.summary()
            out["book"] = _book_payload(broker)
        out["orders"] = [o.to_dict() for o in result.placed]
        return out

    def body() -> None:
        try:
            registry.execute(run, work, result_type="autopilot")
        finally:
            _slots.release()

    threading.Thread(target=body, name=f"autopilot-{run.run_id}", daemon=True).start()
    return {
        "run_id": run.run_id,
        "universe": universe.upper(),
        "status": run.status,
        "names": len(symbols),
        "runs_expected": len(symbols) + min(
            len(symbols), int(build_config().get("max_open_positions", 12))
        ),
        "simulated": True,
    }


@app.get("/api/session")
def session_state() -> dict[str, Any]:
    """Where the market is in its day - the console's clock."""
    from tradingagents_in.dataflows.calendar_in import session_context
    from tradingagents_in.utils.timeutils import as_of_moment

    cfg = build_config()
    ctx = session_context(when=as_of_moment(cfg))
    return {
        "phase": ctx.phase.value,
        "now_ist": ctx.now.strftime("%Y-%m-%d %H:%M"),
        "trade_date": str(ctx.trade_date),
        "is_trading_day": ctx.is_trading_day,
        "minutes_to_close": ctx.minutes_to_close,
        "weekly_expiry": str(ctx.weekly_expiry),
        "days_to_weekly_expiry": ctx.days_to_weekly_expiry,
        "last_trading_day": str(ctx.last_trading_day),
        "as_of": cfg.get("as_of"),
    }


# ---------------------------------------------------------------- the console
#
# The console in frontend/ is served from here as well as from Vercel. Railway
# and Cloud Run run this image and nothing else, so without this the only page
# at / would be whatever the API shipped with - which is how a stale dashboard
# ends up live while the real one sits in a directory the image never copied.
#
# Mounted last on purpose: a mount at "/" matches everything, so every /api
# route has to be defined above it to be reachable.
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

if (_FRONTEND / "index.html").is_file():
    # html=True serves index.html for "/" and 404s to it, so the page owns its
    # own routing rather than the server guessing.
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="console")
    log.info("serving the console from %s", _FRONTEND)
else:
    log.warning("no frontend/ beside the app; / will report that instead of a page")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        # Says what is wrong rather than 404ing: this only happens when the
        # image was built without frontend/, which is a packaging mistake.
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8>"
            "<title>IndianTradingAgents</title>"
            "<body style='font:14px system-ui;background:#080b10;color:#dde6f0;"
            "padding:40px;max-width:46em'>"
            "<h1>API is up, console is missing</h1>"
            "<p>This build has no <code>frontend/</code> directory beside the "
            "app, so there is no page to serve. The API itself is fine - try "
            "<a style='color:#3ddbd9' href='/api/docs'>/api/docs</a> or "
            "<a style='color:#3ddbd9' href='/healthz'>/healthz</a>.</p>"
            "<p>If this is a container, its image needs to copy "
            "<code>frontend/</code>.</p></body>",
            status_code=503,
        )
