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

import os
import re
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.utils.timeutils import parse_date

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


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """Railway's health check. Must not build a runner or touch the network."""
    return {"status": "ok"}


@app.get("/api/config")
def config_summary() -> dict[str, Any]:
    """What this deployment is actually configured to do."""
    cfg = build_config()
    return {
        "llm_provider": cfg.get("llm_provider"),
        "data_provider": cfg.get("data_provider"),
        "online_tools": cfg.get("online_tools"),
        "profile": cfg.get("profile"),
        "capital_inr": cfg.get("capital_inr"),
        "max_debate_rounds": cfg.get("max_debate_rounds"),
        "offline": str(cfg.get("llm_provider", "")).lower() == "offline",
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IndianTradingAgents</title>
<style>
  :root {
    --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a1a; --muted: #6b6b6b;
    --line: #e4e0da; --accent: #1f6f4a; --warn: #9a3412; --chip: #f1efea;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #141414; --panel: #1c1c1c; --ink: #ececec; --muted: #9a9a9a;
      --line: #2e2e2e; --accent: #4ade80; --warn: #fb923c; --chip: #262626;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 18px; margin-bottom: 16px;
  }
  form { display: grid; grid-template-columns: 1fr 150px 150px auto; gap: 10px; align-items: end; }
  @media (max-width: 680px) { form { grid-template-columns: 1fr; } }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
  input, select, button {
    width: 100%; padding: 9px 11px; font-size: 14px; font-family: inherit;
    border: 1px solid var(--line); border-radius: 7px;
    background: var(--bg); color: var(--ink);
  }
  button {
    background: var(--accent); color: #fff; border-color: transparent;
    font-weight: 600; cursor: pointer; white-space: nowrap;
  }
  button:disabled { opacity: 0.55; cursor: progress; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  .chip {
    background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
    padding: 3px 10px; font-size: 12px; cursor: pointer; color: var(--muted);
  }
  .verdict { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .action { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .badge {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    padding: 3px 9px; border-radius: 999px; letter-spacing: 0.04em;
  }
  .ok { background: var(--accent); color: #fff; }
  .no { background: var(--warn); color: #fff; }
  table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }
  td { padding: 7px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
  td:first-child { color: var(--muted); width: 38%; }
  .num { font-variant-numeric: tabular-nums; }
  ul { margin: 8px 0 0; padding-left: 18px; }
  li { margin-bottom: 5px; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
       color: var(--muted); margin: 0 0 8px; font-weight: 600; }
  .err { color: var(--warn); }
  .foot { color: var(--muted); font-size: 12px; margin-top: 24px; text-align: center; }
  .hidden { display: none; }
  .mode { font-size: 12px; color: var(--muted); }
  .spin { display: inline-block; animation: s 1s linear infinite; }
  @keyframes s { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="wrap">
  <h1>IndianTradingAgents</h1>
  <div class="sub">Multi-agent LLM research for NSE &amp; BSE &middot; <span id="mode" class="mode">checking configuration&hellip;</span></div>

  <div class="card">
    <form id="f">
      <div>
        <label for="sym">Symbol</label>
        <input id="sym" value="RELIANCE.NS" autocomplete="off" spellcheck="false">
      </div>
      <div>
        <label for="cap">Capital (INR)</label>
        <input id="cap" type="number" value="1000000" min="1000" step="1000">
      </div>
      <div>
        <label for="prof">Profile</label>
        <select id="prof">
          <option value="retail">retail</option>
          <option value="institutional">institutional</option>
        </select>
      </div>
      <div><button id="go" type="submit">Analyse</button></div>
    </form>
    <div class="chips" id="chips"></div>
  </div>

  <div id="out"></div>
  <div class="foot">Research output, not investment advice. Verify against the
    exchange and consult a SEBI-registered adviser.</div>
</div>

<script>
const PRESETS = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ITC.NS","SBIN.NS","TATAMOTORS.NS"];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const inr = (n) => (n === null || n === undefined || isNaN(n)) ? "—"
  : "₹" + Number(n).toLocaleString("en-IN", {maximumFractionDigits: 2});
const num = (n, d = 2) => (n === null || n === undefined || isNaN(n)) ? "—"
  : Number(n).toLocaleString("en-IN", {minimumFractionDigits: d, maximumFractionDigits: d});

PRESETS.forEach(s => {
  const b = document.createElement("span");
  b.className = "chip"; b.textContent = s.replace(".NS", "");
  b.onclick = () => { $("sym").value = s; $("f").requestSubmit(); };
  $("chips").appendChild(b);
});

fetch("/api/config").then(r => r.json()).then(c => {
  $("mode").textContent = c.offline
    ? "offline engine, synthetic tape — deterministic, no API calls"
    : `LLM ${c.llm_provider} · data ${c.data_provider}`;
  if (c.capital_inr) $("cap").value = c.capital_inr;
  if (c.profile) $("prof").value = c.profile;
}).catch(() => { $("mode").textContent = ""; });

$("f").onsubmit = async (e) => {
  e.preventDefault();
  const btn = $("go"), out = $("out");
  btn.disabled = true; btn.innerHTML = '<span class="spin">●</span> running';
  out.innerHTML = '<div class="card"><span class="mode">The desk is deliberating — analysts, then the bull/bear debate, then risk.</span></div>';
  try {
    const r = await fetch("/api/analyse", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        symbol: $("sym").value,
        capital_inr: Number($("cap").value),
        profile: $("prof").value,
      }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || ("HTTP " + r.status));
    render(d);
  } catch (err) {
    out.innerHTML = `<div class="card err">${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Analyse";
  }
};

function render(d) {
  const p = d.proposal || {};
  const z = p.entry_zone || {};
  const costs = p.costs || {};
  const pm = d.portfolio_decision || {};
  const plan = d.investment_plan || {};
  const risk = d.risk_assessment || {};
  const rows = [];

  const put = (k, v) => rows.push(`<tr><td>${k}</td><td class="num">${v}</td></tr>`);
  if (p.instrument) put("Instrument",
    `${esc(p.instrument)} / ${esc(p.product_type || "")} / ${esc(p.time_horizon || "")}`);
  if (z.low != null) put("Entry band", `${num(z.low)} – ${num(z.high)}`);
  if (p.stop_loss != null) put("Stop-loss", num(p.stop_loss));
  if (p.targets && p.targets.length) put("Targets",
    p.targets.map(t => `${num(t.price)} (${num(t.book_pct, 0)}%)`).join("<br>"));
  put("Position size", num(d.position_size_pct) + "% of capital");
  if (p.quantity != null) put("Quantity", num(p.quantity, 0) + " units");
  put("Capital at risk", inr(d.capital_at_risk_inr));
  if (p.reward_risk != null) put("Reward : risk", num(p.reward_risk) + " : 1");
  // The risk notes carry the round-trip figure; this field is per-side, so it
  // is labelled for what it actually is.
  if (costs.total_pct_of_turnover != null)
    put("Charges", num(costs.total_pct_of_turnover, 3) + "% of turnover");
  if (costs.total_inr != null) put("Charges (abs)", inr(costs.total_inr));

  // The agents can emit the same note from two paths (e.g. a weekly and a
  // monthly expiry landing on one date). Identical bullets just look broken.
  const uniq = (xs) => Array.from(new Set((xs || []).map(x => String(x))));
  const bullets = (xs) => uniq(xs).map(i => `<li>${esc(i)}</li>`).join("");
  const list = (title, items) => uniq(items).length
    ? `<div class="card"><h2>${title}</h2><ul>${bullets(items)}</ul></div>` : "";

  // A breach is a structured object; show its fields, not its JSON.
  const breach = (b) => {
    if (typeof b === "string") return `<li>${esc(b)}</li>`;
    const head = [b.rule, b.severity].filter(Boolean).map(esc).join(" · ");
    const body = [b.detail, b.observed && `observed ${b.observed}`,
                  b.limit && `limit ${b.limit}`, b.remedy]
      .filter(Boolean).map(esc).join(" — ");
    return `<li>${head ? `<strong>${head}</strong>${body ? ": " : ""}` : ""}${body}</li>`;
  };

  const signals = Object.entries(d.analyst_reports || {}).map(([k, v]) => {
    const s = (v && (v.signal || v.verdict || v.stance)) || "";
    const c = v && v.confidence;
    return `<tr><td>${esc(k)}</td><td class="num">${esc(s)}${
      c != null ? " (" + num(c * 100, 0) + "%)" : ""}</td></tr>`;
  }).join("");

  $("out").innerHTML = `
    <div class="card">
      <div class="verdict">
        <span class="action">${esc(d.final_action)}</span>
        <span class="badge ${d.approved ? "ok" : "no"}">${d.approved ? "approved" : "rejected"}</span>
        <span class="mode">confidence ${num(d.confidence * 100, 0)}% &middot;
          ${esc(d.symbol)}${d.company_name ? " &middot; " + esc(d.company_name) : ""}${
            d.sector ? " &middot; " + esc(d.sector) : ""}</span>
      </div>
      <table>${rows.join("")}</table>
    </div>
    ${signals ? `<div class="card"><h2>Analyst signals</h2><table>${signals}</table></div>` : ""}
    ${plan.stance ? `<div class="card"><h2>Research — bull vs bear</h2>
      <div><strong>${esc(plan.stance)}</strong>${
        plan.conviction != null ? ` · conviction ${num(plan.conviction * 100, 0)}%` : ""}${
        plan.winning_side ? ` · debate won by ${esc(plan.winning_side)}` : ""}</div>
      ${plan.verdict ? `<div style="margin-top:8px">${esc(plan.verdict)}</div>` : ""}
      ${uniq(plan.decisive_arguments).length ? `<div style="margin-top:10px"><h2>Decisive</h2><ul>${
        bullets(plan.decisive_arguments)}</ul></div>` : ""}
      ${uniq(plan.dismissed_arguments).length ? `<div style="margin-top:10px"><h2>Dismissed</h2><ul>${
        bullets(plan.dismissed_arguments)}</ul></div>` : ""}
      </div>` : ""}
    ${pm.portfolio_reasoning ? `<div class="card"><h2>Portfolio manager</h2>
      <div>${esc(pm.portfolio_reasoning)}</div>
      ${uniq(pm.conditions).length ? `<ul>${bullets(pm.conditions)}</ul>` : ""}</div>` : ""}
    ${risk.sizing_rationale || (risk.breaches || []).length ? `<div class="card"><h2>Risk manager${
      risk.verdict ? " — " + esc(risk.verdict) : ""}</h2>
      ${risk.sizing_rationale ? `<div>${esc(risk.sizing_rationale)}</div>` : ""}
      ${(risk.breaches || []).length ? `<ul>${
        risk.breaches.map(breach).join("")}</ul>` : ""}
      ${uniq(risk.required_changes).length ? `<ul>${bullets(risk.required_changes)}</ul>` : ""}</div>` : ""}
    ${list("Execution notes", p.execution_notes)}
    ${list("Risk notes", d.risk_notes)}
    ${list("What to watch", plan.key_monitorables)}
    ${list("Errors", d.errors)}
    <div class="card mode">run ${esc(d.run_id)}${
      d.elapsed_seconds != null ? " · " + num(d.elapsed_seconds, 1) + "s" : ""}${
      d.as_of ? " · as of " + esc(d.as_of) : ""}</div>`;
}
</script>
</body>
</html>
"""
