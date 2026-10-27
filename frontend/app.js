/* Dashboard for the IndianTradingAgents API.
 *
 * Deliberately dependency-free: one page calling one endpoint does not need a
 * framework, and a static bundle deploys to Vercel with no build step to break.
 *
 * The API base defaults to this site's own /api, which vercel.json proxies to
 * the backend. That keeps calls same-origin, so there is no CORS to configure
 * in the common case. Pointing it elsewhere is possible and then does need the
 * backend to name this origin.
 */

const PRESETS = [
  "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
  "ICICIBANK.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS",
];
const LS_API = "ita.apiBase";
const LS_HISTORY = "ita.history";
const HISTORY_MAX = 8;

const $ = (id) => document.getElementById(id);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

const num = (n, d = 2) =>
  (n === null || n === undefined || Number.isNaN(Number(n)))
    ? "—"
    : Number(n).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });

const inr = (n) =>
  (n === null || n === undefined || Number.isNaN(Number(n)))
    ? "—"
    : "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });

/* localStorage is per-viewer convenience only. It can throw in a private
 * window or with site data blocked, so every access is guarded and the page
 * works without it. */
const store = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : raw;
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* not essential */ }
  },
  remove(key) {
    try { localStorage.removeItem(key); } catch { /* not essential */ }
  },
};

const apiBase = () => (store.get(LS_API, "") || "/api").replace(/\/+$/, "");
const url = (path) => `${apiBase()}${path}`;

/* ------------------------------------------------------------------ status */

async function checkHealth() {
  const el = $("status");
  el.className = "status";
  el.innerHTML = '<span class="dot"></span>Connecting to the analysis API…';
  try {
    const res = await fetch(url("/config"), { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`API returned ${res.status}`);
    const cfg = await res.json();

    const mode = cfg.offline
      ? "offline engine, synthetic tape — deterministic, no API calls"
      : `LLM ${esc(cfg.llm_provider)} · data ${esc(cfg.data_provider)}`;
    el.className = "status good";
    el.innerHTML = `<span class="dot"></span>Connected · ${mode}`;

    if (cfg.capital_inr) $("capital").value = cfg.capital_inr;
    if (cfg.profile) $("profile").value = cfg.profile;
  } catch (err) {
    el.className = "status bad";
    el.innerHTML =
      `<span class="dot"></span>Cannot reach the API at <code>${esc(apiBase())}</code>. ` +
      explain(err);
  }
}

/* A failed fetch says almost nothing by design, so name the likely causes
 * rather than printing "Failed to fetch" and leaving it there. */
function explain(err) {
  const msg = String(err && err.message ? err.message : err);
  if (/failed to fetch|networkerror|load failed/i.test(msg)) {
    return "The backend may be asleep or down, the URL wrong, or the browser " +
           "blocked it as cross-origin — check Settings, and that this site's " +
           "origin is in ITA_WEB_CORS_ORIGINS on the backend.";
  }
  return esc(msg);
}

/* ----------------------------------------------------------------- history */

function readHistory() {
  try { return JSON.parse(store.get(LS_HISTORY, "[]")) || []; }
  catch { return []; }
}

function pushHistory(entry) {
  const next = [entry, ...readHistory().filter(
    (h) => !(h.symbol === entry.symbol && h.as_of === entry.as_of)
  )].slice(0, HISTORY_MAX);
  store.set(LS_HISTORY, JSON.stringify(next));
  renderHistory();
}

function renderHistory() {
  const rows = readHistory();
  $("history-card").classList.toggle("hidden", rows.length === 0);
  $("history").innerHTML = rows.map((h, i) => `
    <div class="hist-row" data-i="${i}">
      <span>${esc(h.symbol)}${h.as_of ? ` <span class="muted">@ ${esc(h.as_of)}</span>` : ""}</span>
      <span class="num">${esc(h.action)} · ${h.approved ? "approved" : "rejected"} · ${num(h.size, 2)}%</span>
    </div>`).join("");

  $("history").querySelectorAll(".hist-row").forEach((row) => {
    row.onclick = () => {
      const h = readHistory()[Number(row.dataset.i)];
      if (!h) return;
      $("symbol").value = h.symbol;
      $("asof").value = h.as_of || "";
      $("form").requestSubmit();
    };
  });
}

/* ------------------------------------------------------------------ render */

function render(d) {
  const p = d.proposal || {};
  const z = p.entry_zone || {};
  const costs = p.costs || {};
  const pm = d.portfolio_decision || {};
  const plan = d.investment_plan || {};
  const risk = d.risk_assessment || {};

  const uniq = (xs) => Array.from(new Set((xs || []).map(String)));
  const bullets = (xs) => uniq(xs).map((i) => `<li>${esc(i)}</li>`).join("");
  const list = (title, items) => uniq(items).length
    ? `<section class="card"><h2>${title}</h2><ul>${bullets(items)}</ul></section>` : "";

  const rows = [];
  const put = (k, v) => rows.push(`<tr><td>${k}</td><td class="num">${v}</td></tr>`);

  if (p.instrument) {
    put("Instrument",
      `${esc(p.instrument)} / ${esc(p.product_type || "")} / ${esc(p.time_horizon || "")}`);
  }
  if (z.low != null) put("Entry band", `${num(z.low)} – ${num(z.high)}`);
  if (p.stop_loss != null) put("Stop-loss", num(p.stop_loss));
  if (p.targets && p.targets.length) {
    put("Targets", p.targets
      .map((t) => `${num(t.price)} (${num(t.book_pct, 0)}%)`).join("<br>"));
  }
  put("Position size", `${num(d.position_size_pct)}% of capital`);
  if (p.quantity != null) put("Quantity", `${num(p.quantity, 0)} units`);
  put("Capital at risk", inr(d.capital_at_risk_inr));
  if (p.reward_risk != null) put("Reward : risk", `${num(p.reward_risk)} : 1`);
  // Per-side; the round-trip figure is in the risk notes.
  if (costs.total_pct_of_turnover != null) {
    put("Charges", `${num(costs.total_pct_of_turnover, 3)}% of turnover`);
  }
  if (costs.total_inr != null) put("Charges (abs)", inr(costs.total_inr));

  const signals = Object.entries(d.analyst_reports || {}).map(([k, v]) => {
    const sig = (v && (v.signal || v.verdict || v.stance)) || "";
    const conf = v && v.confidence;
    return `<tr><td>${esc(k)}</td><td class="num">${esc(sig)}${
      conf != null ? ` (${num(conf * 100, 0)}%)` : ""}</td></tr>`;
  }).join("");

  const breach = (b) => {
    if (typeof b === "string") return `<li>${esc(b)}</li>`;
    const head = [b.rule, b.severity].filter(Boolean).map(esc).join(" · ");
    const body = [b.detail, b.observed && `observed ${b.observed}`,
                  b.limit && `limit ${b.limit}`, b.remedy]
      .filter(Boolean).map(esc).join(" — ");
    return `<li>${head ? `<strong>${head}</strong>${body ? ": " : ""}` : ""}${body}</li>`;
  };

  const dated = $("asof").value
    ? `<span class="badge dated">as of ${esc($("asof").value)}</span>` : "";

  $("out").innerHTML = `
    <section class="card">
      <div class="verdict">
        <span class="action">${esc(d.final_action)}</span>
        <span class="badge ${d.approved ? "yes" : "no"}">${d.approved ? "approved" : "rejected"}</span>
        ${dated}
        <span class="muted">confidence ${num(d.confidence * 100, 0)}% ·
          ${esc(d.symbol)}${d.company_name ? ` · ${esc(d.company_name)}` : ""}${
            d.sector ? ` · ${esc(d.sector)}` : ""}</span>
      </div>
      <table>${rows.join("")}</table>
    </section>
    ${signals ? `<section class="card"><h2>Analyst signals</h2><table>${signals}</table></section>` : ""}
    ${plan.stance ? `<section class="card"><h2>Research — bull vs bear</h2>
      <div><strong>${esc(plan.stance)}</strong>${
        plan.conviction != null ? ` · conviction ${num(plan.conviction * 100, 0)}%` : ""}${
        plan.winning_side ? ` · debate won by ${esc(plan.winning_side)}` : ""}</div>
      ${plan.verdict ? `<p>${esc(plan.verdict)}</p>` : ""}
      ${uniq(plan.decisive_arguments).length
        ? `<h2>Decisive</h2><ul>${bullets(plan.decisive_arguments)}</ul>` : ""}
      ${uniq(plan.dismissed_arguments).length
        ? `<h2>Dismissed</h2><ul>${bullets(plan.dismissed_arguments)}</ul>` : ""}
    </section>` : ""}
    ${pm.portfolio_reasoning ? `<section class="card"><h2>Portfolio manager</h2>
      <div>${esc(pm.portfolio_reasoning)}</div>
      ${uniq(pm.conditions).length ? `<ul>${bullets(pm.conditions)}</ul>` : ""}
    </section>` : ""}
    ${risk.sizing_rationale || (risk.breaches || []).length
      ? `<section class="card"><h2>Risk manager${
          risk.verdict ? ` — ${esc(risk.verdict)}` : ""}</h2>
         ${risk.sizing_rationale ? `<div>${esc(risk.sizing_rationale)}</div>` : ""}
         ${(risk.breaches || []).length
           ? `<ul>${risk.breaches.map(breach).join("")}</ul>` : ""}
         ${uniq(risk.required_changes).length
           ? `<ul>${bullets(risk.required_changes)}</ul>` : ""}
      </section>` : ""}
    ${list("Execution notes", p.execution_notes)}
    ${list("Risk notes", d.risk_notes)}
    ${list("What to watch", plan.key_monitorables)}
    ${list("Errors", d.errors)}
    <section class="card muted" style="font-size:12px">
      run ${esc(d.run_id)}${d.elapsed_seconds != null ? ` · ${num(d.elapsed_seconds, 1)}s` : ""}${
        d.as_of ? ` · ${esc(d.as_of)}` : ""}
    </section>`;
}

/* ------------------------------------------------------------------- wiring */

PRESETS.forEach((sym) => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.textContent = sym.replace(".NS", "");
  chip.onclick = () => { $("symbol").value = sym; $("form").requestSubmit(); };
  $("presets").appendChild(chip);
});

$("settings-toggle").onclick = () => {
  const panel = $("settings");
  const open = panel.classList.toggle("hidden");
  $("settings-toggle").setAttribute("aria-expanded", String(!open));
  if (!open) $("api-base").value = store.get(LS_API, "");
};

$("save-settings").onclick = async () => {
  const value = $("api-base").value.trim();
  if (value) store.set(LS_API, value); else store.remove(LS_API);
  $("settings").classList.add("hidden");
  $("settings-toggle").setAttribute("aria-expanded", "false");
  await checkHealth();
};

$("reset-settings").onclick = async () => {
  store.remove(LS_API);
  $("api-base").value = "";
  await checkHealth();
};

$("clear-history").onclick = () => { store.remove(LS_HISTORY); renderHistory(); };

$("form").onsubmit = async (event) => {
  event.preventDefault();
  const button = $("go");
  button.disabled = true;
  button.innerHTML = '<span class="spin">●</span> running';
  $("out").innerHTML =
    '<section class="card muted">The desk is deliberating — analysts, then the ' +
    'bull/bear debate, then risk.</section>';

  const asOf = $("asof").value || null;
  try {
    const res = await fetch(url("/analyse"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: $("symbol").value,
        capital_inr: Number($("capital").value),
        profile: $("profile").value,
        as_of: asOf,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    render(data);
    pushHistory({
      symbol: data.symbol, as_of: asOf, action: data.final_action,
      approved: data.approved, size: data.position_size_pct,
    });
  } catch (err) {
    $("out").innerHTML = `<section class="card err">${explain(err)}</section>`;
  } finally {
    button.disabled = false;
    button.textContent = "Analyse";
  }
};

renderHistory();
checkHealth();
