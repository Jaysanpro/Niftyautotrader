/* The trading console.
 *
 * Three panes over one API. The left rail is the desk's standing state (who is
 * on it, what it can reach, where the market is); the middle is the
 * conversation and the live trace of a run; the right is whatever the last
 * question produced.
 *
 * The trace is the reason this is worth building: the agents already narrate
 * themselves, so a run is watchable rather than a spinner followed by a
 * verdict. Everything arrives over SSE from /api/runs/{id}/stream.
 *
 * Dependency-free on purpose - it deploys as static files with no build step.
 */

const API = (localStorage.getItem("ita.apiBase") || "/api").replace(/\/+$/, "");
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (n, d = 2) => (n === null || n === undefined || Number.isNaN(Number(n)))
  ? "—" : Number(n).toLocaleString("en-IN",
      { minimumFractionDigits: d, maximumFractionDigits: d });
const inr = (n) => (n === null || n === undefined || Number.isNaN(Number(n)))
  ? "—" : "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });

const state = {
  agents: [],
  running: false,
  steps: 0,
  errors: 0,
  startedAt: 0,
  decision: null,
  screen: null,
  book: null,
  autopilot: null,
  tab: "start",
  timer: null,
};

/* ------------------------------------------------------------------ boot */

async function get(path) {
  const res = await fetch(API + path, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

async function boot() {
  renderStage();
  try {
    const [team, systems, session] = await Promise.all([
      get("/team"), get("/systems"), get("/session"),
    ]);
    paintTeam(team);
    paintSystems(systems);
    paintSession(session);
  } catch (err) {
    $("chip-mode").textContent = "api unreachable";
    $("chip-mode").className = "chip sim";
    $("team").innerHTML =
      `<div class="empty">Cannot reach the API at <code>${esc(API)}</code>.</div>`;
  }
}

function paintTeam(team) {
  state.agents = team.agents;
  $("team-count").textContent = `${team.agents.length}`;
  $("c-agents").textContent = team.agents.length;
  // Three states, not two. A provider named without its key is the one worth
  // calling out: the desk answers normally, so nothing looks wrong, but no
  // model has been asked anything.
  if (team.missing_api_key) {
    $("chip-mode").textContent = `${team.missing_api_key} not set · deterministic`;
    $("chip-mode").className = "chip warn";
    $("chip-mode").title =
      `${team.provider} is configured but ${team.missing_api_key} is not set, `
      + `so every agent is answering from its deterministic draft. `
      + `Set it in your host's environment variables to use the model.`;
  } else {
    $("chip-mode").textContent = team.offline
      ? "offline engine · deterministic" : `live · ${team.provider}`;
    $("chip-mode").className = "chip " + (team.offline ? "sim" : "live");
    $("chip-mode").title = "";
  }

  const deep = team.agents.find((a) => a.tier === "deep");
  const quick = team.agents.find((a) => a.tier === "quick");
  $("chip-deep").textContent = deep ? deep.model : "deep";
  $("chip-quick").textContent = quick ? quick.model : "quick";

  $("team").innerHTML = team.agents.map((a) => `
    <div class="agent ${a.active ? "" : "off"}" data-agent="${esc(a.role)}"
         data-short="${esc(a.role.replace(/_analyst$|_researcher$/, ""))}">
      <span class="dot"></span>
      <span class="nm">${esc(a.name)}</span>
      <span class="verdict"></span>
    </div>`).join("");
}

function paintSystems(sys) {
  $("systems-mode").textContent = sys.online ? "online" : (sys.data_provider || "offline");
  $("systems").innerHTML = sys.sources.map((s) =>
    `<span class="sys ${esc(s.state)}" title="${esc(s.detail)}"><i></i>${esc(s.name)}</span>`
  ).join("");
}

function paintSession(s) {
  $("session-phase").textContent = s.phase.toLowerCase();
  $("session-meta").innerHTML =
    `<b>${esc(s.now_ist)} IST</b>` +
    `${esc(s.phase.toLowerCase())}${s.minutes_to_close != null
      ? ` · ${s.minutes_to_close}m to close` : ""}<br>` +
    `expiry ${esc(s.weekly_expiry)} (${s.days_to_weekly_expiry}d)`;
  drawDial(s);
}

/* The session as a clock face: the traded part of the day, and where in it
 * we are. A dial because "3 hours left" means less at a glance than an arc. */
function drawDial(s) {
  const c = $("dial");
  const ctx = c.getContext("2d");
  const R = c.width / 2, r = R - 14;
  ctx.clearRect(0, 0, c.width, c.height);

  const ring = (from, to, colour, width) => {
    ctx.beginPath();
    ctx.arc(R, R, r, from, to);
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.lineCap = "round";
    ctx.stroke();
  };
  // 09:15 to 15:30 mapped onto three quarters of the circle, starting top-left.
  const start = -Math.PI * 0.75, end = Math.PI * 0.75;
  ring(start, end, "#1c2733", 9);

  const total = 375; // minutes in the session
  const left = s.minutes_to_close;
  const done = left == null
    ? (s.phase === "CLOSED" || s.phase === "HOLIDAY" ? 1 : 0)
    : Math.max(0, Math.min(1, (total - left) / total));
  if (done > 0) {
    ring(start, start + (end - start) * done,
         s.phase === "REGULAR" ? "#3ddbd9" : "#53657a", 9);
  }
  ctx.beginPath();
  ctx.arc(R, R, 4, 0, Math.PI * 2);
  ctx.fillStyle = s.is_trading_day ? "#3ddbd9" : "#53657a";
  ctx.fill();
}

/* The desk visual: a ring of nodes, one per agent, lighting as they report.
 * Decorative, but tied to real state - it is the same roster, in order. */
let coreFrame = 0;
function drawCore() {
  const c = $("core");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.34;
  const nodes = state.agents.length || 10;
  coreFrame += 1;

  const pts = [];
  for (let i = 0; i < nodes; i++) {
    const a = (i / nodes) * Math.PI * 2 - Math.PI / 2;
    const wobble = state.running ? Math.sin(coreFrame / 18 + i) * 4 : 0;
    pts.push([cx + Math.cos(a) * (R + wobble), cy + Math.sin(a) * (R + wobble)]);
  }
  ctx.strokeStyle = "rgba(61,219,217,0.10)";
  ctx.lineWidth = 1;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j += 3) {
      ctx.beginPath();
      ctx.moveTo(pts[i][0], pts[i][1]);
      ctx.lineTo(pts[j][0], pts[j][1]);
      ctx.stroke();
    }
  }
  state.agents.forEach((agent, i) => {
    if (!pts[i]) return;
    const el = document.querySelector(`[data-agent="${agent.role}"]`);
    const live = el && el.classList.contains("thinking");
    const done = el && el.classList.contains("done");
    ctx.beginPath();
    ctx.arc(pts[i][0], pts[i][1], live ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = live ? "#3ddbd9" : done ? "#4ade80" : "#243240";
    ctx.fill();
  });
  requestAnimationFrame(drawCore);
}

/* ---------------------------------------------------------------- thread */

function say(who, text, mine = false) {
  const div = document.createElement("div");
  div.className = "msg" + (mine ? " you" : "");
  div.innerHTML = `<div class="who">${esc(who)}</div>
                   <div class="bubble">${esc(text)}</div>`;
  $("thread").appendChild(div);
  $("thread").scrollTop = $("thread").scrollHeight;
  return div;
}

function openTrace() {
  const box = document.createElement("div");
  box.className = "trace";
  $("thread").appendChild(box);
  return box;
}

function traceLine(box, agent, message, kind = "") {
  const line = document.createElement("div");
  line.className = "trace-line " + (agent ? "" : "plain ") + kind;
  line.innerHTML = `<span class="ag">${esc(agent || "·")}</span>
                    <span>${esc(message)}</span>`;
  box.appendChild(line);
  $("thread").scrollTop = $("thread").scrollHeight;
}

function markAgent(short, verdict, done = true) {
  const el = document.querySelector(`[data-short="${short}"]`)
    || document.querySelector(`[data-agent="${short}"]`);
  if (!el) return;
  el.classList.remove("thinking");
  el.classList.add(done ? "done" : "thinking");
  if (verdict) el.querySelector(".verdict").textContent = verdict;
}

function resetAgents() {
  document.querySelectorAll(".agent").forEach((el) => {
    el.classList.remove("thinking", "done");
    el.querySelector(".verdict").textContent = "";
  });
}

/* ------------------------------------------------------------- running it */

function setRunning(on, label) {
  state.running = on;
  $("send").disabled = on;
  $("send").innerHTML = on ? '<span class="spin">●</span>' : "→";
  $("core-state").textContent = on ? "working" : "idle";
  const chip = $("c-state");
  chip.textContent = label || (on ? "working" : "idle");
  chip.className = "state " + (on ? "" : label === "failed" ? "failed" : "idle");
  if (on) {
    state.startedAt = Date.now();
    state.timer = setInterval(() => {
      $("c-elapsed").textContent = ((Date.now() - state.startedAt) / 1000).toFixed(1) + "s";
    }, 100);
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

async function analyse(symbol) {
  resetAgents();
  state.steps = 0; state.errors = 0;
  $("c-steps").textContent = "0"; $("c-errors").textContent = "0";
  setRunning(true);
  say("You", symbol, true);
  const box = openTrace();

  let runId;
  try {
    const res = await fetch(API + "/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol,
        capital_inr: Number($("capital").value) || 1000000,
        as_of: $("asof").value || null,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    runId = data.run_id;
  } catch (err) {
    traceLine(box, null, String(err.message || err), "err");
    setRunning(false, "failed");
    say("Desk", "That run could not start.");
    return;
  }

  // SSE rather than polling: the point is to watch it think.
  await new Promise((resolve) => {
    const es = new EventSource(`${API}/runs/${runId}/stream`);
    es.onmessage = (event) => {
      let e;
      try { e = JSON.parse(event.data); } catch { return; }

      if (e.type === "log") {
        state.steps += 1;
        $("c-steps").textContent = state.steps;
        traceLine(box, e.agent ? e.agent.replace(/_/g, " ") : null, e.message,
                  e.level === "error" ? "err" : "");
        if (e.agent) {
          const verdict = (e.message.split(":")[1] || "").trim();
          markAgent(e.agent, verdict.slice(0, 26));
        }
      } else if (e.type === "decision") {
        state.decision = e.decision;
        show("decision");
        say("Desk", summarise(e.decision));
      } else if (e.type === "error") {
        state.errors += 1;
        $("c-errors").textContent = state.errors;
        traceLine(box, null, e.error, "err");
        say("Desk", "The run failed. The trace above says where.");
      }

      if (e.done === true) { es.close(); resolve(); }
    };
    es.onerror = () => { es.close(); resolve(); };
  });

  setRunning(false, state.errors ? "failed" : "idle");
  paintOps();
}

function summarise(d) {
  if (!d) return "No decision came back.";
  const verb = d.approved ? "would" : "would not";
  const size = d.position_size_pct ? ` at ${num(d.position_size_pct)}% of capital` : "";
  return `${d.symbol}: ${d.final_action}. The desk ${verb} take it${size}, `
       + `at ${num(d.confidence * 100, 0)}% confidence.`;
}

async function runScreen(universe) {
  resetAgents();
  state.steps = 0; state.errors = 0;
  $("c-steps").textContent = "0"; $("c-errors").textContent = "0";
  setRunning(true, "screening");
  say("You", `screen ${universe}`, true);
  const box = openTrace();

  let start;
  try {
    const res = await fetch(API + "/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        universe,
        capital_inr: Number($("capital").value) || 1000000,
        as_of: $("asof").value || null,
      }),
    });
    start = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(start.detail || `HTTP ${res.status}`);
  } catch (err) {
    traceLine(box, null, String(err.message || err), "err");
    setRunning(false, "failed");
    say("Desk", "That screen did not start.");
    return;
  }

  // Say the cost up front. A screen is minutes, and a user who knows it is
  // 64 runs waits; one shown a spinner assumes it has hung.
  traceLine(box, null, `${universe}: ${start.names} names, `
    + `${start.runs_expected} full pipeline runs — each name is analysed twice, `
    + `once on merit and once against the book`);
  showProgress(0, start.runs_expected, "starting");

  await new Promise((resolve) => {
    const es = new EventSource(`${API}/runs/${start.run_id}/stream`);
    let merit = 0;
    es.onmessage = (event) => {
      let e;
      try { e = JSON.parse(event.data); } catch { return; }

      if (e.type === "progress") {
        // Pass 2 continues the same bar rather than restarting it: to the
        // person waiting it is one job, not two.
        const overall = e.phase === "merit" ? e.completed : merit + e.completed;
        if (e.phase === "merit") merit = Math.max(merit, e.completed);
        state.steps = overall;
        $("c-steps").textContent = overall;
        showProgress(overall, start.runs_expected, e.phase);
        if (e.symbol) {
          const rr = e.reward_risk != null ? ` · R:R ${num(e.reward_risk)}` : "";
          const mark = e.error ? "failed" : (e.approved ? `${e.action} ✓` : e.action || "—");
          traceLine(box, e.symbol.replace(/\.(NS|BO)$/, ""),
                    `${mark}${rr}`, e.error ? "err" : "");
        }
        if (e.error) { state.errors += 1; $("c-errors").textContent = state.errors; }
      } else if (e.type === "screen") {
        state.screen = e.screen;
        show("screen");
        say("Desk", `${e.screen.scanned} names scanned. `
          + `${e.screen.passed_standalone} passed on merit; `
          + `${e.screen.selected} fit the book once the caps bind.`);
      } else if (e.type === "log") {
        // Screens forward warnings only; an info-level flood of 64 runs would
        // bury the progress that is the point of watching.
        traceLine(box, null, e.message, e.level === "error" ? "err" : "warn");
      } else if (e.type === "error") {
        state.errors += 1;
        $("c-errors").textContent = state.errors;
        traceLine(box, null, e.error, "err");
        say("Desk", "The screen failed. The trace above says where.");
      }

      if (e.done === true) { es.close(); resolve(); }
    };
    es.onerror = () => { es.close(); resolve(); };
  });

  hideProgress();
  setRunning(false, state.errors && !state.screen ? "failed" : "idle");
}

async function runAutopilot(universe) {
  resetAgents();
  state.steps = 0; state.errors = 0;
  $("c-steps").textContent = "0"; $("c-errors").textContent = "0";
  setRunning(true, "autopilot");
  say("You", `autopilot ${universe}`, true);
  const box = openTrace();

  let start;
  try {
    const res = await fetch(API + "/autopilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        universe,
        capital_inr: Number($("capital").value) || 1000000,
        as_of: $("asof").value || null,
        max_new_positions: 3,
      }),
    });
    start = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(start.detail || `HTTP ${res.status}`);
  } catch (err) {
    traceLine(box, null, String(err.message || err), "err");
    setRunning(false, "failed");
    say("Desk", "Autopilot did not start.");
    return;
  }

  traceLine(box, null, `${universe}: marking what is held, then screening `
    + `${start.names} names. Simulated orders only — nothing is sent to a broker.`);
  showProgress(0, start.runs_expected, "starting");

  await new Promise((resolve) => {
    const es = new EventSource(`${API}/runs/${start.run_id}/stream`);
    let merit = 0;
    es.onmessage = (event) => {
      let e;
      try { e = JSON.parse(event.data); } catch { return; }

      if (e.type === "progress") {
        if (e.phase === "mark") {
          traceLine(box, e.symbol.replace(/\.(NS|BO)$/, ""), "re-pricing what is held");
          return;
        }
        const overall = e.phase === "merit" ? e.completed : merit + e.completed;
        if (e.phase === "merit") merit = Math.max(merit, e.completed);
        state.steps = overall;
        $("c-steps").textContent = overall;
        showProgress(overall, start.runs_expected, e.phase);
        if (e.symbol) {
          const rr = e.reward_risk != null ? ` · R:R ${num(e.reward_risk)}` : "";
          const mark = e.error ? "failed" : (e.approved ? `${e.action} ✓` : e.action || "—");
          traceLine(box, e.symbol.replace(/\.(NS|BO)$/, ""), `${mark}${rr}`,
                    e.error ? "err" : "");
        }
      } else if (e.type === "autopilot") {
        state.autopilot = e.autopilot;
        state.book = e.autopilot.book;
        show("book");
        say("Desk", summarisePass(e.autopilot));
      } else if (e.type === "log") {
        traceLine(box, null, e.message, e.level === "error" ? "err" : "warn");
      } else if (e.type === "error") {
        state.errors += 1;
        $("c-errors").textContent = state.errors;
        traceLine(box, null, e.error, "err");
        say("Desk", "The pass failed. The trace above says where.");
      }

      if (e.done === true) { es.close(); resolve(); }
    };
    es.onerror = () => { es.close(); resolve(); };
  });

  hideProgress();
  setRunning(false, state.errors && !state.autopilot ? "failed" : "idle");
}

function summarisePass(a) {
  const bits = [`${a.scanned} scanned`];
  if (a.orders_placed) bits.push(`${a.orders_placed} orders placed, ${a.orders_filled} filled`);
  else bits.push("no new orders");
  const skipped = Object.keys(a.skipped || {}).length;
  if (skipped) bits.push(`${skipped} skipped`);
  if (a.squared_off) bits.push(`${a.squared_off} squared off`);
  return bits.join(" · ") + `. Book is worth ${inr(a.equity_inr)}. Simulated — `
    + "no order left this process.";
}

async function refreshBook() {
  try {
    const res = await fetch(API + "/book");
    state.book = await res.json();
  } catch {
    state.book = null;
  }
  if (state.tab === "book") renderStage();
}

function bookHtml() {
  const b = state.book;
  if (!b) {
    refreshBook();
    return `<p class="note">Loading the book…</p>`;
  }
  if (!b.exists) {
    return `<h1 class="start-head">No book yet</h1>
      <p class="start-sub">${esc(b.note || "")} Try <code>autopilot NIFTYIT</code> —
      it screens the universe, places the orders the desk would place, and keeps
      the positions between passes.</p>
      <p class="disclaimer">Simulated throughout. No order ever leaves this process.</p>`;
  }
  const pnl = (b.equity_inr || 0) - (b.starting_cash_inr || 0);
  const pct = b.starting_cash_inr ? (pnl / b.starting_cash_inr) * 100 : 0;
  const rows = (b.positions || []).map((p) => `
    <tr>
      <td>${esc(p.symbol)}</td>
      <td>${esc(p.product)}</td>
      <td class="n">${p.quantity}</td>
      <td class="n">${num(p.average_price)}</td>
      <td class="n">${p.last_price != null ? num(p.last_price) : "—"}</td>
      <td class="n ${p.unrealised_pnl_inr >= 0 ? "up" : "down"}">${num(p.unrealised_pnl_inr)}</td>
    </tr>`).join("");
  const orders = (b.open_orders || []).map((o) => `
    <tr>
      <td>${esc(o.symbol)}</td>
      <td>${esc(o.side)}</td>
      <td>${esc(o.order_type)}</td>
      <td class="n">${o.quantity}</td>
      <td class="n">${o.trigger_price != null ? num(o.trigger_price)
                      : (o.limit_price != null ? num(o.limit_price) : "—")}</td>
    </tr>`).join("");
  return `
    <h1 class="start-head">Paper book <span class="tag warn-tag">simulated</span></h1>
    <div class="panel"><table class="kv">
      <tr><td>Equity</td><td>${inr(b.equity_inr)}</td></tr>
      <tr><td>Cash</td><td>${inr(b.cash_inr)}</td></tr>
      <tr><td>Realised</td><td>${inr(b.realised_pnl_inr)}</td></tr>
      <tr><td>Since start</td><td class="${pnl >= 0 ? "up" : "down"}">${
        inr(pnl)} (${num(pct)}%)</td></tr>
    </table></div>
    ${rows ? `<div class="panel"><table class="screen-table">
      <thead><tr><th>SYMBOL</th><th>PROD</th><th class="n">QTY</th>
        <th class="n">AVG</th><th class="n">LAST</th><th class="n">P&amp;L</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`
      : `<p class="note">Flat — no open positions.</p>`}
    ${orders ? `<div class="panel"><table class="screen-table">
      <thead><tr><th>RESTING</th><th>SIDE</th><th>TYPE</th>
        <th class="n">QTY</th><th class="n">TRIGGER</th></tr></thead>
      <tbody>${orders}</tbody></table></div>` : ""}
    <p class="disclaimer">Simulated throughout: no order ever leaves this process,
      and the book lives on the server's disk, which a redeploy clears.</p>`;
}

// The bar under the header. Hidden markup rather than a built element, so
// its styling lives with the rest of the console's.
function showProgress(done, total, phase) {
  const bar = $("screen-progress");
  if (!bar) return;
  bar.hidden = false;
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  bar.querySelector("i").style.width = pct.toFixed(1) + "%";
  const label = phase === "fit" ? "fitting to the book" : "ranking on merit";
  let text = `${label} · ${done}/${total} runs`;
  if (done > 2 && state.startedAt) {
    const per = (Date.now() - state.startedAt) / 1000 / done;
    const left = Math.round(per * (total - done));
    if (left > 0) {
      text += ` · about ${left < 90 ? left + "s" : Math.round(left / 60) + " min"} left`;
    }
  }
  bar.querySelector("span").textContent = text;
}

function hideProgress() {
  const bar = $("screen-progress");
  if (bar) bar.hidden = true;
}

function paintOps() {
  const d = state.decision;
  if (!d) { $("ops").innerHTML = '<div class="empty">no run yet</div>'; return; }
  const rows = [
    ["last", d.symbol],
    ["action", d.final_action],
    ["size", d.position_size_pct ? num(d.position_size_pct) + "%" : "—"],
    ["at risk", inr(d.capital_at_risk_inr)],
  ];
  $("ops").innerHTML = rows.map(([k, v]) =>
    `<div class="ops-line"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}

/* ----------------------------------------------------------------- stage */

function show(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("on", t.dataset.tab === tab));
  renderStage();
}

function renderStage() {
  const el = $("stage");
  if (state.tab === "start") return void (el.innerHTML = startHtml());
  if (state.tab === "decision") return void (el.innerHTML = decisionHtml());
  if (state.tab === "screen") return void (el.innerHTML = screenHtml());
  if (state.tab === "book") return void (el.innerHTML = bookHtml());
  el.innerHTML = deskHtml();
}

function startHtml() {
  const cards = [
    ["Analyse one name", "RELIANCE.NS — four analysts, a bull/bear debate, then risk and the book.", "ask:RELIANCE.NS"],
    ["Find one for me", "Screen NIFTY50 and rank it. No symbol needed.", "screen:NIFTY50"],
    ["Screen a sector", "Rank Information Technology and see what fits the book.", "screen:Information Technology"],
    ["Let it trade (paper)", "Autopilot: screen NIFTYIT, place the orders, manage the book. Simulated.", "autopilot:NIFTYIT"],
    ["Re-decide a past session", "Set an as-of date below, then name a stock. Only what was knowable then.", "ask:TCS.NS"],
  ];
  return `
    <h1 class="start-head">What should we look at?</h1>
    <p class="start-sub">Ten agents: four analysts, a bull and a bear who argue over the
      same briefing, a research manager who scores the argument, a trader who writes the
      order, and a risk manager and portfolio manager who can only tighten it. The
      rulebook is plain Python and outranks the model.</p>
    <div class="cards">
      ${cards.map(([h, p, act]) => `
        <button class="card" data-act="${esc(act)}">
          <h4>${esc(h)}</h4><p>${esc(p)}</p>
        </button>`).join("")}
    </div>
    <p class="disclaimer">Research output, not investment advice. Verify against the
      exchange and consult a SEBI-registered adviser.</p>`;
}

function decisionHtml() {
  const d = state.decision;
  if (!d) return `<p class="note">No decision yet. Name a stock in the thread.</p>`;
  const p = d.proposal || {};
  const z = p.entry_zone || {};
  const costs = p.costs || {};
  const plan = d.investment_plan || {};
  const pm = d.portfolio_decision || {};
  const risk = d.risk_assessment || {};
  const uniq = (xs) => Array.from(new Set((xs || []).map(String)));
  const list = (t, xs) => uniq(xs).length
    ? `<div class="panel"><h3>${t}</h3><ul>${uniq(xs).map((i) =>
        `<li>${esc(i)}</li>`).join("")}</ul></div>` : "";

  const rows = [];
  const put = (k, v) => rows.push(`<tr><td>${k}</td><td>${v}</td></tr>`);
  if (p.instrument) put("Instrument",
    `${esc(p.instrument)} / ${esc(p.product_type || "")} / ${esc(p.time_horizon || "")}`);
  if (z.low != null) put("Entry band", `${num(z.low)} – ${num(z.high)}`);
  if (p.stop_loss != null) put("Stop-loss", num(p.stop_loss));
  if (p.targets && p.targets.length) put("Targets",
    p.targets.map((t) => `${num(t.price)} (${num(t.book_pct, 0)}%)`).join("<br>"));
  put("Position size", num(d.position_size_pct) + "%");
  if (p.quantity != null) put("Quantity", num(p.quantity, 0));
  put("Capital at risk", inr(d.capital_at_risk_inr));
  if (p.reward_risk != null) put("Reward : risk", num(p.reward_risk) + " : 1");
  if (costs.total_inr != null) put("Charges", inr(costs.total_inr));

  const signals = Object.entries(d.analyst_reports || {}).map(([k, v]) => {
    const sig = (v && (v.signal || v.verdict || v.stance)) || "";
    const c = v && v.confidence;
    return `<tr><td>${esc(k)}</td><td>${esc(sig)}${
      c != null ? ` (${num(c * 100, 0)}%)` : ""}</td></tr>`;
  }).join("");

  return `
    <div class="verdict-row">
      <span class="action ${esc(d.final_action)}">${esc(d.final_action)}</span>
      <span class="pill ${d.approved ? "yes" : "no"}">${d.approved ? "approved" : "rejected"}</span>
      ${$("asof").value ? `<span class="pill dated">as of ${esc($("asof").value)}</span>` : ""}
      <span class="meta">${num(d.confidence * 100, 0)}% confidence ·
        ${esc(d.symbol)}${d.sector ? " · " + esc(d.sector) : ""}</span>
    </div>
    <div class="panel"><table class="kv">${rows.join("")}</table></div>
    ${signals ? `<div class="panel"><h3>Analyst signals</h3>
      <table class="kv">${signals}</table></div>` : ""}
    ${plan.stance ? `<div class="panel"><h3>Research — bull vs bear</h3>
      <p><strong>${esc(plan.stance)}</strong>${plan.conviction != null
        ? ` · conviction ${num(plan.conviction * 100, 0)}%` : ""}${
        plan.winning_side ? ` · won by ${esc(plan.winning_side)}` : ""}</p>
      ${plan.verdict ? `<p>${esc(plan.verdict)}</p>` : ""}</div>` : ""}
    ${pm.portfolio_reasoning ? `<div class="panel"><h3>Portfolio manager</h3>
      <p>${esc(pm.portfolio_reasoning)}</p>
      ${uniq(pm.conditions).length ? `<ul>${uniq(pm.conditions).map((c) =>
        `<li>${esc(c)}</li>`).join("")}</ul>` : ""}</div>` : ""}
    ${risk.sizing_rationale ? `<div class="panel"><h3>Risk manager${
      risk.verdict ? " — " + esc(risk.verdict) : ""}</h3>
      <p>${esc(risk.sizing_rationale)}</p></div>` : ""}
    ${list("Execution notes", p.execution_notes)}
    ${list("Risk notes", d.risk_notes)}
    ${list("What to watch", plan.key_monitorables)}
    <p class="disclaimer">run ${esc(d.run_id)}${d.elapsed_seconds != null
      ? ` · ${num(d.elapsed_seconds, 1)}s` : ""}${d.as_of ? ` · ${esc(d.as_of)}` : ""}
      <br>Research output, not investment advice.</p>`;
}

function screenHtml() {
  const s = state.screen;
  if (!s) return `<p class="note">No screen yet. Try “screen NIFTY50”.</p>`;
  const rows = (s.candidates || []).slice(0, 40).map((c) => `
    <tr class="${c.selected ? "taken" : ""}">
      <td>${c.selected ? '<span class="mark">✓</span>' : ""}</td>
      <td>${esc(c.symbol)}</td>
      <td>${esc((c.sector || "").slice(0, 18))}</td>
      <td>${esc(c.action)}</td>
      <td class="num">${c.reward_risk != null ? num(c.reward_risk) : "—"}</td>
      <td class="num">${c.conviction != null ? num(c.conviction) : "—"}</td>
      <td class="num">${num(c.selected ? c.final_size_pct : c.size_pct)}</td>
      <td>${esc(c.selected ? "take" : (c.blocked_by || ""))}</td>
    </tr>`).join("");

  return `
    <div class="verdict-row">
      <span class="action" style="font-size:21px">${esc(s.universe)}</span>
      <span class="meta">${s.scanned} scanned · ${s.passed_standalone} on merit ·
        ${s.selected} fit the book</span>
    </div>
    <div class="panel">
      <table class="kv">
        <tr><td>Delivery (needs cash)</td><td>${num(s.delivery_pct)}%</td></tr>
        <tr><td>Intraday (leveraged)</td><td>${num(s.intraday_pct)}%</td></tr>
        <tr><td>Book risk</td><td>${inr(s.book_risk_inr)}</td></tr>
        <tr><td>Sectors</td><td>${esc((s.sectors || []).join(", ") || "—")}</td></tr>
      </table>
    </div>
    <div class="panel">
      <table class="screen-table">
        <thead><tr><th></th><th>symbol</th><th>sector</th><th>act</th>
          <th style="text-align:right">R:R</th><th style="text-align:right">conv</th>
          <th style="text-align:right">size%</th><th>note</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="disclaimer">Ranked on merit against a flat book, then re-run in rank order
      with the book accumulating — so the caps bind against the names worth keeping.</p>`;
}

function deskHtml() {
  if (!state.agents.length) return `<p class="note">Roster unavailable.</p>`;
  const row = (a) => `<tr><td>${esc(a.name)}</td><td>${esc(a.tier)}</td>
    <td>${esc(a.model)}</td></tr>`;
  return `
    <div class="panel"><h3>The desk</h3>
      <table class="kv"><tr><td><strong>agent</strong></td><td><strong>tier</strong></td>
        <td style="text-align:right"><strong>model</strong></td></tr>
        ${state.agents.map(row).join("")}</table>
    </div>
    <div class="panel"><h3>How a run goes</h3>
      <p>Each agent computes a complete deterministic draft before any model is called,
        so the desk still answers with no key and no network, and a provider outage
        degrades rather than fails.</p>
      <p>The bull and bear argue over byte-identical briefings, and the research manager
        scores argument quality rather than counting points. The trader writes an order;
        the risk manager and portfolio manager can only tighten it. Where the model and
        the rulebook disagree, code clamps the answer back inside the rules.</p>
    </div>`;
}

/* ---------------------------------------------------------------- wiring */

$("ask").onsubmit = (e) => {
  e.preventDefault();
  const raw = $("q").value.trim();
  if (!raw || state.running) return;
  $("q").value = "";
  const autoMatch = raw.match(/^\s*autopilot\s+(.+)$/i);
  if (autoMatch) return void runAutopilot(autoMatch[1].trim());
  const screenMatch = raw.match(/^\s*screen\s+(.+)$/i);
  if (screenMatch) runScreen(screenMatch[1].trim());
  else analyse(raw.toUpperCase());
};

document.addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab) return show(tab.dataset.tab);

  const card = e.target.closest(".card");
  if (card && !state.running) {
    const [kind, arg] = card.dataset.act.split(/:(.+)/);
    if (kind === "screen") runScreen(arg);
    else if (kind === "autopilot") runAutopilot(arg);
    else analyse(arg);
  }
});

$("btn-reset").onclick = () => {
  $("thread").innerHTML = "";
  say("Desk", "Cleared. Name a stock, or ask me to find one.");
  resetAgents();
  state.decision = null; state.screen = null;
  paintOps(); show("start");
};

boot();
requestAnimationFrame(drawCore);
