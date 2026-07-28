# IndianTradingAgents (TradingAgents-IN)

A multi-agent LLM research framework for Indian equity markets — NSE and BSE,
cash and F&O, everything in INR and Asia/Kolkata.

Eight specialised agents analyse a scrip, argue about it, propose an order, size
it against a deterministic risk rulebook, and either approve it or explain why
the desk is staying flat. Orchestrated with LangGraph; every agent output is a
validated Pydantic model.

> **This is research software, not investment advice.** It produces a documented
> opinion with its reasoning attached. Verify every number against the exchange
> before acting, and consult a SEBI-registered investment adviser. See
> [Limitations](#limitations) — they are specific and worth reading.

---

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [The agents](#the-agents)
- [India-specific behaviour](#india-specific-behaviour)
- [Configuration](#configuration)
- [LLM providers](#llm-providers)
- [Data sources](#data-sources)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Decision log](#decision-log)
- [Docker](#docker)
- [Development](#development)
- [Limitations](#limitations)

---

## What it does

```
                    ┌──────────── one data fetch, shared by everyone ──────────┐
                    │                                                          │
Fundamentals ──┐    │  Ind-AS financials, promoter holding, pledge, ROCE, peers│
Sentiment ─────┤    │  ET · Moneycontrol · Business Standard · Mint · X · Reddit│
News & Macro ──┼────┤  RBI · Union Budget · FII/DII flows · crude · USDINR · US │
Technical ─────┘    │  RSI · MACD · Bollinger · Supertrend · VWAP · CPR · ADX   │
                    └──────────────────────────────────────────────────────────┘
                                             │
                         Bull ⇄ Bear structured debate (N rounds)
                                             │
                           Research Manager adjudicates → stance
                                             │
                   Trader → entry band · stop · targets · size · instrument
                                             │
                Risk Manager + deterministic SEBI/exchange rulebook
                                             │
                     Portfolio Manager → approve / reject for the book
```

The output is a `FinalDecision`: an action, a confidence, a position size, the
capital at risk, the full debate transcript, every rule that was checked, and a
markdown report you can read end to end. A worked example is in
[`docs/sample_run.md`](docs/sample_run.md).

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/sanjayyaluguli-lang/niftyautotrader
cd niftyautotrader

python -m venv .venv && source .venv/bin/activate
pip install -e ".[graph,data,llm]"
```

The extras are separable on purpose:

| Extra | Brings in | Needed for |
|---|---|---|
| `graph` | langgraph, langchain-core | The LangGraph backend. Without it a built-in sequential executor runs the identical pipeline. |
| `data` | yfinance, feedparser, beautifulsoup4, lxml | Live NSE/BSE prices, RSS news, HTML sources. |
| `llm` | langchain-{openai,anthropic,google-genai,groq,ollama} | Any LLM provider. Without it, the offline engine still runs. |
| `brokers` | smartapi-python, kiteconnect, pyotp | Angel One / Zerodha feeds. |
| `dev` | pytest, ruff, mypy | Tests and linting. |

Then configure:

```bash
cp .env.example .env
$EDITOR .env          # set ITA_LLM_PROVIDER and one API key
```

## Quick start

```bash
# No API key, no network - the deterministic engine on a seeded synthetic tape.
ita analyse RELIANCE.NS --offline

# The real thing
ita analyse RELIANCE.NS --capital 500000 --depth 3

# Full report rather than the summary panel
ita analyse HDFCBANK.NS --report

# BSE, via the .BO suffix
ita analyse TCS.BO

# An index - the trader will pick futures or an option spread
ita analyse BANKNIFTY --capital 5000000

# Several names, with the book accumulating across the batch
ita batch RELIANCE.NS TCS.NS HDFCBANK.NS INFY.NS ITC.NS

# Where are we in the session?
ita session
```

From Python:

```python
from tradingagents_in import IndianTradingAgents

app = IndianTradingAgents(config={"capital_inr": 500_000})
decision = app.analyse("RELIANCE.NS")

print(decision.headline())
# RELIANCE: BUY [APPROVED] size=9.93% conf=67%

print(app.report(decision))     # the full markdown report
```

## The agents

### Analyst team

| Agent | What it actually looks at |
|---|---|
| **Fundamentals** | Ind-AS consolidated financials, quarterly results with YoY, promoter holding and pledge, D/E, interest coverage, ROE/ROCE, OCF/PAT cash conversion, FCF, 3Y CAGRs, peer table with a P/E premium or discount to the sector median, governance flags. |
| **Sentiment** | Headline sentiment across Economic Times, Moneycontrol, Business Standard, LiveMint and BusinessLine, plus StockTwits, Reddit r/IndiaInvestments and X where credentials are present. Reports a score, a buzz level, the retail-versus-institutional split, and flags crowding that looks like a fade. |
| **News & Macro** | RBI policy and repo trajectory, Union Budget and GST, monsoon versus LPA, FII/DII cash-segment flows with a 5-session read, crude, USDINR, US and China cues, India VIX, and a dated event calendar. |
| **Technical** | RSI, MACD, Bollinger with %B, Supertrend, VWAP, ATR, ADX with ±DI, EMA 20/50/200, OBV, Donchian, classic pivots and CPR width, swing levels, volume ratio — across daily, hourly and 15-minute frames, with an explicit multi-timeframe alignment call and relative strength versus the Nifty. |

Every analyst states its stance, what would change its mind, and a confidence
that is haircut by whatever data failed to arrive.

### Research team

The **Bull** and **Bear** researchers receive byte-identical briefings — any
difference in their conclusions comes from their mandate, not from being shown
different numbers. Each round they must answer the opponent's strongest specific
point, concede at least one thing that genuinely holds, and add an argument they
have not made before. Conceding nothing across a whole debate is treated as a
tell, and the research manager discounts that side for it.

Both must ground part of the case in India mechanics: SEBI rules, circuit
filters, ASM/GSM, F&O ban periods, expiry pinning, delivery versus intraday,
STT and stamp duty, promoter pledging, FII/DII regime.

The **Research Manager** scores argument quality — evidence strength, genuine
rebuttal, honest concession — rather than counting points, then commits to a
stance. Where the scrip is not tradable, it says so and stops there rather than
passing the problem down the chain.

### Trader

Produces an order a dealer could work: an entry band with a trigger condition, a
stop at a structural level, staged targets with a book percentage each, a size
derived from the stop distance, a horizon, and an instrument chosen for the
horizon — cash/CNC by default, futures for a carry-forward short, a debit
spread when the view has a date attached.

### Risk Manager and Portfolio Manager

The **Risk Manager** runs the deterministic rulebook (below), then judges what
the rulebook cannot: whether the stop is *fillable*, whether the modelled loss
is the real loss given gap risk, whether the book is already long the same
factor, and whether the edge survives charges. It produces a rupee stress test
across gap and circuit scenarios.

The **Portfolio Manager** decides whether the position belongs in *this* book —
marginal contribution, simultaneous loss, capacity to monitor, opportunity cost.
It can only be more conservative than the risk manager, and code enforces that.

## India-specific behaviour

This is the part that makes it an Indian framework rather than a US one with the
tickers changed.

**Session and calendar.** 09:00–09:15 pre-open, 09:15–15:30 continuous,
15:40–16:00 post-close, all IST. Published equity trading holidays for
2024–2026. Weekly expiry on Thursday for NSE and Tuesday for BSE, per the SEBI
one-weekly-expiry-per-exchange rule, rolled back off holidays. Monthly expiry as
the last weekly weekday of the month. Running past the loaded calendar surfaces
as a data-quality warning rather than a silently wrong expiry.

**Symbols.** `RELIANCE`, `reliance.ns`, `NSE:RELIANCE`, `TCS.BO`, `BSE:TCS`,
six-digit BSE scrip codes, `NIFTY`, `BANKNIFTY`, `SENSEX`, `^NSEI`, `FINNIFTY`,
`NIFTYIT`, `NIFTYPHARMA` all resolve. Explicit suffixes beat the default
exchange. Index names map to their contract codes — the index is NIFTY 50, the
contract is NIFTY, and lot sizes and strike ladders key off the latter.

**Circuit filters.** Bands are inferred per scrip, upper and lower levels
computed, and a stop that sits outside the band is a **BLOCKER**: the scrip
would lock limit-down before the stop could fill, so the real risk is unbounded
across sessions rather than the modelled amount.

**Surveillance.** GSM blocks. ASM warns and cuts size — higher margins,
possible trade-for-trade settlement, a thinner book.

**F&O ban period.** Blocks new derivative positions; only position-reducing
orders are permitted and new ones attract an exchange penalty. Cash is still
allowed, and the framework says so rather than blocking the name outright.

**Product legality.** A cash-segment short is intraday-only and is forced to
MIS; proposing CNC on a short is impossible by construction. Overnight F&O is
gated by mandate. Stock derivatives are monthly-only; only index derivatives
get weeklies. Stock F&O carried into expiry is flagged for physical settlement.

**Costs.** STT (different rates for delivery buy/sell, intraday sell, futures
sell, options sell on premium), exchange transaction charges (separate rates for
cash, futures and options), SEBI turnover fee, stamp duty on the buy side only,
and GST at 18% on brokerage plus exchange plus SEBI charges — but *not* on STT
or stamp duty. Reward-risk is evaluated **net of all of it**, because a 1:1.5
setup on a tight stop is negative after charges, and that is the difference
between a strategy and a donation.

**Corporate actions.** Bonus, split, rights, dividend and demerger are fetched;
a price-adjusting action inside the holding period raises a warning that entry,
stop and targets need re-basing, and that F&O lot sizes adjust on the ex-date.

**Flows.** FII and DII cash-segment net figures with a 5-session read and an
interpretation, because the FII-selling-absorbed-by-DII-SIP regime narrows
breadth even while the index holds.

**Sizing.** Lots are indivisible. If one lot risks more than the entire budget,
the answer is zero lots with the arithmetic shown — not a rounded-down position
that quietly triples the intended risk.

## Configuration

Three layers, each overriding the last:

```
DEFAULT_CONFIG  <  environment (ITA_*)  <  explicit overrides
```

```python
from tradingagents_in import IndianTradingAgents

app = IndianTradingAgents(config={
    "capital_inr": 2_500_000,
    "profile": "institutional",       # tighter limits, more names
    "risk_per_trade_pct": 0.5,
    "max_debate_rounds": 3,           # research depth
    "selected_analysts": ["fundamentals", "technical", "news_macro"],
    "agent_model_tier": {"trader": "deep"},   # nested dicts merge, not replace
})
```

Pass only the keys you want to change. Spreading `**DEFAULT_CONFIG` into your
overrides makes every key explicit and silently disables your `.env`.

### Keys that matter

| Key | Default | Meaning |
|---|---|---|
| `capital_inr` | 1,000,000 | Total trading capital |
| `profile` | `retail` | `retail` or `institutional` — sets a bundle of limits |
| `risk_per_trade_pct` | 1.0 | Loss at the stop, as % of capital. Drives all sizing. |
| `max_position_pct` | 10.0 | Single-name exposure cap |
| `max_sector_pct` | 25.0 | Sector exposure cap |
| `max_portfolio_heat_pct` | 6.0 | Sum of open risk across the book |
| `max_drawdown_pct` | 15.0 | New risk is suspended at this drawdown |
| `min_reward_risk` | 1.5 | Minimum R:R **after** Indian charges |
| `max_debate_rounds` | 2 | Bull/bear rounds |
| `max_risk_rounds` | 1 | Risk reviews. Set to 2+ to enable the trader revision loop. |
| `allow_fno` / `allow_overnight_fno` / `allow_short_cash` | true/true/false | Instrument permissions |
| `allow_expiry_day_entry` | false | New derivative positions on expiry day |
| `avoid_asm_gsm` / `avoid_fno_ban_period` | true/true | Surveillance gates |
| `min_avg_daily_turnover_inr` | ₹5 crore | Liquidity floor |
| `agent_model_tier` | see below | Per-agent `deep` or `quick` model |
| `data_provider` | `yfinance` | `yfinance`, `angelone`, `zerodha`, `synthetic` |

The `institutional` profile tightens risk per trade to 0.5%, single-name to 5%,
sector to 20%, heat to 4%, and the liquidity floor to ₹25 crore, while raising
the position count to 40.

### Model tiers

Analysts run on the quick model, everything from the researchers onward on the
deep model. Override per agent:

```python
"agent_model_tier": {
    "fundamentals_analyst": "deep",   # deep reading of the filings
    "sentiment_analyst": "quick",
    "trader": "deep",
}
```

## LLM providers

Supported: **OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, Groq, OpenRouter,
Ollama** (local), and **offline**.

```bash
ita providers      # which ones have credentials present
```

```bash
ITA_LLM_PROVIDER=anthropic
ITA_DEEP_THINK_MODEL=claude-sonnet-4-5
ITA_QUICK_THINK_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Providers with an OpenAI-compatible endpoint route through `langchain-openai`
with a `base_url` override, so adding another is a single table entry in
`llm/provider.py`.

**`offline` is a real mode, not a stub.** Every agent computes a complete
deterministic draft from the data before any model is called; offline returns
that draft. You get a fully sized, risk-checked decision with no API key. It is
also what the LLM path falls back to when a provider rate-limits, times out or
returns unparseable JSON — so a provider outage degrades the run instead of
ending it.

## Data sources

| Source | Used for | Requires |
|---|---|---|
| yfinance (`.NS` / `.BO`) | OHLCV across timeframes, quotes, fundamentals, corporate actions | `[data]` extra |
| NSE India | Option chains, F&O ban list, surveillance, announcements | — |
| Screener.in | Supplementary fundamentals | `ITA_ENABLE_SCREENER_SCRAPE=true`, respect their ToS |
| RSS: ET, Moneycontrol, Business Standard, Mint, BusinessLine | News and headline sentiment | `[data]` extra |
| StockTwits / Reddit / X | Social sentiment | Optional keys; degrades silently without them |
| Angel One SmartAPI, Zerodha Kite | Live quotes | `[brokers]` extra + credentials |
| `synthetic` | Deterministic seeded tape | — |

Sources fail independently. A failure is recorded, named in the report, and cuts
the confidence of the agents that needed it — it never aborts the run.

## CLI reference

```
ita analyse SYMBOL [--report] [--json]     Analyse one symbol
ita batch SYM [SYM ...] [--independent]    Several symbols, book accumulating
ita session                                Session phase, expiry calendar
ita providers                              LLM providers and credential status
ita config                                 Resolved configuration, changes highlighted
ita graph                                  Agent topology as Mermaid
ita universe [--index NIFTY_IT]            Indices, sectors, lot sizes
ita memory [--stats] [--symbol SYM]        Inspect the decision log
ita outcome RUN_ID --result win --pnl N    Record how a decision turned out
```

Common flags: `--provider`, `--deep-model`, `--quick-model`, `--capital`,
`--profile`, `--risk-per-trade`, `--max-position`, `--depth`, `--analysts`,
`--offline`, `--no-fno`, `--no-save`, `--sequential`, `--json`, `--log-level`.

## Python API

```python
from tradingagents_in import IndianTradingAgents
from tradingagents_in.models.risk import PortfolioSnapshot

# Tell it what you actually hold, or every trade is judged against an empty book
book = PortfolioSnapshot(
    capital_inr=2_500_000,
    deployed_inr=800_000,
    cash_inr=1_700_000,
    open_positions=3,
    portfolio_heat_pct=1.8,
    sector_exposure_pct={"Information Technology": 14.0, "Banking": 18.0},
)

app = IndianTradingAgents(config={"capital_inr": 2_500_000}, portfolio=book)
decision = app.analyse("TCS.NS")

decision.final_action            # Action.BUY
decision.approved                # bool
decision.confidence              # 0.0 - 1.0
decision.position_size_pct       # % of capital
decision.capital_at_risk_inr     # loss at the stop
decision.proposal.entry_zone     # EntryZone(low, high, trigger)
decision.proposal.targets        # [Target(price, book_pct, rationale), ...]
decision.risk_assessment         # verdict, breaches, stress test
decision.debate_transcript       # the full bull/bear exchange
```

More in [`examples/`](examples/): a single-stock run, an index screen with the
book accumulating, an F&O index trade, and a full API walkthrough.

## Decision log

Every run appends to `trading_memory.md` (human-readable) and
`trading_memory.jsonl` (machine-readable). Later runs recall comparable past
decisions — same symbol first, then same sector, then any closed trade — and
inject them into the researcher, trader and PM prompts.

Recall is deliberately keyword-and-recency based rather than embedding-based: a
desk that cannot say *why* a past decision was surfaced cannot audit the advice
it produced.

Close the loop when you know how it went:

```bash
ita outcome RELIANCE_NS-a1b2c3d4 --result win --pnl 18500 \
    --lesson "The 1.5R first target paid for the trade before the expiry chop."
```

```bash
ita memory --stats     # hit rate, net P&L, symbols covered
```

A sample log is in [`docs/trading_memory.sample.md`](docs/trading_memory.sample.md).

## Docker

```bash
docker build -t tradingagents-in .
docker run --rm --env-file .env tradingagents-in analyse RELIANCE.NS

# Persist results and the decision log
docker run --rm --env-file .env \
    -v "$PWD/results:/app/results" \
    -v "$PWD/trading_memory.md:/app/trading_memory.md" \
    tradingagents-in analyse TCS.NS --report
```

Or with compose:

```bash
docker compose run --rm ita analyse HDFCBANK.NS
docker compose run --rm test          # the suite, offline and deterministic
```

## Development

```bash
pip install -e ".[graph,data,llm,dev]"

make test        # pytest, offline + synthetic, no network
make lint        # ruff
make sample      # regenerate docs/sample_run.md
```

The suite runs against the synthetic provider with a fixed seed and the offline
engine: no API key, no network, deterministic. Architecture and design rationale
are in [`docs/architecture.md`](docs/architecture.md).

## Limitations

Stated plainly, because the failure modes are more useful than the features.

- **It has no edge.** It is a structured research pipeline, not an alpha model.
  Nothing here has been backtested, and the synthetic provider is a random walk
  precisely so nobody mistakes it for validation.
- **LLMs are wrong confidently.** The deterministic anchoring and the rulebook
  reduce the blast radius; they do not eliminate it. Check the numbers.
- **Free data is approximate.** yfinance fundamentals lag filings, report an
  insider-holding proxy rather than the SEBI promoter category, and carry no
  pledge data. Treat promoter and pledge figures as unverified unless a filing
  confirms them.
- **The holiday calendar is hardcoded** for 2024–2026 and needs an annual
  refresh from the NSE circular. Past that range, the framework warns rather
  than guessing silently.
- **Lot sizes and F&O eligibility change.** The bundled table is a snapshot;
  confirm against the current NSE contract file before trading.
- **Margins are estimates.** SPAN and exposure numbers here are approximations
  for affordability checks. Your broker's file is authoritative.
- **No execution.** Nothing here places an order. Broker integrations are
  read-only quote feeds.
- **Not a substitute for advice.** SEBI regulates investment advice in India.
  This is a tool for your own research.

## License

Apache-2.0.
