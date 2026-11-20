# Deploying the web front end

`web/app.py` is a FastAPI service wrapping `IndianTradingAgents.analyse()`.
It is built by **`Dockerfile.web`**, not the root `Dockerfile` — that one
builds the run-once CLI image and will fail a health check if deployed.

Measured footprint, so you can size any host: **103 MiB idle, 136 MiB peak**
across six analyses, ~1.8 s first request (the graph is built lazily) and
~0.6 s warm, running the offline engine.

## Google Cloud Run

One-time setup. Replace `PROJECT_ID`:

```bash
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create ita \
    --repository-format=docker --location=asia-south1
```

Then, from the repository root, for the first deploy and every one after:

```bash
gcloud builds submit --config deploy/cloudbuild.yaml
```

The command prints the service URL when it finishes. `asia-south1` is Mumbai;
change `_REGION` in `deploy/cloudbuild.yaml` if you want it elsewhere, and
create the Artifact Registry repository in the same region.

### What the deploy settings are for

| Flag | Why |
| --- | --- |
| `--memory=512Mi` | Headroom over the measured 136 MiB peak. |
| `--concurrency=2` | Matches `ITA_WEB_CONCURRENCY`. The app admits two runs at a time and returns 503 beyond that, so a higher value would turn load into errors rather than new instances. |
| `--min-instances=0` | Scales to zero. An idle deployment costs nothing. |
| `--max-instances=3` | A cost guard. Without a ceiling, sustained traffic fans out indefinitely. |
| `--cpu-boost` | Cold start is where the agent graph gets built. |
| `--timeout=600` | Ample for the offline engine; needed once a real LLM is answering. |

### Cost

Cloud Run's always-free tier is 180,000 vCPU-seconds, 360,000 GiB-seconds and
2M requests per month. At ~0.6 s per warm request on 1 vCPU that is roughly
300,000 analyses a month before anything is billed, and scale-to-zero means an
idle service bills nothing. Verify the current figures at
<https://cloud.google.com/run/pricing> — they do change.

## Running it locally

```bash
pip install -e ".[graph,data,web]"
ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
    uvicorn web.app:app --reload
```

Then open <http://127.0.0.1:8000>.

## Going live with a real LLM

The deployment defaults to the offline engine, so it answers deterministically
with no API key and no per-request cost. To use a real provider, set the key as
a secret rather than an environment variable:

```bash
echo -n "sk-..." | gcloud secrets create ita-llm-key --data-file=-
gcloud run services update ita-web --region=asia-south1 \
    --set-env-vars=ITA_LLM_PROVIDER=anthropic,ITA_DATA_PROVIDER=yfinance,ITA_ONLINE_TOOLS=true \
    --set-secrets=ANTHROPIC_API_KEY=ita-llm-key:latest \
    --memory=1Gi
```

Raise `--memory` and `--timeout` at the same time: a live run holds real market
data and waits on the model, so both the footprint and the latency go up.

Two things worth keeping in mind before that switch. Every request then costs
real money at your provider, and `--max-instances` is the only thing bounding
how fast. And the output stops being deterministic, so a decision you see is
no longer reproducible from the inputs alone.

## Railway

`railway.json` is already in the repo and points the builder at
`Dockerfile.web`, so there is nothing to write:

1. railway.com: New Project, Deploy from GitHub, pick this repo.
2. Variables:

   ```
   ITA_LLM_PROVIDER=offline
   ITA_DATA_PROVIDER=synthetic
   ITA_ONLINE_TOOLS=false
   ITA_WEB_CONCURRENCY=6
   TZ=Asia/Kolkata
   ```

   Railway has no field for these in `railway.json`; they are set in the
   dashboard or with `railway variables`.
3. Settings, Networking, Generate Domain.

`$PORT` is injected and the image already binds it. The health check is
`/healthz`, which answers without building a runner or touching the network.

### Why concurrency 6 here and 2 on Cloud Run

They are limited by different things. Cloud Run is told `--concurrency=2` to
match the app's own semaphore so that load starts another instance instead of
returning 503. Railway serves from one instance, so the semaphore is the only
limiter and a low value would just 503 under light use.

Memory does not argue against raising it. Measured on this image: 103 MiB
idle, 135 MiB after one warm request, **141 MiB with six concurrent
analyses** - the runner cache is shared per (capital, profile), so extra
concurrency costs a few MiB rather than a multiple. The binding constraint is
CPU: each analysis is ~0.6s of it, so six at once on a small instance queue
rather than fail.

## If the console says ANTHROPIC_API_KEY not set

That chip means the deployment names a provider it has no key for. It is
worth calling out because **nothing else looks wrong**: every request still
succeeds, every agent still returns a verdict, and the decisions still read
normally. They are the deterministic drafts. No model has been asked
anything.

Set the key where your host keeps environment variables (Railway: the
project's Variables tab; Cloud Run: `--set-secrets`, as above) and redeploy.
The chip turns teal and reads `live · anthropic`.

If you would rather not spend anything per request, set
`ITA_LLM_PROVIDER=offline` instead. That is the same engine, chosen on
purpose rather than by accident, and the chip says so.

Note that the framework still works either way - the deterministic drafts
are computed in full before any model is called, and the pure-Python risk
rulebook outranks whatever a model says. The LLM adds reasoning and
narration; it does not add the decision.

## How long a screen takes

Worth knowing before you run one, because the honest answer is "minutes".

A screen is two passes of full pipeline runs: every constituent judged against
a flat book, then the best re-run in rank order with the book accumulating.
For NIFTY50 that is **52 + 12 = 64 complete runs**, strictly serial, because
the data sources are free public endpoints and parallelising them is the
fastest way to get rate-limited into a run with no data at all.

Measured on this image with the synthetic provider, which does no I/O at all:

| universe | names | runs | wall clock |
| --- | --- | --- | --- |
| NIFTYIT | 10 | 18 | ~6 s |
| NIFTY50 | 52 | 64 | ~21 s |

That is the floor - pure compute, 0.33 s a run. On `yfinance` each run also
makes about ten network fetches (prices over three timeframes, quote,
benchmark, fundamentals, screener, corporate actions, news, sentiment, macro,
microstructure), so a NIFTY50 screen is minutes rather than seconds. The
toolkit caches per symbol for `cache_ttl_seconds` (15 minutes by default),
which mostly spares the second pass from fetching again.

So `POST /api/screen` does not block for the result. It validates the
universe, starts the work on a thread and returns a run id at once:

```json
{"run_id": "R-...", "universe": "NIFTY50", "names": 52, "runs_expected": 64}
```

Progress then arrives on `GET /api/runs/{run_id}/stream` as server-sent
events, one per name, with that name's verdict:

```
{"type":"progress","phase":"merit","completed":7,"total":52,"symbol":"INFY.NS",
 "action":"BUY","approved":true,"reward_risk":1.48}
```

and the finished screen as a single `{"type":"screen", ...}` event. The
console draws that as a progress bar with a running estimate.

Two reasons it is built this way rather than served synchronously. A request
held open for minutes with nothing on the wire is indistinguishable from a
hang, and gets cut by any proxy with an idle timeout. And a screen that says
"7/52, INFY BUY at R:R 1.48" is telling you something useful while you wait.

If you only want a quick look, screen a sector rather than an index -
`screen Information Technology` is ten names, not fifty-two.

## Where the console comes from

`Dockerfile.web` ships `frontend/`, and the app serves it at `/`. So a Railway
or Cloud Run deployment gives you the console on its own URL, same-origin,
with no CORS to configure and no second service to deploy. That is the
simplest setup and the one to start from.

If `/` ever answers "API is up, console is missing", the image was built
without `frontend/` - check the `COPY frontend/ frontend/` line survived.

## The Vercel dashboard (optional)

Only worth it if you want the page on a CDN separately from the API. The
backend already serves the same files.

`frontend/` is a static dashboard for the same API: no build step, no
framework, so it deploys as files and there is no toolchain to break.

1. Put the backend's URL into `frontend/vercel.json`, replacing
   `REPLACE-ME.a.run.app` in the rewrite. Either host works - a Cloud Run
   `*.a.run.app` or a Railway `*.up.railway.app` - since the rewrite only
   needs somewhere to forward `/api` to.
2. On vercel.com: New Project, import this repo, set **Root Directory** to
   `frontend`, framework preset **Other**. No build command.

The rewrite proxies `/api/*` from the Vercel domain to Cloud Run, so the
browser only ever talks to its own origin and **there is no CORS to
configure**. That is the whole reason it is set up this way.

To call a backend directly instead - a different deployment, or a local one -
open Settings in the dashboard and set the API base. That *is* cross-origin,
so the backend has to name the Vercel domain:

```bash
gcloud run services update ita-web --region=asia-south1 \
    --set-env-vars=ITA_WEB_CORS_ORIGINS=https://your-project.vercel.app
```

Several origins are comma-separated; `ITA_WEB_CORS_ORIGIN_REGEX` covers
preview deployments. Unset by default, so nothing is opened up unasked, and
`*` is deliberately not supported: any page on the internet could then spend
this deployment's analysis budget.

The dashboard also exposes the **as-of** date, so a past session can be
re-decided from the browser using only what was knowable by that close.

## Other hosts

`Dockerfile.web` is a plain image with no Cloud Run specifics, so it runs
anywhere that takes a container. It binds `0.0.0.0` on `$PORT`, which is what
Cloud Run, Koyeb, Render and Fly all inject. Only `railway.json` is
vendor-specific, and every other platform ignores it.
