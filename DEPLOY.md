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

## The Vercel dashboard

`frontend/` is a static dashboard for the same API: no build step, no
framework, so it deploys as files and there is no toolchain to break.

1. Put the Cloud Run URL into `frontend/vercel.json`, replacing
   `REPLACE-ME.a.run.app` in the rewrite.
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
