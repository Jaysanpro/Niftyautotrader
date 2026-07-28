# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# lxml and pandas wheels cover most of it; build-essential is only needed if a
# wheel is missing for the target platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY tradingagents_in/__init__.py tradingagents_in/__init__.py
RUN pip install --upgrade pip && pip install -e ".[graph,data,llm]"

COPY tradingagents_in/ tradingagents_in/
COPY examples/ examples/
COPY tests/ tests/

# Results and the decision log are the two things worth mounting out.
RUN mkdir -p /app/results /app/.cache

# A non-root user, because nothing here needs root.
RUN useradd --create-home --uid 1000 ita && chown -R ita:ita /app
USER ita

# The container runs one CLI command and exits, so there is no long-lived
# process to health-check.
ENTRYPOINT ["ita"]
CMD ["--help"]
