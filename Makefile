.PHONY: help install check test lint fmt typecheck sample demo docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with every extra
	pip install -e ".[graph,data,llm,dev]"

check:  ## Everything before a push: lint, suite, a real run, the examples
	@echo "==> lint"
	ruff check tradingagents_in/ tests/ examples/ web/
	@echo "==> suite"
	python -m pytest -q
	@echo "==> deterministic end-to-end run"
	ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
	  python -m tradingagents_in.cli.main analyse RELIANCE.NS --no-save
	@echo "==> sample-run generator"
	python examples/generate_sample_run.py
	@echo "==> examples"
	ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
	  python examples/fno_index_trade.py NIFTY
	ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
	  python examples/python_api.py
	@echo "==> all green"

test:  ## Run the suite (offline, synthetic, no network)
	python -m pytest -q

lint:  ## Lint with ruff
	ruff check tradingagents_in/ tests/ examples/ web/

fmt:  ## Auto-fix what ruff can
	ruff check --fix tradingagents_in/ tests/ examples/ web/

typecheck:  ## Type-check with mypy
	mypy tradingagents_in/

sample:  ## Regenerate docs/sample_run.md
	python examples/generate_sample_run.py

demo:  ## One deterministic run, no key and no network needed
	ITA_LLM_PROVIDER=offline ITA_DATA_PROVIDER=synthetic ITA_ONLINE_TOOLS=false \
	  python -m tradingagents_in.cli.main analyse RELIANCE.NS --no-save

docker:  ## Build the image
	docker build -t tradingagents-in .

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info .coverage htmlcov
