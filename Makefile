.DEFAULT_GOAL := help

.PHONY: help setup fmt lint typecheck test check web-check clean gen-types gen-docs web-setup web-dev evaluate train run

SAMPLES ?= 3000
SEED ?= 42
HOST ?= 0.0.0.0
PORT ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python dependencies from the lockfile (npm ci --prefix web installs web dependencies)
	uv sync

fmt:  ## Format and auto-fix Python sources
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and linting without modifying files
	uv run ruff format --check .
	uv run ruff check .

typecheck:  ## Run the static type checker
	uv run mypy

test:  ## Run the test suite
	uv run pytest

check: lint typecheck test web-check  ## Run every verification; must be green before committing

web-check:  ## Run the web lint, typecheck, tests and production build
	@if [ ! -d web/node_modules ]; then npm ci --prefix web; fi
	npm run lint --prefix web
	npm run typecheck --prefix web
	npm test --prefix web
	npm run build --prefix web

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

gen-types:  ## Generate TypeScript schema types from the canonical schema
	uv run python tools/gen_web_types.py

gen-docs:  ## Generate the README indicator table from the canonical schema
	uv run python tools/gen_readme_table.py

web-setup:  ## Install web dependencies from the lockfile
	npm ci --prefix web

web-dev:  ## Run the Vite development server
	npm run dev --prefix web

# Evaluate the synthetic classifier and regenerate reports/evaluation.json and reports/evaluation.md.
evaluate:  ## Measure classifier and baseline performance on synthetic data
	uv run python -c 'from aburrimiento.evaluate import evaluate, write_report; from pathlib import Path; write_report(evaluate(n_samples=$(SAMPLES), seed=$(SEED)), Path("reports"))'

run:  ## Run the API with reload on the configured host and port
	uv run uvicorn aburrimiento.api:app --reload --host $(HOST) --port $(PORT)

train:  ## Train and save the model artifact, then regenerate matching evaluation reports
	$(MAKE) evaluate SAMPLES=$(SAMPLES) SEED=$(SEED)
	uv run python -c 'from aburrimiento.artifacts import main; main()' --samples $(SAMPLES) --seed $(SEED)
