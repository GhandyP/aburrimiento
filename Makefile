.DEFAULT_GOAL := help

.PHONY: help setup fmt lint typecheck test check clean gen-types evaluate

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python dependencies from the lockfile
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

check: lint typecheck test  ## Run every verification; must be green before committing

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

gen-types:  ## Generate TypeScript schema types from the canonical schema
	uv run python tools/gen_web_types.py

# Evaluate the synthetic classifier and regenerate reports/evaluation.json and reports/evaluation.md.
evaluate:  ## Measure classifier and baseline performance on synthetic data
	uv run python -m aburrimiento.evaluate
