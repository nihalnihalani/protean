# Protean developer Makefile.
# CPU-only targets; GPU work (real grading) requires torch+triton via `.[gpu]`.

PYTHON ?= python3
VENV   ?= .venv
PY      = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test lint format typecheck redteam clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local virtualenv if missing.
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv ## Install the package with the test extra (CPU-only).
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[test]"

install-dev: venv ## Install package plus dev tooling (ruff, mypy, pytest-cov).
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test: ## Run the full test suite (CPU-only; CUDA paths skip).
	$(PY) -m pytest -q

lint: ## Lint and check formatting with ruff.
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-fix lint issues and format the codebase.
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck: ## Run mypy over the package source.
	$(PY) -m mypy src/protean

redteam: ## Verify obvious reward hacks fail closed (needs GPU for real grading).
	$(PY) scripts/check_redteam.py

clean: ## Remove caches and build artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
