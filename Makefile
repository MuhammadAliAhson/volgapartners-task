.PHONY: install format lint typecheck test check run-api

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[api,dev]"

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

check: lint typecheck test

run-api:
	$(PYTHON) -m uvicorn transcriber.api.app:app --host 0.0.0.0 --port 8000
