export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test test-data test-ci cov data calibrate summarize-eval

setup:
	uv sync --extra dev
	uv run pre-commit install

data:
	uv sync --extra dev --extra data
	uv run ticketintel data

lint:
	uv run ruff check src tests benchmarks

format:
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not ml and not data"

test-data:
	uv sync --extra dev --extra data
	uv run pytest -m "data"

cov:
	uv run pytest -m "not ml" --cov=ticketintel --cov-report=term-missing

calibrate:
	uv sync --extra dev --extra data
	uv run ticketintel calibrate

summarize-eval:
	uv sync --extra dev --extra ml
	uv run ticketintel summarize-eval

# Reproduces the light CI job exactly: dev dependencies only, so a module that
# type-checks locally because the heavy extras are installed still fails here.
test-ci:
	uv run --isolated --extra dev mypy
	uv run --isolated --extra dev ruff check src tests benchmarks
	uv run --isolated --extra dev ruff format --check src tests benchmarks
	uv run --isolated --extra dev pytest -m "not ml and not data"
