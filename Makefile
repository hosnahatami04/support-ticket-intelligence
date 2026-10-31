export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test test-data cov data

setup:
	uv sync --extra dev
	uv run pre-commit install

data:
	uv sync --extra dev --extra data
	uv run ticketintel data

lint:
	uv run ruff check src tests

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
