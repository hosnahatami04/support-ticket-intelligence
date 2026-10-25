export PYTHONHASHSEED := 0

.PHONY: setup lint format typecheck test cov

setup:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest -m "not ml"

cov:
	uv run pytest -m "not ml" --cov=ticketintel --cov-report=term-missing
