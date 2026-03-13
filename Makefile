.PHONY: setup sync lint format test

setup:
	git config core.hooksPath .githooks
	uv sync

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	cd app/backend && uv run pytest tests/ -v
