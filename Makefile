.PHONY: setup sync lint format test run download-weights

setup:
	git config core.hooksPath .githooks
	uv sync
	doppler setup --project cannabis-maturity-app --config dev --scope app/api

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	cd app/backend && uv run pytest tests/ -v

run:
	cd app/api && doppler run -- uv run python run.py

download-weights:
	cd app/api && doppler run -- uv run python ../scripts/download_weights.py
