# Trichome / Cannabis Maturity CV — Project Context

This repo implements cannabis flower maturity assessment using computer vision (trichome detection/classification and stigma color analysis). It includes thesis-derived ML code, with plans for a FastAPI + Modal GPU API and a React Native mobile app.

## For AI Agents

- **Code style and conventions:** Follow **AGENTS.md** at the repo root. It defines Python standards (imports, types, private members, testing, linting, etc.). All Python code in this repo must comply.
- **Architecture and roadmap:** See **app/docs/plans/** for the app design and phased roadmap (MVP → auth → multi-plant).

## Quick Reference

- **Backend (ML):** `app/backend/src/` — `TrichomeDetector`, `StigmaDetector`, Pydantic models, maturity assessment.
- **Lint:** `ruff check .` and `ruff format --check .`
- **Tests:** `cd app/backend && uv run pytest tests/ -v`
- **Planned:** `app/api/` (FastAPI), `app/modal/` (GPU inference), `app/mobile/` (Expo).

When reviewing code or suggesting changes, use AGENTS.md for style and conventions.
