# Trichome App — Design Documentation

A mobile app for cannabis growers to assess plant maturity using computer vision. Snap a macro photo, get instant trichome/stigma analysis with harvest recommendations.

## Quick Links

| Document | Description |
|----------|-------------|
| [Architecture](./2026-03-07-architecture.md) | Tech stack, system diagram, project structure, Modal GPU setup |
| [Data Model & API](./2026-03-07-data-model.md) | Database schema, API endpoints, all phases |
| [Screens & UX](./2026-03-07-screens.md) | All screen designs, wireframes, user flows |

## Phases

| Phase | Status | Docs |
|-------|--------|------|
| **Phase 1: MVP** — Snap & Analyze | Planned | [Overview](./phase-1-mvp/overview.md) · [Tasks](./phase-1-mvp/tasks.md) |
| **Phase 2: Auth** — User Accounts | Planned | [Overview](./phase-2-auth/overview.md) · [Tasks](./phase-2-auth/tasks.md) |
| **Phase 3: Plants** — Multi-Plant Management | Planned | [Overview](./phase-3-plants/overview.md) · [Tasks](./phase-3-plants/tasks.md) |
| **Phase 4: Guide & Education** — Lens Guide, Quality Check, Learn | Planned | [Overview](./phase-4-guide-education/overview.md) · [Tasks](./phase-4-guide-education/tasks.md) |

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: MVP | Weeks 1-5 | Snap & analyze app with correction flow |
| Phase 2: Auth | Weeks 5-7 | User accounts, history migration, profile |
| Phase 3: Plants | Weeks 7-10 | Plant tracking, timeline, prediction, export, notifications |
| Phase 4: Guide & Education | Weeks 11-12 | Lens guide, photo quality check, educational content |

## Team

| Person | Role | Focus |
|--------|------|-------|
| **Etay** | ML engineer & backend developer | Modal GPU, ML inference pipeline, FastAPI core, deployment |
| **Gili** | Full-stack developer (frontend-leaning) | Expo app, all screens, review flow, Supabase services |

## Context

This repo contains the published thesis algorithm for cannabis maturity assessment (published in Agriculture, MDPI Q1, 2026). The app wraps the existing `TrichomeDetector` and `StigmaDetector` from `app/backend/src/` into a user-friendly mobile experience. Monorepo approach — app code lives alongside thesis research code.
