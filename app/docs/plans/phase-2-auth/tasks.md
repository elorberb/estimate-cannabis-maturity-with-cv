# Phase 2: User Accounts — Tasks

**10 tasks** | Weeks 5-7

## Task Breakdown

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| P2.1 | **Supabase Auth config** | Etay | Enable Auth in Supabase dashboard, configure Google + Apple OAuth providers, set redirect URLs | Phase 1 complete | Small |
| P2.2 | **Auth API endpoints** | Etay | FastAPI endpoints for signup, login, logout, me, migrate. JWT validation middleware | P2.1 | Medium |
| P2.3 | **Auth middleware** | Etay | Middleware to extract user from Bearer token, make user optional (backward compat) | P2.2 | Small |
| P2.4 | **DB migration — user_id** | Etay | Add user_id column, RLS policies, migration for device_id -> user_id | P2.1 | Small |
| P2.5 | **Auth screen (mobile)** | Gili | Login/signup screen with email + Google + Apple. Token storage (SecureStore) | P2.2 | Large |
| P2.6 | **Auth state management** | Gili | Global auth context, auto-refresh tokens, protected routes | P2.5 | Medium |
| P2.7 | **Profile screen** | Gili | User info, stats, settings, logout | P2.6 | Medium |
| P2.8 | **Device migration flow** | Gili | On first login, prompt to migrate anonymous history. Call migrate endpoint | P2.2, P2.6 | Small |
| P2.9 | **Update existing screens** | Gili | Home shows user-specific history, history screen filters by user | P2.6 | Small |
| P2.10 | **Auth tests** | Etay + Gili | Backend auth tests, mobile auth flow tests | P2.2-P2.9 | Medium |

## Assignment Summary

| Person | Tasks | Focus |
|--------|-------|-------|
| **Etay** | P2.1, P2.2, P2.3, P2.4 (4 tasks) | Supabase Auth setup, API endpoints, middleware, DB migration |
| **Gili** | P2.5, P2.6, P2.7, P2.8, P2.9 (5 tasks) | Auth screen, state management, profile, migration UX |
| **Both** | P2.10 (1 task) | Testing |

## Work Order

**Week 5-6**
- Etay: P2.1, P2.2, P2.3, P2.4 (auth backend)
- Gili: P2.5, P2.6 (auth screen + state management) — can start once P2.2 is ready

**Week 6-7**
- Gili: P2.7, P2.8, P2.9 (profile, migration, screen updates)
- Both: P2.10 (testing)
