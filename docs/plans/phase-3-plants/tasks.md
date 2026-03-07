# Phase 3: Multi-Plant Management — Tasks

**17 tasks** | Weeks 7-10

## Task Breakdown

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| P3.1 | **Plants DB schema** | Etay | Create plants table, add plant_id to analyses, RLS policies | Phase 2 complete | Small |
| P3.2 | **Plants CRUD API** | Etay | POST/GET/PATCH/DELETE endpoints for plants | P3.1 | Medium |
| P3.3 | **Plant analyses API** | Etay | GET /plants/{id}/analyses, timeline data aggregation | P3.1 | Small |
| P3.4 | **Harvest prediction API** | Etay | Trend analysis algorithm, prediction endpoint. Uses numpy for linear regression on trichome % over time | P3.3 | Large |
| P3.5 | **Export — PDF generation** | Etay | Generate PDF with plant summary + maturity chart + analysis table. Use reportlab or weasyprint | P3.3 | Medium |
| P3.6 | **Export — CSV endpoint** | Etay | Stream CSV of plant analysis data | P3.3 | Small |
| P3.7 | **Push notification backend** | Etay | Expo push tokens storage, cron job for scheduled reminders + maturity alerts | P3.4 | Large |
| P3.8 | **Plants list screen** | Gili | Plant cards grouped by status, add plant button, search/filter | P3.2 | Medium |
| P3.9 | **Add/Edit plant screen** | Gili | Plant form with strain autocomplete, date picker | P3.2 | Medium |
| P3.10 | **Plant detail — info + chart** | Gili | Plant header, area chart (react-native-chart-kit or Victory Native) showing trichome distribution over time | P3.3 | Large |
| P3.11 | **Plant detail — prediction banner** | Gili | Display harvest prediction with confidence indicator | P3.4, P3.10 | Small |
| P3.12 | **Plant detail — analysis timeline** | Gili | Vertical card timeline of analyses, tap to view results | P3.3, P3.10 | Medium |
| P3.13 | **Quick analyze from plant** | Gili | Camera/upload flow pre-linked to a plant_id | P3.10 | Small |
| P3.14 | **Export/report screen** | Gili | Export options UI, PDF preview, CSV download, share sheet | P3.5, P3.6 | Medium |
| P3.15 | **Push notification setup (mobile)** | Gili | Expo push token registration, notification handling, schedule settings per plant | P3.7 | Medium |
| P3.16 | **Navigation updates** | Gili | Add plants tab to bottom navigation, update routing | P3.8 | Small |
| P3.17 | **Phase 3 tests** | Etay + Gili | Backend: plant CRUD, prediction, export. Mobile: plant flows | All P3 tasks | Medium |

## Assignment Summary

| Person | Tasks | Focus |
|--------|-------|-------|
| **Etay** | P3.1-P3.7 (7 tasks) | DB schema, plant API, prediction algorithm, export, push notifications backend |
| **Gili** | P3.8-P3.16 (9 tasks) | All plant screens, chart, timeline, export UI, push setup, navigation |
| **Both** | P3.17 (1 task) | Testing |

## Work Order

**Weeks 7-8 — Core Plant Management**
- Etay: P3.1, P3.2, P3.3 (DB + CRUD + timeline API)
- Gili: P3.8, P3.9, P3.16 (plant list, form, navigation)

**Weeks 8-9 — Timeline + Prediction**
- Etay: P3.4, P3.5, P3.6 (prediction algorithm, export)
- Gili: P3.10, P3.12, P3.13 (chart, timeline, quick analyze)

**Weeks 9-10 — Notifications + Polish**
- Etay: P3.7 (push notification backend)
- Gili: P3.11, P3.14, P3.15 (prediction banner, export screen, push setup)
- Both: P3.17 (testing)
