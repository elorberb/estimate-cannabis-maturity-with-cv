# Phase 1: MVP — Tasks

**29 tasks** | Weeks 1-5

## Workstream A: Infrastructure & Setup

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| A1 | **Supabase project setup** | Etay | Create Supabase project, configure `analyses` table (MVP schema), create storage bucket for images, generate API keys | None | Small |
| A2 | **Modal setup** | Etay | Create Modal account, set up project, configure GPU access (T4), create secrets for model weights storage | None | Small |
| A3 | **Expo project scaffold** | Gili | Initialize Expo managed project with TypeScript, Expo Router, configure app.json (name, icons, splash) | None | Small |
| A4 | **FastAPI project scaffold** | Etay | Set up `app/api/` with pyproject.toml, FastAPI app skeleton, CORS config, health check endpoint, environment config | None | Small |
| A5 | **CI/CD pipeline** | Etay + Gili | GitHub Actions for: backend tests, mobile build (EAS Build), linting | A3, A4 | Medium |

## Workstream B: GPU Inference (Modal)

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| B1 | **Package ML models for Modal** | Etay | Create Modal image with ultralytics, opencv, numpy. Upload YOLO model weights to Modal volumes | A2 | Medium |
| B2 | **Trichome inference function** | Etay | Wrap `TrichomeDetector` in a Modal `@app.function(gpu="T4")`. Accept image bytes, return detections + distribution | B1 | Medium |
| B3 | **Stigma inference function** | Etay | Wrap `StigmaDetector` in a Modal function. Return stigma detections + orange/green ratios | B1 | Medium |
| B4 | **Annotation rendering** | Etay | Function to draw trichome + stigma annotations on original image (colored bboxes, stigma outlines). Return annotated image bytes | B2, B3 | Medium |
| B5 | **Crop extraction** | Etay | Extract cropped trichome/stigma images from bbox regions for the review screen. Return as list of base64 thumbnails | B2, B3 | Small |
| B6 | **Combined analysis endpoint** | Etay | Single Modal function that orchestrates B2-B5: trichome detect -> stigma detect -> annotate -> crop -> return full result | B2, B3, B4, B5 | Medium |

## Workstream C: FastAPI Backend

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| C1 | **API schemas** | Etay | Pydantic models for API request/response: `AnalysisRequest`, `AnalysisResponse`, `CorrectionRequest`. Reuse existing models from `app/backend/src/models.py` where possible | A4 | Small |
| C2 | **Supabase storage service** | Gili | Service class to upload images to Supabase Storage, generate public URLs, download images | A1, A4 | Small |
| C3 | **Supabase database service** | Gili | Service class for CRUD on `analyses` table: create, get by ID, list by device_id, update corrections | A1, A4 | Small |
| C4 | **Modal client service** | Etay | Service class to call Modal inference function, handle response, error handling + timeouts | A4, B6 | Medium |
| C5 | **POST /api/v1/analyze** | Etay | Full endpoint: receive image upload -> store in Supabase Storage -> call Modal -> store results in DB -> return response with annotated image URL + stats | C1, C2, C3, C4 | Large |
| C6 | **GET /api/v1/analyses/{id}** | Gili | Retrieve single analysis by ID | C1, C3 | Small |
| C7 | **GET /api/v1/analyses** | Gili | List analyses filtered by device_id, with pagination | C1, C3 | Small |
| C8 | **PATCH /api/v1/analyses/{id}/corrections** | Gili | Accept corrected classifications, store alongside originals, recalculate distribution + maturity | C1, C3 | Medium |
| C9 | **Backend tests** | Etay + Gili | Unit tests for all services and endpoints. Integration test with mocked Modal | C5, C6, C7, C8 | Medium |
| C10 | **Deploy FastAPI** | Etay | Deploy to Railway/Fly.io, configure environment variables, verify health check | C5 | Small |

## Workstream D: Mobile App — Core Screens

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| D1 | **API client service** | Gili | TypeScript service (`api.ts`) to call all backend endpoints. Type definitions matching API schemas | A3, C1 | Small |
| D2 | **Home screen** | Gili | App branding, large "Analyze" button, recent analyses preview (thumbnails + maturity badges) | A3, D1 | Medium |
| D3 | **Camera/Upload screen** | Gili | Expo Camera integration for photo capture + ImagePicker for gallery selection. Tips overlay for macro photography guidance | A3 | Medium |
| D4 | **Analyzing screen** | Gili | Loading/progress screen shown during inference. Status text animation ("Detecting trichomes...", "Analyzing stigmas...") | A3 | Small |
| D5 | **Results screen** | Gili | Display annotated image, trichome distribution bar, stigma ratios, maturity badge, harvest recommendation. Buttons: Review, Save, Share | D1 | Large |
| D6 | **History screen** | Gili | Scrollable list of past analyses. Thumbnail + date + maturity badge. Pull-to-refresh. Tap to navigate to results | D1 | Medium |

## Workstream E: Mobile App — Review/Correct Flow

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| E1 | **Review screen — Trichomes tab** | Gili | Grid of cropped trichome images with colored borders (grey/white/orange). Tap to cycle classification. Live-updating distribution stats | D5 | Large |
| E2 | **Review screen — Stigmas tab** | Gili | Grid of cropped stigma images with color overlay. Tap to toggle orange/green classification. Live-updating ratios | D5 | Large |
| E3 | **Correction submission** | Gili | "Confirm & Save" button that sends corrected data via PATCH endpoint. Optimistic UI update | E1, E2, C8 | Medium |
| E4 | **Share functionality** | Gili | Share results as image (screenshot of results screen) or link via native share sheet | D5 | Small |

## Workstream F: Integration & Polish

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| F1 | **End-to-end integration test** | Etay + Gili | Upload image from mobile -> verify full flow through FastAPI -> Modal -> Supabase -> results displayed correctly | C10, D5 | Medium |
| F2 | **Error handling** | Gili | Handle: network errors, Modal cold starts/timeouts, invalid images, large files. User-friendly error messages | All D tasks | Medium |
| F3 | **Loading & empty states** | Gili | Skeleton loaders, empty state for history, retry on failure | D2, D5, D6 | Small |
| F4 | **App icon & splash screen** | Gili | Design and configure app icon and splash screen for iOS + Android | A3 | Small |

## Dependency Graph

```
A1 ─────────────────┬──▶ C2, C3
A2 ──▶ B1 ──┬──▶ B2 ┤
             └──▶ B3 ┤──▶ B4, B5 ──▶ B6 ──▶ C4 ──▶ C5 ──▶ C10 ──▶ F1
A3 ──▶ D1 ──┬──▶ D2  │
             ├──▶ D5 ──▶ E1, E2 ──▶ E3
             ├──▶ D6  │
             └──▶ D3, D4
A4 ──▶ C1 ──▶ C5, C6, C7, C8 ──▶ C9
```

## Assignment Summary

| Person | Tasks | Focus Areas |
|--------|-------|-------------|
| **Etay** | A1, A2, A4, B1-B6, C1, C4, C5, C10 (14 tasks) | Modal GPU setup, ML inference pipeline, FastAPI core, deployment |
| **Gili** | A3, C2, C3, C6, C7, C8, D1-D6, E1-E4, F2-F4 (17 tasks) | Expo app, all screens, review flow, Supabase services, read/correct endpoints |
| **Both** | A5, C9, F1 (3 tasks) | CI/CD, testing, integration |

## Suggested Work Order

**Week 1 — Setup (parallel)**
- Etay: A1, A2, A4, B1 (infra + model packaging)
- Gili: A3 (Expo scaffold)

**Week 2-3 — Core Build (parallel)**
- Etay: B2, B3, B4, B5, B6, C1, C4, C5 (ML pipeline + main API endpoint)
- Gili: D1, D2, D3, D4 (API client + core screens). Can mock API responses initially

**Week 3-4 — Integration + Review Flow**
- Etay: C10, C9 (deploy + tests)
- Gili: D5, D6, E1, E2, E3, C2, C3, C6, C7, C8 (results screen, history, review flow, Supabase services)

**Week 5 — Polish & Ship**
- Both: F1 (end-to-end integration)
- Gili: F2, F3, F4, E4 (error handling, polish, share)
- Both: A5 (CI/CD)
