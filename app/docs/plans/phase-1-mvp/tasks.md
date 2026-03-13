# Phase 1: MVP — Tasks

**29 tasks** | Weeks 1-5

## infra: Infrastructure & Setup

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #17 | **infra-17: Supabase project setup** | Etay | Create Supabase project, configure `analyses` table (MVP schema), create storage bucket for images, generate API keys | None | Small |
| #18 | **infra-18: Modal setup** | Etay | Create Modal account, set up project, configure GPU access (T4), create secrets for model weights storage | None | Small |
| #19 | **infra-19: Expo project scaffold** | Gili | Initialize Expo managed project with TypeScript, Expo Router, configure app.json (name, icons, splash) | None | Small |
| #20 | **infra-20: FastAPI project scaffold** | Etay | Set up `app/api/` with pyproject.toml, FastAPI app skeleton, CORS config, health check endpoint, environment config | None | Small |
| #21 | **infra-21: CI/CD pipeline** | Etay + Gili | GitHub Actions for: backend tests, mobile build (EAS Build), linting | #19, #20 | Medium |

## back: GPU Inference (Modal)

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #22 | **back-22: Package ML models for Modal** | Etay | Create Modal image with ultralytics, opencv, numpy. Upload YOLO model weights to Modal volumes | #18 | Medium |
| #23 | **back-23: Trichome inference function** | Etay | Wrap `TrichomeDetector` in a Modal `@app.function(gpu="T4")`. Accept image bytes, return detections + distribution | #22 | Medium |
| #24 | **back-24: Stigma inference function** | Etay | Wrap `StigmaDetector` in a Modal function. Return stigma detections + orange/green ratios | #22 | Medium |
| #25 | **back-25: Annotation rendering** | Etay | Function to draw trichome + stigma annotations on original image (colored bboxes, stigma outlines). Return annotated image bytes | #23, #24 | Medium |
| #26 | **back-26: Crop extraction** | Etay | Extract cropped trichome/stigma images from bbox regions for the review screen. Return as list of base64 thumbnails | #23, #24 | Small |
| #27 | **back-27: Combined analysis endpoint** | Etay | Single Modal function that orchestrates #23–#26: trichome detect -> stigma detect -> annotate -> crop -> return full result | #23, #24, #25, #26 | Medium |

## back: FastAPI Backend

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #28 | **back-28: API schemas** | Etay | Pydantic models for API request/response: `AnalysisRequest`, `AnalysisResponse`, `CorrectionRequest`. Reuse existing models from `app/backend/src/models.py` where possible | #20 | Small |
| #29 | **back-29: Supabase storage service** | Gili | Service class to upload images to Supabase Storage, generate public URLs, download images | #17, #20 | Small |
| #30 | **back-30: Supabase database service** | Gili | Service class for CRUD on `analyses` table: create, get by ID, list by device_id, update corrections | #17, #20 | Small |
| #31 | **back-31: Modal client service** | Etay | Service class to call Modal inference function, handle response, error handling + timeouts | #20, #27 | Medium |
| #32 | **back-32: POST /api/v1/analyze** | Etay | Full endpoint: receive image upload -> store in Supabase Storage -> call Modal -> store results in DB -> return response with annotated image URL + stats | #28, #29, #30, #31 | Large |
| #33 | **back-33: GET /api/v1/analyses/{id}** | Gili | Retrieve single analysis by ID | #28, #30 | Small |
| #34 | **back-34: GET /api/v1/analyses** | Gili | List analyses filtered by device_id, with pagination | #28, #30 | Small |
| #35 | **back-35: PATCH /api/v1/analyses/{id}/corrections** | Gili | Accept corrected classifications, store alongside originals, recalculate distribution + maturity | #28, #30 | Medium |
| #36 | **back-36: Backend tests** | Etay + Gili | Unit tests for all services and endpoints. Integration test with mocked Modal | #32, #33, #34, #35 | Medium |
| #37 | **back-37: Deploy FastAPI** | Etay | Deploy to Railway/Fly.io, configure environment variables, verify health check | #32 | Small |

## front: Mobile App — Core Screens

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #38 | **front-38: API client service** | Gili | TypeScript service (`api.ts`) to call all backend endpoints. Type definitions matching API schemas | #19, #28 | Small |
| #39 | **front-39: Home screen** | Gili | App branding, large "Analyze" button, recent analyses preview (thumbnails + maturity badges) | #19, #38 | Medium |
| #40 | **front-40: Camera/Upload screen** | Gili | Expo Camera integration for photo capture + ImagePicker for gallery selection. Tips overlay for macro photography guidance | #19 | Medium |
| #41 | **front-41: Analyzing screen** | Gili | Loading/progress screen shown during inference. Status text animation ("Detecting trichomes...", "Analyzing stigmas...") | #19 | Small |
| #42 | **front-42: Results screen** | Gili | Display annotated image, trichome distribution bar, stigma ratios, maturity badge, harvest recommendation. Buttons: Review, Save, Share | #38 | Large |
| #43 | **front-43: History screen** | Gili | Scrollable list of past analyses. Thumbnail + date + maturity badge. Pull-to-refresh. Tap to navigate to results | #38 | Medium |

## front: Mobile App — Review/Correct Flow

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #44 | **front-44: Review screen — Trichomes tab** | Gili | Grid of cropped trichome images with colored borders (grey/white/orange). Tap to cycle classification. Live-updating distribution stats | #42 | Large |
| #45 | **front-45: Review screen — Stigmas tab** | Gili | Grid of cropped stigma images with color overlay. Tap to toggle orange/green classification. Live-updating ratios | #42 | Large |
| #46 | **front-46: Correction submission** | Gili | "Confirm & Save" button that sends corrected data via PATCH endpoint. Optimistic UI update | #44, #45, #35 | Medium |
| #47 | **front-47: Share functionality** | Gili | Share results as image (screenshot of results screen) or link via native share sheet | #42 | Small |

## front: Integration & Polish

| Issue | Task | Assignee | Description | Dependencies | Scope |
|-------|------|----------|-------------|--------------|-------|
| #48 | **front-48: End-to-end integration test** | Etay + Gili | Upload image from mobile -> verify full flow through FastAPI -> Modal -> Supabase -> results displayed correctly | #37, #42 | Medium |
| #49 | **front-49: Error handling** | Gili | Handle: network errors, Modal cold starts/timeouts, invalid images, large files. User-friendly error messages | All front tasks | Medium |
| #50 | **front-50: Loading & empty states** | Gili | Skeleton loaders, empty state for history, retry on failure | #39, #42, #43 | Small |
| #51 | **front-51: App icon & splash screen** | Gili | Design and configure app icon and splash screen for iOS + Android | #19 | Small |

## Dependency Graph

```
#17 ──────────────────┬──▶ #29, #30
#18 ──▶ #22 ──┬──▶ #23 ┤
               └──▶ #24 ┤──▶ #25, #26 ──▶ #27 ──▶ #31 ──▶ #32 ──▶ #37 ──▶ #48
#19 ──▶ #38 ──┬──▶ #39  │
               ├──▶ #42 ──▶ #44, #45 ──▶ #46
               ├──▶ #43  │
               └──▶ #40, #41
#20 ──▶ #28 ──▶ #32, #33, #34, #35 ──▶ #36
```

## Assignment Summary

| Person | Issues | Focus Areas |
|--------|--------|-------------|
| **Etay** | #17, #18, #20, #22–#28, #31, #32, #37 (14 tasks) | Modal GPU setup, ML inference pipeline, FastAPI core, deployment |
| **Gili** | #19, #29, #30, #33, #34, #35, #38–#47, #49–#51 (17 tasks) | Expo app, all screens, review flow, Supabase services, read/correct endpoints |
| **Both** | #21, #36, #48 (3 tasks) | CI/CD, testing, integration |

## Suggested Work Order

**Week 1 — Setup (parallel)**
- Etay: #17, #18, #20, #22 (infra + model packaging)
- Gili: #19 (Expo scaffold)

**Week 2-3 — Core Build (parallel)**
- Etay: #23, #24, #25, #26, #27, #28, #31, #32 (ML pipeline + main API endpoint)
- Gili: #38, #39, #40, #41 (API client + core screens). Can mock API responses initially

**Week 3-4 — Integration + Review Flow**
- Etay: #37, #36 (deploy + tests)
- Gili: #42, #43, #44, #45, #46, #29, #30, #33, #34, #35 (results screen, history, review flow, Supabase services)

**Week 5 — Polish & Ship**
- Both: #48 (end-to-end integration)
- Gili: #49, #50, #51, #47 (error handling, polish, share)
- Both: #21 (CI/CD)
