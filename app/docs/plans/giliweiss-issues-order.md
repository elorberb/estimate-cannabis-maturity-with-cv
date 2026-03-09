## Gili Weiss – Phase 1 Issue Order

This file tracks the working order for all open Phase 1 issues assigned to `giliweiss` in `elorberb/estimate-cannabis-maturity-with-cv`.  
The goal is to move from infrastructure → backend → mobile → integration/polish, while respecting dependencies.

### Recommended working order

1. **[A3] Expo project scaffold (#19)**
   - **Scope**: Small  
   - **Area**: Infrastructure & setup  
   - **Description**: Initialize Expo managed project with TypeScript and Expo Router, configure `app.json` (name, icons, splash).  
   - **Dependencies**: None.

2. **[C2] Supabase storage service (#29)**
   - **Scope**: Small  
   - **Area**: Backend – FastAPI  
   - **Description**: Service class to upload images to Supabase Storage, generate public URLs, and download images.  
   - **Dependencies**: `[A1] Supabase project setup` (done), `[A4] FastAPI project scaffold` (done).

3. **[C3] Supabase database service (#30)**
   - **Scope**: Small  
   - **Area**: Backend – FastAPI  
   - **Description**: Service class for CRUD on `analyses` table: create, get by ID, list by `device_id`, update corrections.  
   - **Dependencies**: `[A1] Supabase project setup` (done), `[A4] FastAPI project scaffold` (done).

4. **[C6] GET /api/v1/analyses/{id} (#33)**
   - **Scope**: Small  
   - **Area**: Backend – FastAPI  
   - **Description**: Retrieve single analysis by ID using the Supabase services.  
   - **Dependencies**: `[C1] API schemas`, `[C3] Supabase database service`.

5. **[C7] GET /api/v1/analyses (#34)**
   - **Scope**: Small  
   - **Area**: Backend – FastAPI  
   - **Description**: List analyses filtered by `device_id`, with pagination.  
   - **Dependencies**: `[C1] API schemas`, `[C3] Supabase database service`.

6. **[C8] PATCH /api/v1/analyses/{id}/corrections (#35)**
   - **Scope**: Medium  
   - **Area**: Backend – FastAPI  
   - **Description**: Accept corrected trichome/stigma classifications, store alongside originals, recalculate distribution and maturity assessment.  
   - **Dependencies**: `[C1] API schemas`, `[C3] Supabase database service`.

7. **[C9] Backend tests (#36)**
   - **Scope**: Medium  
   - **Area**: Backend – FastAPI  
   - **Description**: Unit tests for backend services and endpoints, plus integration test(s) with mocked Modal.  
   - **Dependencies**: `[C5] POST /api/v1/analyze`, `[C6]`, `[C7]`, `[C8]`.

8. **[D1] API client service (#38)**
   - **Scope**: Small  
   - **Area**: Mobile – Core screens  
   - **Description**: TypeScript service (for example `api.ts`) that calls backend endpoints with types matching API schemas.  
   - **Dependencies**: `[A3] Expo project scaffold`, `[C1] API schemas`.

9. **[D2] Home screen (#39)**
   - **Scope**: Medium  
   - **Area**: Mobile – Core screens  
   - **Description**: Home screen with app branding, large “Analyze” button, and recent analyses preview.  
   - **Dependencies**: `[A3] Expo project scaffold`, `[D1] API client service`.

10. **[D3] Camera/Upload screen (#40)**
    - **Scope**: Medium  
    - **Area**: Mobile – Core screens  
    - **Description**: Expo Camera integration for photo capture and ImagePicker for gallery selection, with macro photography tips.  
    - **Dependencies**: `[A3] Expo project scaffold`.

11. **[D4] Analyzing screen (#41)**
    - **Scope**: Small  
    - **Area**: Mobile – Core screens  
    - **Description**: Loading/progress screen shown during inference, with status text like “Detecting trichomes…”, “Classifying…”, “Analyzing stigmas…”.  
    - **Dependencies**: `[A3] Expo project scaffold`.

12. **[D5] Results screen (#42)**
    - **Scope**: Large  
    - **Area**: Mobile – Core screens  
    - **Description**: Results view showing annotated image, trichome distribution, stigma ratios, maturity badge, and harvest recommendation, with buttons for review, save, and share.  
    - **Dependencies**: `[D1] API client service`.

13. **[D6] History screen (#43)**
    - **Scope**: Medium  
    - **Area**: Mobile – Core screens  
    - **Description**: Scrollable list of past analyses with thumbnail, date, and maturity badge; pull-to-refresh; tap to open results.  
    - **Dependencies**: `[D1] API client service`.

14. **[E1] Review screen — Trichomes tab (#44)**
    - **Scope**: Large  
    - **Area**: Mobile – Review/Correct flow  
    - **Description**: Grid of cropped trichome images with colored borders (clear/cloudy/amber); tap to cycle classification with live-updating distribution stats.  
    - **Dependencies**: `[D5] Results screen`, Modal crop/annotation outputs.

15. **[E2] Review screen — Stigmas tab (#45)**
    - **Scope**: Large  
    - **Area**: Mobile – Review/Correct flow  
    - **Description**: Grid of cropped stigma images with overlays; tap to toggle orange/green classification; live-updating ratios.  
    - **Dependencies**: `[D5] Results screen`, Modal crop/annotation outputs.

16. **[E3] Correction submission (#46)**
    - **Scope**: Medium  
    - **Area**: Mobile – Review/Correct flow  
    - **Description**: “Confirm & Save” flow that sends corrections via PATCH endpoint with optimistic UI and stores originals plus corrections.  
    - **Dependencies**: `[E1]`, `[E2]`, `[C8] PATCH corrections endpoint`.

17. **[E4] Share functionality (#47)**
    - **Scope**: Small  
    - **Area**: Mobile – Review/Correct flow  
    - **Description**: Share results as image (screenshot of results screen) or link via the native share sheet.  
    - **Dependencies**: `[D5] Results screen`.

18. **[F1] End-to-end integration test (#48)**
    - **Scope**: Medium  
    - **Area**: Integration & polish  
    - **Description**: Verify full flow from mobile upload through FastAPI → Modal → Supabase → results display.  
    - **Dependencies**: FastAPI deployed, Modal pipeline working, `[D5] Results screen`.

19. **[F2] Error handling (#49)**
    - **Scope**: Medium  
    - **Area**: Integration & polish  
    - **Description**: Handle network errors, Modal cold starts/timeouts, invalid images, large files; show user-friendly errors across the app.  
    - **Dependencies**: All D tasks (D1–D6) and basic integration working.

20. **[F3] Loading & empty states (#50)**
    - **Scope**: Small  
    - **Area**: Integration & polish  
    - **Description**: Skeleton loaders, history empty state (“No analyses yet”), and retry on failures.  
    - **Dependencies**: `[D2] Home screen`, `[D5] Results screen`, `[D6] History screen`.

21. **[F4] App icon & splash screen (#51)**
    - **Scope**: Small  
    - **Area**: Integration & polish  
    - **Description**: Design and configure app icon and splash screen for iOS and Android.  
    - **Dependencies**: `[A3] Expo project scaffold`.

22. **[A5] CI/CD pipeline (#21)**
    - **Scope**: Medium  
    - **Area**: Infrastructure & setup  
    - **Description**: GitHub Actions for backend tests, mobile build (EAS Build), and linting.  
    - **Dependencies**: `[A3] Expo project scaffold`, `[A4] FastAPI project scaffold`.

