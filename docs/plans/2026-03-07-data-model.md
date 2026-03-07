# Data Model & API Endpoints

## Database Schema

### Phase 1: MVP

```sql
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    device_id TEXT,                    -- anonymous device tracking (MVP)
    image_url TEXT NOT NULL,           -- original uploaded image
    annotated_image_url TEXT,          -- image with detections drawn
    trichome_distribution JSONB,      -- {clear: 0.15, cloudy: 0.65, amber: 0.20}
    stigma_ratios JSONB,              -- {orange: 0.72, green: 0.28}
    maturity_stage TEXT,              -- "early"|"developing"|"peak"|"mature"|"late"
    recommendation TEXT,              -- harvest recommendation text
    detections JSONB,                 -- full detection details (bboxes, classes, confidences)
    corrections JSONB                 -- user corrections (nullable, filled via PATCH)
);
```

### Phase 2: Auth Additions

```sql
-- Add user_id to analyses (nullable for backward compat with MVP anonymous data)
ALTER TABLE analyses ADD COLUMN user_id UUID REFERENCES auth.users(id);

-- Enable Row-Level Security
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

-- Users can only see their own analyses
CREATE POLICY "Users see own analyses" ON analyses
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

-- Users can only create analyses for themselves
CREATE POLICY "Users create own analyses" ON analyses
    FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### Phase 3: Plant Management Additions

```sql
CREATE TABLE plants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id),
    name TEXT NOT NULL,
    strain TEXT,
    planted_at DATE,
    notes TEXT,
    status TEXT DEFAULT 'active',    -- 'active' | 'harvested'
    harvested_at DATE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Add plant_id to analyses (nullable — not all analyses are linked to plants)
ALTER TABLE analyses ADD COLUMN plant_id UUID REFERENCES plants(id);

-- RLS: users see only their own plants
ALTER TABLE plants ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users see own plants" ON plants
    FOR ALL USING (auth.uid() = user_id);
```

---

## API Endpoints

### Phase 1: MVP

```
POST /api/v1/analyze
  - Input: multipart image upload
  - Process: upload to Supabase Storage → call Modal GPU → store results
  - Response: AnalysisResponse

GET  /api/v1/analyses/{id}
  - Returns a single analysis result

GET  /api/v1/analyses?device_id=xxx&limit=20
  - Returns analysis history for a device (MVP, no auth)

PATCH /api/v1/analyses/{id}/corrections
  - Input: list of corrected trichome/stigma classifications
  - Stores user corrections alongside original predictions
```

### Phase 2: Auth

```
POST /api/v1/auth/signup       — Proxied to Supabase Auth
POST /api/v1/auth/login        — Proxied to Supabase Auth
POST /api/v1/auth/logout       — Invalidate session
GET  /api/v1/auth/me           — Current user profile
POST /api/v1/auth/migrate      — Migrate device_id analyses to authenticated user
```

All existing endpoints now accept optional `Authorization: Bearer <token>` header. If present, analyses are tied to the user. If absent, device_id behavior continues (backward compatible).

### Phase 3: Plant Management

```
POST   /api/v1/plants                  — Create a new plant
GET    /api/v1/plants                  — List user's plants
GET    /api/v1/plants/{id}             — Get plant detail
PATCH  /api/v1/plants/{id}             — Update plant (name, strain, status)
DELETE /api/v1/plants/{id}             — Soft delete plant

GET    /api/v1/plants/{id}/analyses    — List analyses for a plant (timeline data)
GET    /api/v1/plants/{id}/prediction  — Get harvest prediction based on analysis trend

POST   /api/v1/plants/{id}/export/pdf  — Generate PDF report
GET    /api/v1/plants/{id}/export/csv  — Download CSV of analysis data
```

The existing `POST /api/v1/analyze` endpoint adds an optional `plant_id` parameter.

### Phase 4: Content & Quality

```
GET  /api/v1/content/lens-guide     — Lens guide content (JSON, updatable without app release)
GET  /api/v1/content/education      — Educational content sections (JSON)
POST /api/v1/analyze/quality-check  — Quick image quality assessment (sharpness score)
```
