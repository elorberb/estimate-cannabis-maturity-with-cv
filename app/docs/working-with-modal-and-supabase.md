# Working with Modal and Supabase

How to run and use the Modal GPU inference app and Supabase (database + storage) for the cannabis maturity API.

## Prerequisites

- **Modal**: [modal.com](https://modal.com) account and CLI (`pip install modal` or use project `uv`).
- **Supabase**: [supabase.com](https://supabase.com) project (or local Supabase via CLI).
- **API**: From repo root, `cd app/api && uv sync`.

---

## 1. Modal (GPU inference)

The inference app runs on Modal and is used by the FastAPI app when `inference_mode=modal`.

### One-time setup

1. **Auth**
   ```bash
   modal token new
   ```
   Follow the browser flow to log in.

2. **Upload model weights** (required before inference works)
   ```bash
   cd app/modal && uv run modal run upload_weights.py
   ```
   This reads weights from the repo (e.g. `checkpoints/trichome_detection/yolov9_best.pt`) and uploads them to the Modal volume `cannabis-maturity-model-weights`. Ensure those checkpoint paths exist or adjust paths in `upload_weights.py`.

3. **Deploy the app**
   ```bash
   cd app/modal && uv run modal deploy inference.py
   ```
   The deployed app name is `cannabis-maturity-inference` (used by the API’s `modal_app_name` setting).

### Day-to-day

- **Redeploy after code changes** in `app/modal` or `app/backend/src`:
  ```bash
  cd app/modal && uv run modal run inference.py
  ```
  Or for a long-lived deployment: `uv run modal deploy inference.py`.

- **Test inference locally** (calls the deployed app):
  ```bash
  cd app/modal && uv run modal run inference.py
  ```
  The `main()` entrypoint only prints a success message; real calls go through the API or `modal.Function.lookup(..., "MaturityAnalyzer.analyze").remote(image_bytes)`.

- **Inspect volume**
  ```bash
  modal volume ls cannabis-maturity-model-weights
  ```

---

## 2. Supabase (database + storage)

The API uses Supabase for the `analyses` table and for image storage (bucket `images`).

### Option A: Hosted Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. In the dashboard: **Settings → API** copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role** key (secret) → `SUPABASE_SERVICE_KEY`
3. **Storage**: create a bucket named `images` (private is fine; the API uses the service key).
4. **Database**: apply the schema. Either run the migration SQL from the repo, or use the Supabase CLI (see Option B) and then link to the hosted project.

Migration to apply (from `app/supabase/migrations/20260307163836_create_analyses_table.sql`):

```sql
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    device_id TEXT,
    image_url TEXT NOT NULL,
    annotated_image_url TEXT,
    trichome_distribution JSONB,
    stigma_ratios JSONB,
    maturity_stage TEXT,
    recommendation TEXT,
    detections JSONB,
    corrections JSONB
);
```

### Option B: Local Supabase (CLI)

1. Install [Supabase CLI](https://supabase.com/docs/guides/cli).
2. From the app (or repo):
   ```bash
   cd app/supabase && supabase start
   ```
   This starts Postgres, Studio, and Storage locally. Use the printed URLs and keys for local development.
3. Migrations in `app/supabase/migrations/` are applied when you run `supabase start` (or `supabase db reset`). The `analyses` table is created by the migration above.
4. Create the `images` bucket in Studio (http://127.0.0.1:54323) under Storage.

---

## 3. API configuration

Run the API from `app/api` with a `.env` file (or environment variables). Pydantic-settings reads `app/api/.env` when you run the app from `app/api`.

### Required for Supabase (and for `/analyze` with storage/DB)

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

Optional:

- `SUPABASE_STORAGE_BUCKET=images` (default)
- `INFERENCE_MODE=modal` or `local` (default `local`)
- `MODAL_APP_NAME=cannabis-maturity-inference` (when using Modal)

### Running the API

```bash
cd app/api && uv run uvicorn src.main:app --reload
```

- With `INFERENCE_MODE=local`, the API uses the local ML stack (checkpoints on disk; see `config.py` paths).
- With `INFERENCE_MODE=modal`, the API calls the deployed Modal app; ensure the app is deployed and weights are uploaded.

### Flow when using Modal + Supabase

1. Client uploads an image to `POST /api/v1/analyze`.
2. API uploads the image to Supabase Storage and gets a URL.
3. API calls Modal `MaturityAnalyzer.analyze(image_bytes)` and gets trichome/stigma results and an annotated image.
4. API uploads the annotated image to Storage, then inserts a row into `analyses` (via `DatabaseService`) with URLs and result JSON.
5. API returns the analysis response including `id`, `image_url`, `annotated_image_url`, and maturity fields.

---

## Quick reference

| Task | Command / location |
|------|--------------------|
| Modal login | `modal token new` |
| Upload weights | `cd app/modal && uv run modal run upload_weights.py` |
| Deploy Modal app | `cd app/modal && uv run modal deploy inference.py` |
| List weights | `modal volume ls cannabis-maturity-model-weights` |
| Local Supabase | `cd app/supabase && supabase start` |
| API env | `app/api/.env` with `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, optional `INFERENCE_MODE=modal` |
| Run API | `cd app/api && uv run uvicorn src.main:app --reload` |
