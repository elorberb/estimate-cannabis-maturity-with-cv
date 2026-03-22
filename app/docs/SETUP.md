# App setup — get running on your machine

Everything you need to run the backend API, download model weights, and start the mobile app. Covers both Mac and Windows.

---

## Step 1 — Install tools (one-time)

**uv** (Python package manager):

- Mac: `brew install uv`
- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Doppler** (secrets manager — replaces `.env`):

- Mac: `brew install dopplerhq/cli/doppler`
- Windows: `winget install Doppler.doppler`

**Node.js** (for the mobile app — skip if already installed): https://nodejs.org

Restart your terminal after installing.

---

## Step 2 — Pull latest

```bash
git checkout main && git pull
```

---

## Step 3 — Secrets (Doppler)

All secrets (Supabase URL, keys, etc.) are in Doppler. Ask Etay for a workspace invite, then:

```bash
doppler login   # opens browser
doppler setup --project cannabis-maturity-app --config dev --scope app/api
```

No `.env` file needed — Doppler injects secrets automatically when you run the API.

---

## Step 4 — Install Python dependencies

```bash
cd app/api && uv sync
```

---

## Step 5 — Download model weights

The ML models (~102 MB) are stored in Supabase. Download them once:

```bash
# Mac
make download-weights

# Windows (from app/api)
doppler run -- uv run python ../scripts/download_weights.py
```

---

## Step 6 — Run the API

```bash
# Mac
make run

# Windows (from app/api)
doppler run -- uv run python run.py
```

The API starts at **http://localhost:8000**

| URL | What it is |
|---|---|
| http://localhost:8000/health | Health check |
| http://localhost:8000/docs | Interactive API docs |
| `POST /api/v1/analyze` | Main endpoint |

---

## Step 7 — Test with a sample image

Sample trichome photos are in `app/tests/fixtures/images/` — one per growth day.

**Swagger UI** (easiest): go to http://localhost:8000/docs → `POST /api/v1/analyze` → Try it out → upload any sample image.

**curl:**
```bash
# Mac
curl -X POST "http://localhost:8000/api/v1/analyze?device_id=test" \
  -F "file=@app/tests/fixtures/images/sample_day1.jpg"

# Windows PowerShell
curl -X POST "http://localhost:8000/api/v1/analyze?device_id=test" `
  -F "file=@app/tests/fixtures/images/sample_day1.jpg"
```

---

## Step 8 — Run the mobile app

```bash
cd app/mobile && npm install && npx expo start
```

Set the API base URL to your machine's IP (not `localhost`) so your phone/emulator can reach it.
Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac) → look for the IPv4 address.

---

## Daily workflow

```bash
# Terminal 1 — backend (Mac)
make run

# Terminal 1 — backend (Windows, from app/api)
doppler run -- uv run python run.py

# Terminal 2 — mobile app
cd app/mobile && npx expo start
```

---

## Supabase (already set up — nothing to do)

The shared Supabase project has the `analyses` table and `images` bucket already configured. Doppler has the credentials. You don't need to touch it.

---

## Modal GPU inference (Etay only)

```bash
cd app/modal && uv run modal run upload_weights.py   # upload weights once
cd app/modal && uv run modal deploy inference.py     # deploy
```

Switch to Modal by setting `INFERENCE_MODE=modal` in Doppler (dashboard → cannabis-maturity-app → dev).

---

## Troubleshooting

**`doppler: command not found`** — restart terminal after installing.

**`uv: command not found`** — restart terminal after installing.

**API 500 on `/analyze`** — weights not downloaded yet, re-run Step 5.

**Mobile app can't reach API** — use your machine's IP address, not `localhost`.
