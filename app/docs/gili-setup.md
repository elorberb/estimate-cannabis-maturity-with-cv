# Setup guide for Gili

Everything you need to get the full stack running on your Windows machine — backend API, database, secrets, model weights, and mobile app.

---

## Step 1 — Install tools (one-time)

Open **PowerShell** and run each of these:

**uv** (Python package manager):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Doppler** (secrets manager — replaces `.env`):
```powershell
winget install Doppler.doppler
```

**Node.js** (for the mobile app — skip if already installed):
Download from https://nodejs.org and install the LTS version.

Restart PowerShell after installing.

---

## Step 2 — Pull latest

```powershell
git checkout main && git pull
```

---

## Step 3 — Secrets (Doppler)

Etay has already added all secrets (Supabase URL, keys, etc.) to Doppler and invited you to the workspace.

```powershell
# Log in (opens browser — use the invite email)
doppler login

# Link the api folder to the project
doppler setup --project cannabis-maturity-app --config dev --scope app/api
```

That's it — no `.env` file needed. Doppler injects all secrets automatically when you run the API.

---

## Step 4 — Install Python dependencies

```powershell
cd app/api
uv sync
```

---

## Step 5 — Download model weights

The ML models (~102 MB) are stored in Supabase. Download them once:

```powershell
# Still in app/api
doppler run -- uv run python ../scripts/download_weights.py
```

This downloads to `checkpoints/` at the repo root. You don't need to touch them again unless something changes.

---

## Step 6 — Verify the database

The shared Supabase project already has the `analyses` table and `images` storage bucket set up (Etay did this). You don't need to do anything — Doppler already has the connection details.

To verify, run the API (Step 7) and hit the health endpoint:
```
http://localhost:8000/health  →  {"status": "ok"}
```

---

## Step 7 — Run the API

```powershell
# From app/api
doppler run -- uv run python run.py
```

The API starts at **http://localhost:8000**

| URL | What it is |
|---|---|
| http://localhost:8000/health | Health check |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| `POST /api/v1/analyze` | Main endpoint — upload image, get maturity result |

---

## Step 8 — Test with a sample image

Sample trichome photos are in `app/tests/fixtures/images/` — one per growth day from the experiment.

**Option A — using the Swagger UI:**
1. Go to http://localhost:8000/docs
2. Click `POST /api/v1/analyze` → **Try it out**
3. Set `device_id` to anything (e.g. `test`)
4. Upload any image from `app/tests/fixtures/images/`
5. Hit **Execute** — you'll see the full JSON response

**Option B — using curl in PowerShell:**
```powershell
curl -X POST "http://localhost:8000/api/v1/analyze?device_id=test" `
  -F "file=@../tests/fixtures/images/sample_day1.jpg"
```

---

## Step 9 — Run the mobile app

```powershell
cd app/mobile
npm install
npx expo start
```

Point the app at the API. In the mobile app config, set the API base URL to `http://<your-machine-ip>:8000` (use your local IP, not `localhost`, so your phone/emulator can reach it).

Find your IP: run `ipconfig` in PowerShell and look for **IPv4 Address**.

---

## Daily workflow

Once everything is set up, you only need:

```powershell
# Terminal 1 — backend
cd app/api && doppler run -- uv run python run.py

# Terminal 2 — mobile app
cd app/mobile && npx expo start
```

---

## Troubleshooting

**`doppler: command not found`** — restart PowerShell after installing, or add Doppler to your PATH.

**`uv: command not found`** — restart PowerShell after installing uv.

**API starts but `/analyze` returns 500** — check the terminal output for the error. Most likely the weights aren't downloaded yet (re-run Step 5).

**Mobile app can't reach API** — make sure you're using your machine's IP address (not `localhost`) in the mobile app config.

**Weights download fails** — make sure Doppler is linked (Step 3) so the script has access to the Supabase credentials.
