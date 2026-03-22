# App setup — get running on your machine

Use this when you clone the repo and want to run the API (and optionally the mobile app) with Modal and Supabase. All paths are from the **repository root** unless noted.

> **Windows / Gili?** See [gili-setup.md](./gili-setup.md) for a step-by-step Windows guide.

---

## 1. Clone and install tools

```bash
git clone <repo-url>
cd thesis
```

- **uv** (Python package manager): [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Doppler** (secrets): [install Doppler CLI](https://docs.doppler.com/docs/install-cli)
- **Supabase CLI** (optional — only if running Supabase locally)

---

## 2. Secrets (Doppler)

All environment variables are managed in Doppler. Ask Etay for a workspace invite.

```bash
doppler login
doppler setup --project cannabis-maturity-app --config dev --scope app/api
```

No `.env` file needed — Doppler injects secrets at runtime.

---

## 3. Supabase

The shared hosted project is already set up (`analyses` table + `images` bucket). Nothing to do if you're using the shared project.

If you need a local Supabase instead:
```bash
cd app/supabase && supabase start
```
Then update Doppler (or a local `.env`) with the printed API URL and service_role key.

---

## 4. Model weights (local inference only)

Skip if using `INFERENCE_MODE=modal`.

```bash
make download-weights
# or: cd app/api && doppler run -- uv run python ../scripts/download_weights.py
```

Downloads ~102 MB to `checkpoints/` (one-time).

---

## 5. Modal (only if INFERENCE_MODE=modal)

```bash
# Upload weights to Modal volume (one-time)
cd app/modal && uv run modal run upload_weights.py

# Deploy
cd app/modal && uv run modal deploy inference.py
```

---

## 6. Run the API

```bash
make run
# or: cd app/api && doppler run -- uv run python run.py
```

- Health: http://127.0.0.1:8000/health
- Docs: http://127.0.0.1:8000/docs

---

## 7. Mobile app (optional)

```bash
cd app/mobile && npm install && npx expo start
```

---

## Checklist

| Step | What to do |
|------|------------|
| 1 | Clone repo, install uv + Doppler |
| 2 | `doppler login` + `doppler setup` |
| 3 | Use shared Supabase project (already set up) |
| 4 | If local inference: `make download-weights` |
| 5 | If Modal: upload weights + `modal deploy inference.py` |
| 6 | `make run` |
| 7 | Optional: `cd app/mobile && npx expo start` |

More detail: [Working with Modal and Supabase](./working-with-modal-and-supabase.md).
