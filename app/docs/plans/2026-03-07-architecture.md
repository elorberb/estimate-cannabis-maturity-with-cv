# Architecture

## System Diagram

```
React Native (Expo)  ──HTTPS──▶  FastAPI (CPU host)  ──API──▶  Modal (GPU)
     iOS/Android                   Railway/Fly.io              YOLO inference
                                        │
                                        ▼
                                   Supabase
                                 (Postgres + Storage + future Auth)
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Mobile App | React Native (Expo) | iOS + Android cross-platform app |
| API Server | FastAPI | API gateway, orchestrates inference + storage |
| GPU Inference | Modal (serverless GPU) | Runs TrichomeDetector + StigmaDetector on GPU |
| Database + Storage | Supabase | PostgreSQL, file storage, future authentication |

## Why These Choices

- **Expo**: Cross-platform with excellent camera APIs
- **FastAPI**: Python — directly reuses existing ML code, auto-generates OpenAPI docs
- **Modal**: Serverless GPU, pay-per-inference, scale-to-zero, Python-native
- **Supabase**: PostgreSQL + auth + storage in one service. When Phase 2 (auth) comes, just enable it — no migration

## Modal GPU Functions

Reuse existing code from `app/backend/src/`:

| File | What to Reuse |
|------|---------------|
| `trichome_detector.py` | TrichomeDetector class |
| `stigma_detector.py` | StigmaDetector class |
| `models.py` | All Pydantic models |
| `distribution.py` | `get_maturity_assessment()`, aggregate functions |
| `utils.py` | Image processing utilities |
| `config.py` | Configuration patterns |

```python
# Modal function (pseudocode)
@app.function(gpu="T4", image=modal_image)
def analyze_image(image_bytes: bytes) -> dict:
    trichome_result = trichome_detector.analyze(image_array)
    stigma_result = stigma_detector.analyze(image_array)
    maturity = get_maturity_assessment(trichome_result.distribution)
    annotated = draw_annotations(image_array, trichome_result, stigma_result)
    return {
        "trichome_distribution": ...,
        "stigma_ratios": ...,
        "maturity_stage": ...,
        "recommendation": ...,
        "detections": ...,       # individual bbox crops for review screen
        "annotated_image": ...,  # base64 or uploaded URL
    }
```

## Project Structure (Monorepo)

Everything lives in this repo (`repo-trichome-backend`). The thesis research code stays as-is; app code is added alongside it.

```
repo-trichome-backend/
├── src/                   # EXISTING — thesis research code (training, experiments, Streamlit)
├── notebooks/             # EXISTING — experimental notebooks
├── scripts/               # EXISTING — utility scripts
├── docs/                  # EXISTING — documentation + images
├── requirements/          # EXISTING — research dependencies
├── app/
│   ├── backend/           # EXISTING — ML package (TrichomeDetector, StigmaDetector, models)
│   │   ├── src/
│   │   └── tests/
│   ├── api/               # NEW — FastAPI server
│   │   ├── src/
│   │   │   ├── main.py        # FastAPI app, CORS, lifespan
│   │   │   ├── routes/
│   │   │   │   └── analysis.py # /analyze, /analyses endpoints
│   │   │   ├── services/
│   │   │   │   ├── modal_client.py  # Modal GPU function caller
│   │   │   │   └── storage.py       # Supabase storage service
│   │   │   ├── models/
│   │   │   │   └── schemas.py  # API request/response schemas
│   │   │   └── config.py       # Environment config
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── modal/             # NEW — Modal GPU functions
│   │   ├── inference.py       # GPU inference function (imports from app/backend/)
│   │   └── pyproject.toml
│   └── mobile/            # NEW — React Native (Expo) app
│       ├── app/               # Expo Router file-based routing
│       │   ├── (tabs)/
│       │   │   ├── index.tsx      # Home screen
│       │   │   └── history.tsx    # History screen
│       │   ├── camera.tsx         # Camera/upload screen
│       │   ├── analyzing.tsx      # Loading screen
│       │   ├── results/[id].tsx   # Results screen
│       │   └── review/[id].tsx    # Review/correct screen
│       ├── components/
│       ├── services/
│       │   └── api.ts         # API client
│       ├── app.json
│       └── package.json
├── pyproject.toml         # EXISTING — root project config
└── AGENTS.md              # EXISTING — coding standards
```

**Key benefit**: Modal functions in `app/modal/` can directly import from `app/backend/src/` — no package publishing or cross-repo deps needed.
