from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes import analysis

app = FastAPI(
    title="Trichome Analysis API",
    description="Cannabis maturity assessment via trichome and stigma analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
