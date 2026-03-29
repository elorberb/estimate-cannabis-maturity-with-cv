from __future__ import annotations

import base64
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from supabase import create_client

from config import settings
from models.schemas import (
    AnalysisListItem,
    AnalysisListResponse,
    AnalyzeResponse,
    PlantAnalysisHistory,
    PlantAnalysisItem,
    PlantCreate,
    PlantListResponse,
    PlantResponse,
)
from services.database_service import DatabaseService
from services.inference_error import InferenceError
from services.storage_service import StorageService

router = APIRouter(tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _supabase():
    return create_client(settings.supabase_url, settings.supabase_service_key)


# ─── ANALYZE ─────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(
    request: Request,
    file: UploadFile,
    device_id: str,
    plant_id: str | None = None,
) -> AnalyzeResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported image type: {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 20 MB limit",
        )

    storage = StorageService(_supabase(), settings.supabase_storage_bucket)
    db = DatabaseService(_supabase())

    image_url = storage.upload_image(image_bytes, content_type=file.content_type or "image/jpeg")

    try:
        result: dict = await request.app.state.inference_service.analyze(image_bytes)
    except InferenceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    annotated_image_url: str | None = None
    if result.get("annotated_image_b64"):
        annotated_bytes = base64.b64decode(result["annotated_image_b64"])
        annotated_image_url = storage.upload_image(annotated_bytes, content_type="image/jpeg")

    record = db.save_analysis(device_id, image_url, annotated_image_url, result, plant_id)

    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        device_id=device_id,
        plant_id=plant_id,
        image_url=image_url,
        annotated_image_url=annotated_image_url,
        trichome_result=result["trichome_result"],
        stigma_result=result["stigma_result"],
        maturity_stage=result["maturity_stage"],
        recommendation=result["recommendation"],
        trichome_crops_b64=result.get("trichome_crops_b64"),
        stigma_crops_b64=result.get("stigma_crops_b64"),
    )


# ─── ANALYSES LIST / DELETE ──────────────────────────────────────────────────

@router.get("/analyses", response_model=AnalysisListResponse)
def list_analyses(device_id: str | None = None, limit: int = 20) -> AnalysisListResponse:
    db = DatabaseService(_supabase())
    records = db.list_analyses(device_id=device_id, limit=limit)
    items = [
        AnalysisListItem(
            id=r["id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            device_id=r["device_id"],
            plant_id=r.get("plant_id"),
            image_url=r["image_url"],
            annotated_image_url=r.get("annotated_image_url"),
            maturity_stage=r["maturity_stage"],
            recommendation=r["recommendation"],
            trichome_distribution=r.get("trichome_distribution"),
            stigma_ratios=r.get("stigma_ratios"),
        )
        for r in records
    ]
    return AnalysisListResponse(items=items, total=len(items))


@router.get("/analyses/{analysis_id}", response_model=AnalyzeResponse)
def get_analysis(analysis_id: str) -> AnalyzeResponse:
    db = DatabaseService(_supabase())
    record = db.get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    detections = record.get("detections") or {}
    trichome_dist = record.get("trichome_distribution") or {"clear": 0, "cloudy": 0, "amber": 0}
    stigma_ratios = record.get("stigma_ratios") or {"green": 0.0, "orange": 0.0}
    trichome_detections = detections.get("trichomes") or []
    stigma_detections = detections.get("stigmas") or []
    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        device_id=record["device_id"],
        plant_id=record.get("plant_id"),
        image_url=record["image_url"],
        annotated_image_url=record.get("annotated_image_url"),
        trichome_result={
            "detections": trichome_detections,
            "distribution": trichome_dist,
            "total_count": sum(trichome_dist.values()),
        },
        stigma_result={
            "detections": stigma_detections,
            "avg_green_ratio": stigma_ratios.get("green", 0.0),
            "avg_orange_ratio": stigma_ratios.get("orange", 0.0),
            "total_count": len(stigma_detections),
        },
        maturity_stage=record["maturity_stage"],
        recommendation=record["recommendation"],
        trichome_crops_b64=None,
        stigma_crops_b64=None,
    )


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str) -> None:
    db = DatabaseService(_supabase())
    if not db.get_analysis(analysis_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    db.delete_analysis(analysis_id)


@router.delete("/plants/{plant_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plant(plant_id: str) -> None:
    db = DatabaseService(_supabase())
    if not db.get_plant(plant_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found")
    db.delete_plant(plant_id)


# ─── PLANTS ──────────────────────────────────────────────────────────────────

@router.post("/plants", response_model=PlantResponse, status_code=status.HTTP_201_CREATED)
def create_plant(body: PlantCreate) -> PlantResponse:
    db = DatabaseService(_supabase())
    record = db.create_plant(
        name=body.name,
        metadata=body.metadata,
    )
    return PlantResponse(
        id=record["id"],
        created_by=record.get("created_by"),
        name=record["name"],
        status=record["status"],
        metadata=record.get("metadata") or {},
        created_at=datetime.fromisoformat(record["created_at"]),
    )


@router.get("/plants", response_model=PlantListResponse)
def list_plants(name: str | None = None) -> PlantListResponse:
    db = DatabaseService(_supabase())
    records = db.list_plants(name=name)
    items = [
        PlantResponse(
            id=r["id"],
            created_by=r.get("created_by"),
            name=r["name"],
            status=r["status"],
            metadata=r.get("metadata") or {},
            created_at=datetime.fromisoformat(r["created_at"]),
        )
        for r in records
    ]
    return PlantListResponse(items=items, total=len(items))


# ─── LINK ANALYSIS TO PLANT ──────────────────────────────────────────────────

@router.patch("/analyses/{analysis_id}/plant")
def link_analysis_to_plant(analysis_id: str, plant_id: str) -> dict:
    db = DatabaseService(_supabase())
    if not db.get_analysis(analysis_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    db.link_plant_to_analysis(analysis_id, plant_id)
    return {"analysis_id": analysis_id, "plant_id": plant_id}


# ─── PLANT ANALYSIS HISTORY ──────────────────────────────────────────────────

@router.get("/plants/{plant_id}/analyses", response_model=PlantAnalysisHistory)
def list_plant_analyses(plant_id: str) -> PlantAnalysisHistory:
    db = DatabaseService(_supabase())
    records = db.list_plant_analyses(plant_id)
    items = [
        PlantAnalysisItem(
            id=r["id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            device_id=r["device_id"],
            plant_id=r.get("plant_id"),
            image_url=r["image_url"],
            annotated_image_url=r.get("annotated_image_url"),
            maturity_stage=r["maturity_stage"],
            recommendation=r["recommendation"],
            trichome_distribution=r.get("trichome_distribution"),
            stigma_ratios=r.get("stigma_ratios"),
        )
        for r in records
    ]
    return PlantAnalysisHistory(plant_id=plant_id, items=items, total=len(items))
