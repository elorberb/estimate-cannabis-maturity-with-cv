from __future__ import annotations

import base64
import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from supabase import create_client

from src.config import settings
from src.models.schemas import AnalyzeResponse, DetectionItem, StigmaRatios, TrichomeDistribution
from src.services.database_service import DatabaseService
from src.services.modal_client import InferenceError, ModalClient
from src.services.storage_service import StorageService

router = APIRouter(tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _get_supabase_client():
    return create_client(settings.supabase_url, settings.supabase_service_key)


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(
    file: UploadFile = File(...),
    device_id: str = Form(...),
) -> AnalyzeResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported file type: {file.content_type}. Must be one of {_ALLOWED_CONTENT_TYPES}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="File too large. Maximum size is 20 MB.",
        )

    supabase = _get_supabase_client()
    storage = StorageService(supabase, settings.supabase_storage_bucket)
    db = DatabaseService(supabase)
    modal_client = ModalClient(settings.modal_app_name)

    image_url = storage.upload_image(image_bytes, content_type=file.content_type or "image/jpeg")

    try:
        result_payload = await modal_client.analyze(image_bytes)
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference failed: {e}",
        ) from e

    annotated_image_url: str | None = None
    if result_payload.get("annotated_image_b64"):
        annotated_bytes = base64.b64decode(result_payload["annotated_image_b64"])
        annotated_image_url = storage.upload_image(annotated_bytes, content_type="image/jpeg")

    record = db.save_analysis(device_id, image_url, annotated_image_url, result_payload)

    trichome = result_payload["trichome_result"]
    stigma = result_payload["stigma_result"]
    distribution = trichome["distribution"]

    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.datetime.fromisoformat(record["created_at"]),
        image_url=image_url,
        annotated_image_url=annotated_image_url,
        trichome_distribution=TrichomeDistribution(
            clear=distribution.get("clear", 0),
            cloudy=distribution.get("cloudy", 0),
            amber=distribution.get("amber", 0),
            total=trichome["total_count"],
        ),
        stigma_ratios=StigmaRatios(
            avg_green_ratio=stigma["avg_green_ratio"],
            avg_orange_ratio=stigma["avg_orange_ratio"],
            total_count=stigma["total_count"],
        ),
        maturity_stage=result_payload["maturity_stage"],
        recommendation=result_payload["recommendation"],
        detections=[
            DetectionItem(
                x_min=d["bbox"]["x_min"],
                y_min=d["bbox"]["y_min"],
                x_max=d["bbox"]["x_max"],
                y_max=d["bbox"]["y_max"],
                trichome_type=d["trichome_type"],
                confidence=d["confidence"],
            )
            for d in trichome["detections"]
        ],
        counts={
            "trichome_total": trichome["total_count"],
            "stigma_total": stigma["total_count"],
        },
    )
