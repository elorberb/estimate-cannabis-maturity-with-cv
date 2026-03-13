from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from supabase import create_client

from config import settings
from models.schemas import AnalyzeResponse
from services.database_service import DatabaseService
from services.inference_error import InferenceError
from services.storage_service import StorageService

router = APIRouter(tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _supabase():
    return create_client(settings.supabase_url, settings.supabase_service_key)


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_image(request: Request, file: UploadFile, device_id: str) -> AnalyzeResponse:
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

    record = db.save_analysis(device_id, image_url, annotated_image_url, result)

    return AnalyzeResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        device_id=device_id,
        image_url=image_url,
        annotated_image_url=annotated_image_url,
        trichome_result=result["trichome_result"],
        stigma_result=result["stigma_result"],
        maturity_stage=result["maturity_stage"],
        recommendation=result["recommendation"],
        trichome_crops_b64=result.get("trichome_crops_b64"),
        stigma_crops_b64=result.get("stigma_crops_b64"),
    )
