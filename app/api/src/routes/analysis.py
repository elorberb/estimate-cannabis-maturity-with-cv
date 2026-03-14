from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from models.schemas import AnalyzeResponse
from services.inference_error import InferenceError

router = APIRouter(tags=["analysis"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


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

    try:
        result: dict = await request.app.state.inference_service.analyze(image_bytes)
    except InferenceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    analysis_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    return AnalyzeResponse(
        id=analysis_id,
        created_at=now,
        device_id=device_id,
        image_url="",
        annotated_image_url=None,
        trichome_result=result["trichome_result"],
        stigma_result=result["stigma_result"],
        maturity_stage=result["maturity_stage"],
        recommendation=result["recommendation"],
        trichome_crops_b64=result.get("trichome_crops_b64"),
        stigma_crops_b64=result.get("stigma_crops_b64"),
    )
