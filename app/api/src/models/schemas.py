from __future__ import annotations

from datetime import datetime

from cannabis_maturity.models import MaturityStage, StigmaResult, TrichomeResult
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    id: str
    created_at: datetime
    device_id: str
    image_url: str
    annotated_image_url: str | None
    trichome_result: TrichomeResult
    stigma_result: StigmaResult
    maturity_stage: MaturityStage
    recommendation: str
    trichome_crops_b64: list[str] | None
    stigma_crops_b64: list[str] | None


class AnalysisListItem(BaseModel):
    id: str
    created_at: datetime
    device_id: str
    image_url: str
    maturity_stage: MaturityStage
    recommendation: str


class AnalysisListResponse(BaseModel):
    items: list[AnalysisListItem]
    total: int
