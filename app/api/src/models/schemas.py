from __future__ import annotations

from datetime import datetime

from cannabis_maturity.models import MaturityStage, StigmaResult, TrichomeResult
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    id: str
    created_at: datetime
    device_id: str
    plant_id: str | None
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
    plant_id: str | None
    image_url: str
    annotated_image_url: str | None
    maturity_stage: MaturityStage
    recommendation: str
    trichome_distribution: dict | None
    stigma_ratios: dict | None


class AnalysisListResponse(BaseModel):
    items: list[AnalysisListItem]
    total: int


class PlantCreate(BaseModel):
    name: str
    metadata: dict = {}


class PlantResponse(BaseModel):
    id: str
    created_by: str | None
    name: str
    status: str
    metadata: dict
    created_at: datetime


class PlantListResponse(BaseModel):
    items: list[PlantResponse]
    total: int


class PlantAnalysisItem(BaseModel):
    id: str
    created_at: datetime
    device_id: str
    plant_id: str | None
    image_url: str
    annotated_image_url: str | None
    maturity_stage: MaturityStage
    recommendation: str
    trichome_distribution: dict | None
    stigma_ratios: dict | None


class PlantAnalysisHistory(BaseModel):
    plant_id: str
    items: list[PlantAnalysisItem]
    total: int
