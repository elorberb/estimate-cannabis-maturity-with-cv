from __future__ import annotations

import datetime

from pydantic import BaseModel


class TrichomeDistribution(BaseModel):
    clear: int
    cloudy: int
    amber: int
    total: int


class StigmaRatios(BaseModel):
    avg_green_ratio: float
    avg_orange_ratio: float
    total_count: int


class DetectionItem(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    trichome_type: str  # "clear" | "cloudy" | "amber"
    confidence: float


class AnalyzeResponse(BaseModel):
    id: str
    created_at: datetime.datetime
    image_url: str
    annotated_image_url: str | None
    trichome_distribution: TrichomeDistribution
    stigma_ratios: StigmaRatios
    maturity_stage: str
    recommendation: str
    detections: list[DetectionItem]
    counts: dict[str, int]


class AnalysisListItem(BaseModel):
    id: str
    created_at: datetime.datetime
    maturity_stage: str
    recommendation: str
    image_url: str


class AnalysisListResponse(BaseModel):
    items: list[AnalysisListItem]
    total: int


class HealthResponse(BaseModel):
    status: str
