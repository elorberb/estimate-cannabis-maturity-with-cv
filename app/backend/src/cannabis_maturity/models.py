from __future__ import annotations

import enum

from pydantic import BaseModel


class TrichomeType(str, enum.Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    AMBER = "amber"


class MaturityStage(str, enum.Enum):
    EARLY = "early"
    DEVELOPING = "developing"
    PEAK = "peak"
    MATURE = "mature"
    LATE = "late"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    bbox: BoundingBox
    trichome_type: TrichomeType
    confidence: float


class TrichomeResult(BaseModel):
    detections: list[Detection]
    distribution: dict[TrichomeType, int]
    total_count: int


class StigmaDetection(BaseModel):
    polygon: list[list[int]]
    green_ratio: float
    orange_ratio: float


class StigmaResult(BaseModel):
    detections: list[StigmaDetection]
    avg_green_ratio: float
    avg_orange_ratio: float
    total_count: int


class AnalysisResult(BaseModel):
    trichome_result: TrichomeResult
    stigma_result: StigmaResult
    maturity_stage: MaturityStage
    recommendation: str
    annotated_image_b64: str | None = None
    trichome_crops_b64: list[str] | None = None
    stigma_crops_b64: list[str] | None = None
