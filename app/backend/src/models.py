from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, computed_field, model_validator


class TrichomeType(IntEnum):
    CLEAR = 1
    CLOUDY = 2
    AMBER = 3


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @computed_field
    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @computed_field
    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field
    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def extend(self, margin: float, image_width: int, image_height: int) -> "BoundingBox":
        margin_w = int(margin * self.width)
        margin_h = int(margin * self.height)
        return BoundingBox(
            x_min=max(0, self.x_min - margin_w),
            y_min=max(0, self.y_min - margin_h),
            x_max=min(image_width, self.x_max + margin_w),
            y_max=min(image_height, self.y_max + margin_h),
        )


class TrichomeDetection(BaseModel):
    bbox: BoundingBox
    trichome_type: TrichomeType
    confidence: float

    def to_dict(self) -> dict:
        return {
            "bbox": {
                "x_min": self.bbox.x_min,
                "y_min": self.bbox.y_min,
                "x_max": self.bbox.x_max,
                "y_max": self.bbox.y_max,
            },
            "type": self.trichome_type.name.lower(),
            "type_id": self.trichome_type.value,
            "confidence": self.confidence,
        }


class TrichomeDistribution(BaseModel):
    clear_count: int = 0
    cloudy_count: int = 0
    amber_count: int = 0

    @computed_field
    @property
    def total_count(self) -> int:
        return self.clear_count + self.cloudy_count + self.amber_count

    @computed_field
    @property
    def clear_ratio(self) -> float:
        return self.clear_count / self.total_count if self.total_count > 0 else 0.0

    @computed_field
    @property
    def cloudy_ratio(self) -> float:
        return self.cloudy_count / self.total_count if self.total_count > 0 else 0.0

    @computed_field
    @property
    def amber_ratio(self) -> float:
        return self.amber_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "counts": {
                "clear": self.clear_count,
                "cloudy": self.cloudy_count,
                "amber": self.amber_count,
                "total": self.total_count,
            },
            "percentages": {
                "clear": round(self.clear_ratio * 100, 2),
                "cloudy": round(self.cloudy_ratio * 100, 2),
                "amber": round(self.amber_ratio * 100, 2),
            },
        }

    @classmethod
    def from_detections(cls, detections: list["TrichomeDetection"]) -> "TrichomeDistribution":
        clear = sum(1 for d in detections if d.trichome_type == TrichomeType.CLEAR)
        cloudy = sum(1 for d in detections if d.trichome_type == TrichomeType.CLOUDY)
        amber = sum(1 for d in detections if d.trichome_type == TrichomeType.AMBER)
        return cls(clear_count=clear, cloudy_count=cloudy, amber_count=amber)


class AnalysisResult(BaseModel):
    detections: list[TrichomeDetection] = []
    distribution: Optional[TrichomeDistribution] = None
    image_path: Optional[str] = None

    @model_validator(mode="after")
    def compute_distribution(self) -> "AnalysisResult":
        if self.distribution is None and self.detections:
            self.distribution = TrichomeDistribution.from_detections(self.detections)
        return self

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "total_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "distribution": self.distribution.to_dict() if self.distribution else None,
        }


class StigmaDetection(BaseModel):
    bbox: BoundingBox
    confidence: float
    orange_ratio: float = 0.0
    green_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bbox": {
                "x_min": self.bbox.x_min,
                "y_min": self.bbox.y_min,
                "x_max": self.bbox.x_max,
                "y_max": self.bbox.y_max,
            },
            "confidence": self.confidence,
            "orange_ratio": round(self.orange_ratio, 4),
            "green_ratio": round(self.green_ratio, 4),
        }


class StigmaAnalysisResult(BaseModel):
    detections: list[StigmaDetection] = []
    overall_orange_ratio: float = 0.0
    overall_green_ratio: float = 0.0
    image_path: Optional[str] = None

    @model_validator(mode="after")
    def compute_overall_ratios(self) -> "StigmaAnalysisResult":
        if self.detections and self.overall_orange_ratio == 0.0 and self.overall_green_ratio == 0.0:
            total_orange = sum(d.orange_ratio for d in self.detections)
            total_green = sum(d.green_ratio for d in self.detections)
            n = len(self.detections)
            self.overall_orange_ratio = total_orange / n if n > 0 else 0.0
            self.overall_green_ratio = total_green / n if n > 0 else 0.0
        return self

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_stigmas": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "overall_orange_ratio": round(self.overall_orange_ratio, 4),
            "overall_green_ratio": round(self.overall_green_ratio, 4),
        }
