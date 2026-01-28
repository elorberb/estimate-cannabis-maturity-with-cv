"""Data models for trichome detection and classification."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TrichomeType(IntEnum):
    """Trichome maturity classification types."""
    CLEAR = 1
    CLOUDY = 2
    AMBER = 3


@dataclass
class BoundingBox:
    """Represents a bounding box for a detected object."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def extend(self, margin: float, image_width: int, image_height: int) -> "BoundingBox":
        """Return a new bounding box extended by the given margin ratio."""
        margin_w = int(margin * self.width)
        margin_h = int(margin * self.height)
        return BoundingBox(
            x_min=max(0, self.x_min - margin_w),
            y_min=max(0, self.y_min - margin_h),
            x_max=min(image_width, self.x_max + margin_w),
            y_max=min(image_height, self.y_max + margin_h),
        )


@dataclass
class TrichomeDetection:
    """Represents a single detected trichome."""
    bbox: BoundingBox
    trichome_type: TrichomeType
    confidence: float

    def to_dict(self) -> dict:
        """Convert detection to dictionary format."""
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


@dataclass
class TrichomeDistribution:
    """Represents the distribution of trichome types in an image."""
    clear_count: int = 0
    cloudy_count: int = 0
    amber_count: int = 0

    @property
    def total_count(self) -> int:
        return self.clear_count + self.cloudy_count + self.amber_count

    @property
    def clear_ratio(self) -> float:
        return self.clear_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def cloudy_ratio(self) -> float:
        return self.cloudy_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def amber_ratio(self) -> float:
        return self.amber_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert distribution to dictionary format."""
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
    def from_detections(cls, detections: list[TrichomeDetection]) -> "TrichomeDistribution":
        """Create distribution from a list of detections."""
        clear = sum(1 for d in detections if d.trichome_type == TrichomeType.CLEAR)
        cloudy = sum(1 for d in detections if d.trichome_type == TrichomeType.CLOUDY)
        amber = sum(1 for d in detections if d.trichome_type == TrichomeType.AMBER)
        return cls(clear_count=clear, cloudy_count=cloudy, amber_count=amber)


@dataclass
class AnalysisResult:
    """Complete result of trichome analysis for an image."""
    detections: list[TrichomeDetection] = field(default_factory=list)
    distribution: Optional[TrichomeDistribution] = None
    image_path: Optional[str] = None

    def __post_init__(self):
        if self.distribution is None and self.detections:
            self.distribution = TrichomeDistribution.from_detections(self.detections)

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "image_path": self.image_path,
            "total_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "distribution": self.distribution.to_dict() if self.distribution else None,
        }
