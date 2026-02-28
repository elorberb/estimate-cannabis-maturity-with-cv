"""Configuration for the trichome backend package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for TrichomeDetector."""

    detection_model_path: Optional[Path] = None
    classification_model_path: Optional[Path] = None
    confidence_threshold: float = 0.5
    device: str = "cuda:0"
    bbox_extension_margin: float = 0.25
    apply_nms: bool = True
    nms_iou_threshold: float = 0.7
    filter_large_objects: bool = True
    large_object_threshold_ratio: float = 10.0

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "detection_model_path": str(self.detection_model_path)
            if self.detection_model_path
            else None,
            "classification_model_path": str(self.classification_model_path)
            if self.classification_model_path
            else None,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "bbox_extension_margin": self.bbox_extension_margin,
            "apply_nms": self.apply_nms,
            "nms_iou_threshold": self.nms_iou_threshold,
            "filter_large_objects": self.filter_large_objects,
            "large_object_threshold_ratio": self.large_object_threshold_ratio,
        }


# Default class mappings (matching the thesis codebase)
CLASS_MAPPINGS = {
    "trichome": 0,  # Generic detection class
    "clear": 1,
    "cloudy": 2,
    "amber": 3,
}

# Reverse mapping
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_MAPPINGS.items()}

# Classification model output mapping (YOLO classification model)
CLASSIFICATION_OUTPUT_MAPPING = {
    0: "amber",
    1: "clear",
    2: "cloudy",
}

# Detection model class mapping (for single-stage models)
DETECTION_OUTPUT_MAPPING = {
    0: "trichome",  # Generic - needs classification
    1: "clear",
    2: "cloudy",
    3: "amber",
}
