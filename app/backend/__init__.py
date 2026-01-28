"""
Trichome Backend Package

A simple YOLO-based backend for detecting and classifying cannabis trichomes
to calculate maturity distribution.

Example usage:
    from app.backend import TrichomeDetector

    # Initialize detector with YOLO model
    detector = TrichomeDetector(
        detection_model_path="path/to/yolo_model.pt",
        classification_model_path="path/to/classification_model.pt",  # Optional
    )

    # Analyze an image
    result = detector.analyze("cannabis_image.jpg")

    # Get distribution
    print(result.distribution.to_dict())
    # {'counts': {'clear': 45, 'cloudy': 120, 'amber': 35, 'total': 200},
    #  'percentages': {'clear': 22.5, 'cloudy': 60.0, 'amber': 17.5}}

    # Get maturity assessment
    from app.backend import get_maturity_assessment
    assessment = get_maturity_assessment(result.distribution)
    print(assessment['recommendation'])
"""

__version__ = "0.1.0"

from app.backend.models import (
    AnalysisResult,
    BoundingBox,
    TrichomeDetection,
    TrichomeDistribution,
    TrichomeType,
)
from app.backend.detector import TrichomeDetector
from app.backend.distribution import (
    aggregate_distributions,
    aggregate_results,
    get_maturity_assessment,
    load_distribution,
    save_distribution,
    save_result,
)
from app.backend.config import DetectorConfig, CLASS_MAPPINGS, CLASS_ID_TO_NAME
from app.backend.utils import (
    draw_detections,
    load_image,
    save_image,
    visualize_result,
    validate_image,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "AnalysisResult",
    "BoundingBox",
    "TrichomeDetection",
    "TrichomeDistribution",
    "TrichomeType",
    # Detector
    "TrichomeDetector",
    # Distribution
    "aggregate_distributions",
    "aggregate_results",
    "get_maturity_assessment",
    "load_distribution",
    "save_distribution",
    "save_result",
    # Config
    "DetectorConfig",
    "CLASS_MAPPINGS",
    "CLASS_ID_TO_NAME",
    # Utils
    "draw_detections",
    "load_image",
    "save_image",
    "visualize_result",
    "validate_image",
]
