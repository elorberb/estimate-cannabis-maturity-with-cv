"""Trichome detection and classification using YOLO models."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .models import (
    AnalysisResult,
    BoundingBox,
    TrichomeDetection,
    TrichomeDistribution,
    TrichomeType,
)

logger = logging.getLogger(__name__)


class TrichomeDetector:
    """
    Detects and classifies cannabis trichomes using YOLO models.

    Supports two modes:
    1. Single-stage: One YOLO model that detects and classifies in one pass
    2. Two-stage: Detection model + separate classification model

    Example usage:
        # Single-stage mode (detection model outputs classes directly)
        detector = TrichomeDetector(detection_model_path="yolov8_trichomes.pt")
        result = detector.analyze("cannabis_image.jpg")

        # Two-stage mode (separate detection and classification)
        detector = TrichomeDetector(
            detection_model_path="yolov8_detect.pt",
            classification_model_path="yolov8_classify.pt"
        )
        result = detector.analyze("cannabis_image.jpg")
    """

    # Default class mapping for classification model output
    CLASSIFICATION_TO_TYPE = {
        0: TrichomeType.AMBER,
        1: TrichomeType.CLEAR,
        2: TrichomeType.CLOUDY,
    }

    # Default class mapping for single-stage detection model
    DETECTION_CLASS_TO_TYPE = {
        0: None,  # Generic trichome (needs classification)
        1: TrichomeType.CLEAR,
        2: TrichomeType.CLOUDY,
        3: TrichomeType.AMBER,
    }

    def __init__(
        self,
        detection_model_path: Union[str, Path],
        classification_model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.5,
        device: str = "cuda:0",
        bbox_extension_margin: float = 0.25,
    ):
        """
        Initialize the trichome detector.

        Args:
            detection_model_path: Path to YOLO detection model (.pt file)
            classification_model_path: Optional path to YOLO classification model.
                If None, assumes detection model outputs class labels directly.
            confidence_threshold: Minimum confidence for detections (default: 0.5)
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
            bbox_extension_margin: Margin ratio for extending bbox before classification
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.bbox_extension_margin = bbox_extension_margin
        self._detection_model = None
        self._classification_model = None
        self._detection_model_path = Path(detection_model_path)
        self._classification_model_path = (
            Path(classification_model_path) if classification_model_path else None
        )

    def _load_models(self) -> None:
        """Lazy load YOLO models."""
        if self._detection_model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "ultralytics package is required. Install with: pip install ultralytics"
                )

            logger.info(f"Loading detection model from {self._detection_model_path}")
            self._detection_model = YOLO(str(self._detection_model_path))

            if self._classification_model_path:
                logger.info(
                    f"Loading classification model from {self._classification_model_path}"
                )
                self._classification_model = YOLO(str(self._classification_model_path))

    @property
    def is_two_stage(self) -> bool:
        """Return True if using two-stage detection+classification."""
        return self._classification_model_path is not None

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        filter_large_objects: bool = True,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.7,
    ) -> AnalysisResult:
        """
        Analyze an image to detect and classify trichomes.

        Args:
            image: Image path or numpy array (BGR format)
            filter_large_objects: Remove outlier detections based on size
            apply_nms: Apply non-maximum suppression
            nms_iou_threshold: IoU threshold for NMS

        Returns:
            AnalysisResult containing detections and distribution
        """
        self._load_models()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image_path = None
            img_array = image

        image_height, image_width = img_array.shape[:2]

        # Run detection
        detections = self._detect(img_array)

        # Filter large objects (outliers)
        if filter_large_objects and detections:
            detections = self._filter_large_objects(detections)

        # Apply NMS
        if apply_nms and detections:
            detections = self._non_max_suppression(detections, nms_iou_threshold)

        # Run classification if two-stage mode
        if self.is_two_stage and detections:
            detections = self._classify(img_array, detections, image_width, image_height)

        # Build result
        distribution = TrichomeDistribution.from_detections(detections)
        return AnalysisResult(
            detections=detections,
            distribution=distribution,
            image_path=image_path,
        )

    def _detect(self, image: np.ndarray) -> list[TrichomeDetection]:
        """Run YOLO detection on image."""
        results = self._detection_model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                bbox = BoundingBox(
                    x_min=float(xyxy[0]),
                    y_min=float(xyxy[1]),
                    x_max=float(xyxy[2]),
                    y_max=float(xyxy[3]),
                )

                # Map class ID to trichome type
                trichome_type = self.DETECTION_CLASS_TO_TYPE.get(cls_id)
                if trichome_type is None:
                    # Generic trichome, will need classification
                    trichome_type = TrichomeType.CLOUDY  # Default, will be updated

                detections.append(
                    TrichomeDetection(
                        bbox=bbox,
                        trichome_type=trichome_type,
                        confidence=conf,
                    )
                )

        logger.info(f"Detected {len(detections)} trichomes")
        return detections

    def _classify(
        self,
        image: np.ndarray,
        detections: list[TrichomeDetection],
        image_width: int,
        image_height: int,
    ) -> list[TrichomeDetection]:
        """Classify each detected trichome using the classification model."""
        classified = []

        for detection in detections:
            # Extend bounding box
            extended_bbox = detection.bbox.extend(
                self.bbox_extension_margin, image_width, image_height
            )

            # Crop image
            crop = image[
                int(extended_bbox.y_min) : int(extended_bbox.y_max),
                int(extended_bbox.x_min) : int(extended_bbox.x_max),
            ]

            if crop.size == 0:
                continue

            # Classify
            results = self._classification_model(crop, verbose=False)
            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    cls_id = int(probs.data.argmax().cpu().numpy())
                    trichome_type = self.CLASSIFICATION_TO_TYPE.get(
                        cls_id, TrichomeType.CLOUDY
                    )

                    classified.append(
                        TrichomeDetection(
                            bbox=detection.bbox,
                            trichome_type=trichome_type,
                            confidence=detection.confidence,
                        )
                    )

        logger.info(f"Classified {len(classified)} trichomes")
        return classified

    def _filter_large_objects(
        self, detections: list[TrichomeDetection], size_threshold_ratio: float = 10.0
    ) -> list[TrichomeDetection]:
        """Filter out unusually large detections (likely false positives)."""
        if not detections:
            return detections

        areas = [d.bbox.area for d in detections]
        median_area = np.median(areas)
        threshold = median_area * size_threshold_ratio

        filtered = [d for d in detections if d.bbox.area <= threshold]
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered {removed} large objects")
        return filtered

    def _non_max_suppression(
        self, detections: list[TrichomeDetection], iou_threshold: float
    ) -> list[TrichomeDetection]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if not detections:
            return detections

        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        while sorted_detections:
            best = sorted_detections.pop(0)
            keep.append(best)

            # Filter remaining based on IoU
            sorted_detections = [
                d
                for d in sorted_detections
                if self._compute_iou(best.bbox, d.bbox) < iou_threshold
            ]

        removed = len(detections) - len(keep)
        if removed > 0:
            logger.info(f"NMS removed {removed} overlapping detections")
        return keep

    @staticmethod
    def _compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1.x_min, box2.x_min)
        y1 = max(box1.y_min, box2.y_min)
        x2 = min(box1.x_max, box2.x_max)
        y2 = min(box1.y_max, box2.y_max)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
