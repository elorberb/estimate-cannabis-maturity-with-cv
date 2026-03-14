from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cannabis_maturity.color_classifier import ColorClassifier
from cannabis_maturity.models import BoundingBox, StigmaDetection, StigmaResult


class StigmaDetector:
    def __init__(self, segmentation_model: Any, color_classifier: ColorClassifier) -> None:
        self._segmentation_model = segmentation_model
        self._color_classifier = color_classifier

    def detect(self, image_bgr: np.ndarray) -> StigmaResult:
        results = self._segmentation_model.predict(source=image_bgr, conf=0.3, iou=0.45)
        result = results[0]

        if result.masks is None:
            return StigmaResult(detections=[], avg_green_ratio=0.0, avg_orange_ratio=0.0, total_count=0)

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        detections: list[StigmaDetection] = []

        for i, mask in enumerate(masks):
            resized_mask = cv2.resize(mask, (w, h))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)

            x_min, y_min, x_max, y_max = map(int, boxes[i])
            crop_rgb = segmented[y_min:y_max, x_min:x_max]

            green_ratio, orange_ratio = self._color_classifier.classify(crop_rgb)

            detections.append(
                StigmaDetection(
                    bbox=BoundingBox(x_min=float(x_min), y_min=float(y_min), x_max=float(x_max), y_max=float(y_max)),
                    green_ratio=green_ratio,
                    orange_ratio=orange_ratio,
                )
            )

        total = len(detections)
        avg_green = sum(d.green_ratio for d in detections) / total if total else 0.0
        avg_orange = sum(d.orange_ratio for d in detections) / total if total else 0.0

        return StigmaResult(
            detections=detections,
            avg_green_ratio=avg_green,
            avg_orange_ratio=avg_orange,
            total_count=total,
        )
