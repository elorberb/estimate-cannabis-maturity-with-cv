from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cannabis_maturity.models import StigmaDetection, StigmaResult
from cannabis_maturity.stigma_color_classifier import StigmaColorClassifier


class StigmaDetector:
    def __init__(self, segmentation_model: Any, color_classifier: StigmaColorClassifier) -> None:
        self._segmentation_model = segmentation_model
        self._color_classifier = color_classifier

    def detect(self, image_bgr: np.ndarray) -> StigmaResult:
        results = self._segmentation_model.predict(source=image_bgr, conf=0.3, iou=0.45)
        result = results[0]

        if result.masks is None:
            return StigmaResult(detections=[], avg_green_ratio=0.0, avg_orange_ratio=0.0, total_count=0)

        masks = result.masks.data.cpu().numpy()
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        detections: list[StigmaDetection] = []

        for mask in masks:
            resized_mask = cv2.resize(mask, (w, h))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            pts = largest.squeeze()
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            polygon: list[list[int]] = pts.tolist()

            x, y, cw, ch = cv2.boundingRect(largest)
            segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
            crop_rgb = segmented[y : y + ch, x : x + cw]

            green_ratio, orange_ratio = self._color_classifier.classify(crop_rgb)

            detections.append(
                StigmaDetection(
                    polygon=polygon,
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
