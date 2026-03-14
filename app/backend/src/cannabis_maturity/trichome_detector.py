from __future__ import annotations

from typing import Any

import numpy as np

from cannabis_maturity.models import BoundingBox, Detection, TrichomeResult, TrichomeType


class TrichomeDetector:
    _CLASSIFICATION_MAP: dict[int, TrichomeType] = {
        0: TrichomeType.AMBER,
        1: TrichomeType.CLEAR,
        2: TrichomeType.CLOUDY,
    }
    _BBOX_MARGIN = 0.25

    def __init__(self, detection_model: Any, classification_model: Any) -> None:
        self._detection_model = detection_model
        self._classification_model = classification_model

    def detect(self, image_bgr: np.ndarray) -> TrichomeResult:
        results = self._detection_model.predict(source=image_bgr, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        h, w = image_bgr.shape[:2]

        detections: list[Detection] = []
        distribution: dict[TrichomeType, int] = {t: 0 for t in TrichomeType}

        for box in boxes:
            x_min, y_min, x_max, y_max = map(float, box)
            bw = x_max - x_min
            bh = y_max - y_min
            x_min_ext = max(0, int(x_min - bw * self._BBOX_MARGIN))
            y_min_ext = max(0, int(y_min - bh * self._BBOX_MARGIN))
            x_max_ext = min(w, int(x_max + bw * self._BBOX_MARGIN))
            y_max_ext = min(h, int(y_max + bh * self._BBOX_MARGIN))

            crop = image_bgr[y_min_ext:y_max_ext, x_min_ext:x_max_ext]
            cls_results = self._classification_model(crop)
            class_id = int(cls_results[0].probs.data.argmax())
            trichome_type = self._CLASSIFICATION_MAP[class_id]

            detections.append(
                Detection(
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    trichome_type=trichome_type,
                    confidence=float(cls_results[0].probs.data.max()),
                )
            )
            distribution[trichome_type] += 1

        return TrichomeResult(
            detections=detections,
            distribution=distribution,
            total_count=len(detections),
        )
