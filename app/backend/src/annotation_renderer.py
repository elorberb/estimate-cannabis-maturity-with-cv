from __future__ import annotations

import cv2
import numpy as np

from src.models import StigmaResult, TrichomeResult, TrichomeType


class AnnotationRenderer:
    _TRICHOME_COLORS: dict[TrichomeType, tuple[int, int, int]] = {
        TrichomeType.CLEAR: (200, 200, 200),
        TrichomeType.CLOUDY: (255, 255, 255),
        TrichomeType.AMBER: (0, 165, 255),
    }
    _STIGMA_COLOR: tuple[int, int, int] = (0, 255, 0)

    @staticmethod
    def render(
        image_bgr: np.ndarray,
        trichome_result: TrichomeResult,
        stigma_result: StigmaResult,
    ) -> np.ndarray:
        annotated = image_bgr.copy()

        for det in trichome_result.detections:
            color = AnnotationRenderer._TRICHOME_COLORS[det.trichome_type]
            pt1 = (int(det.bbox.x_min), int(det.bbox.y_min))
            pt2 = (int(det.bbox.x_max), int(det.bbox.y_max))
            cv2.rectangle(annotated, pt1, pt2, color, 2)

        for det in stigma_result.detections:
            pts = np.array(det.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=AnnotationRenderer._STIGMA_COLOR, thickness=2)

        return annotated
