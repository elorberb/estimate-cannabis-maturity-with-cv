from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage
from sahi.slicing import slice_image
from ultralytics import YOLO

from cannabis_maturity.models import BoundingBox, Detection, TrichomeResult, TrichomeType


def _nms(
    boxes: list[tuple[float, float, float, float, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    if not boxes:
        return []
    arr = np.array(boxes)
    x1, y1, x2, y2, scores = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        inter = (
            np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
            * np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        )
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])) for i in keep]


class TrichomeDetector:
    _CLASSIFICATION_MAP: dict[int, TrichomeType] = {
        0: TrichomeType.AMBER,
        1: TrichomeType.CLEAR,
        2: TrichomeType.CLOUDY,
    }
    _BBOX_MARGIN = 0.25
    _DETECTION_CONF = 0.3

    def __init__(
        self,
        detection_model_path: str,
        classification_model: Any,
        use_sliced_inference: bool = True,
        patch_size: int = 512,
        overlap: float = 0.2,
    ) -> None:
        self._detection_model = YOLO(detection_model_path)
        self._classification_model = classification_model
        self._use_sliced_inference = use_sliced_inference
        self._patch_size = patch_size
        self._overlap = overlap

    def detect(self, image_bgr: np.ndarray) -> TrichomeResult:
        boxes = self._get_boxes_sliced(image_bgr) if self._use_sliced_inference else self._get_boxes_full(image_bgr)
        h, w = image_bgr.shape[:2]

        detections: list[Detection] = []
        distribution: dict[TrichomeType, int] = {t: 0 for t in TrichomeType}

        for x_min, y_min, x_max, y_max in boxes:
            bw, bh = x_max - x_min, y_max - y_min
            cx0 = max(0, int(x_min - bw * self._BBOX_MARGIN))
            cy0 = max(0, int(y_min - bh * self._BBOX_MARGIN))
            cx1 = min(w, int(x_max + bw * self._BBOX_MARGIN))
            cy1 = min(h, int(y_max + bh * self._BBOX_MARGIN))

            crop = image_bgr[cy0:cy1, cx0:cx1]
            cls_result = self._classification_model(crop)
            class_id = int(cls_result[0].probs.data.argmax())
            trichome_type = self._CLASSIFICATION_MAP[class_id]

            detections.append(Detection(
                bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                trichome_type=trichome_type,
                confidence=float(cls_result[0].probs.data.max()),
            ))
            distribution[trichome_type] += 1

        return TrichomeResult(detections=detections, distribution=distribution, total_count=len(detections))

    def _get_boxes_full(self, image_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        results = self._detection_model.predict(source=image_bgr, conf=self._DETECTION_CONF, verbose=False)
        return [tuple(map(float, box)) for box in results[0].boxes.xyxy.cpu().numpy()]

    def _get_boxes_sliced(self, image_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        slice_result = slice_image(
            PILImage.fromarray(image_rgb),
            slice_height=self._patch_size,
            slice_width=self._patch_size,
            overlap_height_ratio=self._overlap,
            overlap_width_ratio=self._overlap,
        )
        patches = [s.image for s in slice_result.sliced_image_list]
        offsets = [s.starting_pixel for s in slice_result.sliced_image_list]

        batch_results = self._detection_model.predict(
            source=patches, conf=self._DETECTION_CONF, verbose=False
        )

        raw_boxes: list[tuple[float, float, float, float, float]] = []
        for result, (ox, oy) in zip(batch_results, offsets):
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                raw_boxes.append((float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy), float(conf)))

        return _nms(raw_boxes)
