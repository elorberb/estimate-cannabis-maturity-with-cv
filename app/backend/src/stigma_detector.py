import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from models import BoundingBox, StigmaDetection, StigmaAnalysisResult

logger = logging.getLogger(__name__)


class StigmaDetector:
    ORANGE_HSV_LOW = np.array([10, 80, 70])
    ORANGE_HSV_HIGH = np.array([25, 255, 255])
    GREEN_HSV_LOW = np.array([35, 40, 40])
    GREEN_HSV_HIGH = np.array([85, 255, 255])

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._model_path = Path(model_path)

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("ultralytics package required")
            self._model = YOLO(str(self._model_path))

    def analyze(self, image: Union[str, Path, np.ndarray]) -> StigmaAnalysisResult:
        self._load_model()

        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image_path = None
            img_bgr = image

        results = self._model(
            img_bgr,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                mask = masks[i]

                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (img_bgr.shape[1], img_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                binary_mask = (mask_resized > 0.5).astype(np.uint8)

                orange_ratio, green_ratio = self._classify_colors(img_bgr, binary_mask)

                bbox = BoundingBox(
                    x_min=float(xyxy[0]),
                    y_min=float(xyxy[1]),
                    x_max=float(xyxy[2]),
                    y_max=float(xyxy[3]),
                )

                detections.append(
                    StigmaDetection(
                        bbox=bbox,
                        confidence=conf,
                        orange_ratio=orange_ratio,
                        green_ratio=green_ratio,
                    )
                )

        return StigmaAnalysisResult(detections=detections, image_path=image_path)

    def _classify_colors(self, img_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        orange_mask = cv2.inRange(img_hsv, self.ORANGE_HSV_LOW, self.ORANGE_HSV_HIGH)
        green_mask = cv2.inRange(img_hsv, self.GREEN_HSV_LOW, self.GREEN_HSV_HIGH)

        orange_in_stigma = cv2.bitwise_and(orange_mask, orange_mask, mask=mask)
        green_in_stigma = cv2.bitwise_and(green_mask, green_mask, mask=mask)

        total_pixels = np.sum(mask > 0)
        if total_pixels == 0:
            return 0.0, 0.0

        orange_pixels = np.sum(orange_in_stigma > 0)
        green_pixels = np.sum(green_in_stigma > 0)

        orange_ratio = orange_pixels / total_pixels
        green_ratio = green_pixels / total_pixels

        total_ratio = orange_ratio + green_ratio
        if total_ratio > 0:
            orange_ratio = orange_ratio / total_ratio
            green_ratio = green_ratio / total_ratio

        return orange_ratio, green_ratio
