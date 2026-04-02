from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from cannabis_maturity.models import BoundingBox, StigmaResult, TrichomeResult


class CropExtractor:
    _THUMBNAIL_SIZE = 128

    @staticmethod
    def extract_trichome_crops(image_bgr: np.ndarray, result: TrichomeResult) -> list[str]:
        return [CropExtractor._crop_to_b64(image_bgr, det.bbox) for det in result.detections]

    @staticmethod
    def extract_stigma_crops(image_bgr: np.ndarray, result: StigmaResult) -> list[str]:
        return [CropExtractor._polygon_crop_to_b64(image_bgr, det.polygon) for det in result.detections]

    @staticmethod
    def _crop_to_b64(image_bgr: np.ndarray, bbox: BoundingBox) -> str:
        crop = image_bgr[int(bbox.y_min) : int(bbox.y_max), int(bbox.x_min) : int(bbox.x_max)]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        pil_img = pil_img.resize((CropExtractor._THUMBNAIL_SIZE, CropExtractor._THUMBNAIL_SIZE))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _polygon_crop_to_b64(image_bgr: np.ndarray, polygon: list[list[int]]) -> str:
        pts = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = image_bgr[y : y + h, x : x + w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        pil_img = pil_img.resize((CropExtractor._THUMBNAIL_SIZE, CropExtractor._THUMBNAIL_SIZE))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
