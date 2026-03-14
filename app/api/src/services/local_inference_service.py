from __future__ import annotations

import asyncio
import base64

import cv2
import numpy as np
from cannabis_maturity.annotation_renderer import AnnotationRenderer
from cannabis_maturity.stigma_color_classifier import StigmaColorClassifier
from cannabis_maturity.crop_extractor import CropExtractor
from cannabis_maturity.maturity_assessor import MaturityAssessor
from cannabis_maturity.models import AnalysisResult
from cannabis_maturity.stigma_detector import StigmaDetector
from cannabis_maturity.trichome_detector import TrichomeDetector
from ultralytics import YOLO

from services.inference_error import InferenceError


class LocalInferenceService:
    def __init__(
        self,
        detection_model_path: str,
        classification_model_path: str,
        segmentation_model_path: str,
    ) -> None:
        self._detection_model = YOLO(detection_model_path)
        self._classification_model = YOLO(classification_model_path)
        self._segmentation_model = YOLO(segmentation_model_path)

    async def analyze(self, image_bytes: bytes) -> dict:
        return await asyncio.to_thread(self._run_pipeline, image_bytes)

    def _run_pipeline(self, image_bytes: bytes) -> dict:
        image_array = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InferenceError("Could not decode image")

        trichome_result = TrichomeDetector(
            self._detection_model, self._classification_model
        ).detect(image_bgr)

        stigma_result = StigmaDetector(
            self._segmentation_model, StigmaColorClassifier()
        ).detect(image_bgr)

        maturity_stage, recommendation = MaturityAssessor.assess(
            trichome_result.distribution,
            stigma_result.avg_green_ratio,
            stigma_result.avg_orange_ratio,
        )

        annotated_image = AnnotationRenderer.render(image_bgr, trichome_result, stigma_result)
        _, encoded_buffer = cv2.imencode(".jpg", annotated_image)
        annotated_image_b64 = base64.b64encode(encoded_buffer.tobytes()).decode()

        return AnalysisResult(
            trichome_result=trichome_result,
            stigma_result=stigma_result,
            maturity_stage=maturity_stage,
            recommendation=recommendation,
            annotated_image_b64=annotated_image_b64,
            trichome_crops_b64=CropExtractor.extract_trichome_crops(image_bgr, trichome_result),
            stigma_crops_b64=CropExtractor.extract_stigma_crops(image_bgr, stigma_result),
        ).model_dump()
