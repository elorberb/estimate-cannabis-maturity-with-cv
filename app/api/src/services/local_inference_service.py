from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from cannabis_maturity.annotation_renderer import AnnotationRenderer
from cannabis_maturity.crop_extractor import CropExtractor
from cannabis_maturity.maturity_assessor import MaturityAssessor
from cannabis_maturity.models import AnalysisResult
from cannabis_maturity.stigma_color_classifier import StigmaColorClassifier
from cannabis_maturity.stigma_detector import StigmaDetector
from cannabis_maturity.trichome_detector import TrichomeDetector
from ultralytics import YOLO

from services.inference_error import InferenceError

logger = logging.getLogger(__name__)


class LocalInferenceService:
    def __init__(
        self,
        detection_model_path: str,
        classification_model_path: str,
        segmentation_model_path: str,
        trichome_patch_size: int = 512,
        trichome_overlap: float = 0.2,
        debug_save_results: bool = False,
        debug_output_dir: str = "inference_samples",
    ) -> None:
        classification_model = YOLO(classification_model_path)
        segmentation_model = YOLO(segmentation_model_path)

        self._trichome_detector = TrichomeDetector(
            detection_model_path=detection_model_path,
            classification_model=classification_model,
            patch_size=trichome_patch_size,
            overlap=trichome_overlap,
        )
        self._stigma_detector = StigmaDetector(segmentation_model, StigmaColorClassifier())
        self._debug_save_results = debug_save_results
        self._debug_output_dir = Path(debug_output_dir)

    async def analyze(self, image_bytes: bytes) -> dict:
        return await asyncio.to_thread(self._run_pipeline, image_bytes)

    def _run_pipeline(self, image_bytes: bytes) -> dict:
        image_array = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise InferenceError("Could not decode image")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            trichome_future = executor.submit(self._trichome_detector.detect, image_bgr)
            stigma_future = executor.submit(self._stigma_detector.detect, image_bgr)
            trichome_result = trichome_future.result()
            stigma_result = stigma_future.result()
        t1 = time.perf_counter()

        maturity_stage, recommendation = MaturityAssessor.assess(
            trichome_result.distribution,
            stigma_result.avg_green_ratio,
            stigma_result.avg_orange_ratio,
        )
        t2 = time.perf_counter()

        annotated_image = AnnotationRenderer.render(image_bgr, trichome_result, stigma_result)
        _, encoded_buffer = cv2.imencode(".jpg", annotated_image)
        annotated_image_b64 = base64.b64encode(encoded_buffer.tobytes()).decode()
        t3 = time.perf_counter()

        logger.info(
            "[inference] total=%.2fs  detection_parallel=%.2fs  maturity=%.3fs  annotate=%.2fs",
            t3 - t0, t1 - t0, t2 - t1, t3 - t2,
        )

        result = AnalysisResult(
            trichome_result=trichome_result,
            stigma_result=stigma_result,
            maturity_stage=maturity_stage,
            recommendation=recommendation,
            annotated_image_b64=annotated_image_b64,
            trichome_crops_b64=CropExtractor.extract_trichome_crops(image_bgr, trichome_result),
            stigma_crops_b64=CropExtractor.extract_stigma_crops(image_bgr, stigma_result),
        )

        if self._debug_save_results:
            self._save_debug_output(image_bgr, annotated_image, result)

        return result.model_dump()

    def _save_debug_output(
        self,
        image_bgr: np.ndarray,
        annotated_image: np.ndarray,
        result: AnalysisResult,
    ) -> None:
        run_dir = self._debug_output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (run_dir / "trichomes").mkdir(parents=True, exist_ok=True)
        (run_dir / "stigmas").mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(run_dir / "original.jpg"), image_bgr)
        cv2.imwrite(str(run_dir / "annotated.jpg"), annotated_image)

        for i, det in enumerate(result.trichome_result.detections):
            crop = image_bgr[int(det.bbox.y_min):int(det.bbox.y_max), int(det.bbox.x_min):int(det.bbox.x_max)]
            cv2.imwrite(str(run_dir / "trichomes" / f"{i}_{det.trichome_type.value}.jpg"), crop)

        for i, det in enumerate(result.stigma_result.detections):
            crop = image_bgr[int(det.bbox.y_min):int(det.bbox.y_max), int(det.bbox.x_min):int(det.bbox.x_max)]
            cv2.imwrite(str(run_dir / "stigmas" / f"{i}.jpg"), crop)

        summary = result.model_dump(exclude={"annotated_image_b64", "trichome_crops_b64", "stigma_crops_b64"})
        (run_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
