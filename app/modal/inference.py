"""Modal GPU inference for cannabis maturity analysis."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import modal
import numpy as np

from constants import (
    CLASSIFICATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    SEGMENTATION_MODEL_PATH,
    VOLUME_NAME,
)

app = modal.App("cannabis-maturity-inference")

_BACKEND_SRC = Path(__file__).parent.parent / "backend" / "src"

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pydantic>=2.0.0",
    )
    .add_local_dir(str(_BACKEND_SRC), remote_path="/root")
)

with inference_image.imports():
    from cannabis_maturity.annotation_renderer import AnnotationRenderer
    from cannabis_maturity.color_classifier import ColorClassifier
    from cannabis_maturity.crop_extractor import CropExtractor
    from cannabis_maturity.maturity_assessor import MaturityAssessor
    from cannabis_maturity.models import AnalysisResult
    from cannabis_maturity.stigma_detector import StigmaDetector
    from cannabis_maturity.trichome_detector import TrichomeDetector
    from ultralytics import YOLO

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.cls(
    gpu="T4",
    image=inference_image,
    volumes={"/models": model_volume},
)
class MaturityAnalyzer:
    @modal.enter()
    def load_models(self) -> None:
        try:
            self._detection_model = YOLO(DETECTION_MODEL_PATH)
            self._classification_model = YOLO(CLASSIFICATION_MODEL_PATH)
            self._segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)
        except Exception as e:
            self._detection_model = None
            self._classification_model = None
            self._segmentation_model = None
            print(f"Warning: Could not load model weights: {e}")
            print("Running without model weights — inference will fail gracefully.")

    @modal.method()
    def analyze(self, image_bytes: bytes) -> dict:
        if self._detection_model is None:
            raise RuntimeError(
                "Model weights not loaded. Upload weights to the cannabis-maturity-model-weights volume."
            )

        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image bytes")

        trichome_detector = TrichomeDetector(self._detection_model, self._classification_model)
        trichome_result = trichome_detector.detect(image_bgr)

        color_classifier = ColorClassifier()
        stigma_detector = StigmaDetector(self._segmentation_model, color_classifier)
        stigma_result = stigma_detector.detect(image_bgr)

        stage, recommendation = MaturityAssessor.assess(
            trichome_result.distribution,
            stigma_result.avg_green_ratio,
            stigma_result.avg_orange_ratio,
        )

        annotated = AnnotationRenderer.render(image_bgr, trichome_result, stigma_result)
        trichome_crops = CropExtractor.extract_trichome_crops(image_bgr, trichome_result)
        stigma_crops = CropExtractor.extract_stigma_crops(image_bgr, stigma_result)

        _, buf = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buf.tobytes()).decode()

        return AnalysisResult(
            trichome_result=trichome_result,
            stigma_result=stigma_result,
            maturity_stage=stage,
            recommendation=recommendation,
            annotated_image_b64=annotated_b64,
            trichome_crops_b64=trichome_crops,
            stigma_crops_b64=stigma_crops,
        ).model_dump()


@app.local_entrypoint()
def main() -> None:
    """Smoke test — run with: modal run app/modal/inference.py"""
    print("Modal inference app loaded successfully.")
    print("To run inference, deploy and call MaturityAnalyzer.analyze.remote(image_bytes)")
