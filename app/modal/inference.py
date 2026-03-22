from __future__ import annotations

import base64
import json
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

_MODAL_DIR = Path(__file__).parent
_BACKEND_SRC = Path(__file__).parent.parent / "backend" / "src"

inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pydantic>=2.0.0",
        "sahi>=0.11.0",
    )
    .add_local_dir(str(_BACKEND_SRC), remote_path="/root")
    .add_local_file(str(_MODAL_DIR / "constants.py"), remote_path="/root/constants.py")
)

with inference_image.imports():
    from cannabis_maturity.annotation_renderer import AnnotationRenderer
    from cannabis_maturity.crop_extractor import CropExtractor
    from cannabis_maturity.maturity_assessor import MaturityAssessor
    from cannabis_maturity.models import AnalysisResult
    from cannabis_maturity.stigma_color_classifier import StigmaColorClassifier
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
            classification_model = YOLO(CLASSIFICATION_MODEL_PATH)
            segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)
            self._trichome_detector = TrichomeDetector(
                detection_model_path=DETECTION_MODEL_PATH,
                classification_model=classification_model,
                patch_size=512,
                overlap=0.2,
            )
            self._stigma_detector = StigmaDetector(segmentation_model, StigmaColorClassifier())
        except Exception as e:
            self._trichome_detector = None
            self._stigma_detector = None
            print(f"Warning: Could not load model weights: {e}")

    @modal.method()
    def analyze(self, image_bytes: bytes) -> dict:
        if self._trichome_detector is None:
            raise RuntimeError(
                "Model weights not loaded. Upload weights to the cannabis-maturity-model-weights volume."
            )

        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image bytes")

        trichome_result = self._trichome_detector.detect(image_bgr)
        stigma_result = self._stigma_detector.detect(image_bgr)

        stage, recommendation = MaturityAssessor.assess(
            trichome_result.distribution,
            stigma_result.avg_green_ratio,
            stigma_result.avg_orange_ratio,
        )

        annotated = AnnotationRenderer.render(image_bgr, trichome_result, stigma_result)
        _, buf = cv2.imencode(".jpg", annotated)

        return AnalysisResult(
            trichome_result=trichome_result,
            stigma_result=stigma_result,
            maturity_stage=stage,
            recommendation=recommendation,
            annotated_image_b64=base64.b64encode(buf.tobytes()).decode(),
            trichome_crops_b64=CropExtractor.extract_trichome_crops(image_bgr, trichome_result),
            stigma_crops_b64=CropExtractor.extract_stigma_crops(image_bgr, stigma_result),
        ).model_dump(mode="json")


@app.local_entrypoint()
def main() -> None:
    print("Modal inference app loaded successfully.")
    print("To run inference, deploy and call MaturityAnalyzer.analyze.remote(image_bytes)")
