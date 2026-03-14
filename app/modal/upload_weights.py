"""Upload model weight files to the Modal volume.

Usage (run from repo root):
    cd app/modal && uv run modal run upload_weights.py

Uploads:
    checkpoints/trichome_detection/yolov9_best.pt             -> trichome_detection.pt
    checkpoints/trichome_classification/yolov8/large_fold0.pt -> trichome_classification.pt
    checkpoints/stigma_segmentation/yolov8s_best.pt           -> stigma_segmentation.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

from constants import (
    CLASSIFICATION_MODEL_PATH,
    DETECTION_MODEL_PATH,
    SEGMENTATION_MODEL_PATH,
    VOLUME_NAME,
)

_REPO_ROOT = Path(__file__).parent.parent.parent

app = modal.App("cannabis-maturity-upload-weights")


class WeightUploader:
    _WEIGHTS: list[tuple[Path, str]] = [
        (
            _REPO_ROOT / "checkpoints/trichome_detection/yolov9_best.pt",
            DETECTION_MODEL_PATH.lstrip("/"),
        ),
        (
            _REPO_ROOT / "checkpoints/trichome_classification/yolov8/large_fold0.pt",
            CLASSIFICATION_MODEL_PATH.lstrip("/"),
        ),
        (
            _REPO_ROOT / "checkpoints/stigma_segmentation/yolov8s_best.pt",
            SEGMENTATION_MODEL_PATH.lstrip("/"),
        ),
    ]

    @staticmethod
    def run() -> None:
        missing = [str(p) for p, _ in WeightUploader._WEIGHTS if not p.exists()]
        if missing:
            print("ERROR: missing weight files:")
            for p in missing:
                print(f"  {p}")
            return

        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        with vol.batch_upload(force=True) as batch:
            for local_path, remote_name in WeightUploader._WEIGHTS:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"Uploading {local_path.name} ({size_mb:.1f} MB) -> /{remote_name}")
                batch.put_file(str(local_path), f"/{remote_name}")

        print(f"\nAll weights uploaded to volume '{VOLUME_NAME}'.")
        print(f"Verify with: modal volume ls {VOLUME_NAME}")


@app.local_entrypoint()
def main() -> None:
    WeightUploader.run()
