"""Download model weights from Supabase Storage for local inference.

Usage (run from app/api):
    cd app/api && uv run python ../scripts/download_weights.py

Downloads to:
    checkpoints/trichome_detection/yolov9_best.pt
    checkpoints/trichome_classification/yolov8/medium_fold0.pt   (medium — for local dev)
    checkpoints/stigma_segmentation/yolov8s_best.pt

Note: production Modal inference uses the large classification model.
For local testing the medium model is sufficient.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

_REPO_ROOT = Path(__file__).parent.parent.parent
_BUCKET = "model-weights"

_WEIGHTS = [
    ("trichome_detection.pt", _REPO_ROOT / "checkpoints/trichome_detection/yolov9_best.pt"),
    ("trichome_classification.pt", _REPO_ROOT / "checkpoints/trichome_classification/yolov8/medium_fold0.pt"),
    ("stigma_segmentation.pt", _REPO_ROOT / "checkpoints/stigma_segmentation/yolov8s_best.pt"),
]


def main() -> None:
    load_dotenv(Path(__file__).parent.parent / "api" / ".env")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in app/api/.env")
        sys.exit(1)

    client = create_client(url, key)

    for remote_name, local_path in _WEIGHTS:
        if local_path.exists():
            print(f"  skip  {local_path.name} (already exists)")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {remote_name} -> {local_path.relative_to(_REPO_ROOT)} ...")
        data: bytes = client.storage.from_(_BUCKET).download(remote_name)
        local_path.write_bytes(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"  OK ({size_mb:.0f} MB)")

    print("\nAll weights ready. Run the API with INFERENCE_MODE=local.")


if __name__ == "__main__":
    main()
