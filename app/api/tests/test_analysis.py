import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

MOCK_RESULT = {
    "trichome_result": {
        "detections": [],
        "distribution": {"clear": 5, "cloudy": 10, "amber": 2},
        "total_count": 17,
    },
    "stigma_result": {
        "detections": [],
        "avg_green_ratio": 0.6,
        "avg_orange_ratio": 0.4,
        "total_count": 3,
    },
    "maturity_stage": "peak",
    "recommendation": "Plant is at peak maturity. Harvest soon.",
    "annotated_image_b64": None,
    "trichome_crops_b64": [],
    "stigma_crops_b64": [],
}

MOCK_DB_RECORD = {
    "id": "test-uuid",
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "device_id": "test-device",
    "image_url": "https://example.com/image.jpg",
    "annotated_image_url": None,
    "result": MOCK_RESULT,
}


def test_analyze_invalid_content_type():
    response = client.post(
        "/api/v1/analyze",
        data={"device_id": "test-device"},
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422


def test_analyze_success():
    with (
        patch("src.routes.analysis.StorageService") as mock_storage_cls,
        patch("src.routes.analysis.DatabaseService") as mock_db_cls,
        patch("src.routes.analysis.ModalClient") as mock_modal_cls,
        patch("src.routes.analysis.create_client"),
    ):
        mock_storage = MagicMock()
        mock_storage.upload_image.return_value = "https://example.com/image.jpg"
        mock_storage_cls.return_value = mock_storage

        mock_db = MagicMock()
        mock_db.save_analysis.return_value = MOCK_DB_RECORD
        mock_db_cls.return_value = mock_db

        mock_modal = MagicMock()
        mock_modal.analyze = AsyncMock(return_value=MOCK_RESULT)
        mock_modal_cls.return_value = mock_modal

        small_jpg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00"
            b"\x08\x01\x01\x00\x00?\x00\xfb\xd7\xff\xd9"
        )

        response = client.post(
            "/api/v1/analyze",
            data={"device_id": "test-device"},
            files={"file": ("test.jpg", small_jpg, "image/jpeg")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["maturity_stage"] == "peak"
        assert data["trichome_distribution"]["total"] == 17
