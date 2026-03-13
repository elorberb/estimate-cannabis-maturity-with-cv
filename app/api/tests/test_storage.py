from unittest.mock import MagicMock, patch

import pytest

from src.services.storage import StorageService

URL = "https://example.supabase.co"
SERVICE_KEY = "test-service-key"
BUCKET = "images"
PATH = "plants/abc123.jpg"
DATA = b"fake-image-bytes"
CONTENT_TYPE = "image/jpeg"


def _make_service() -> StorageService:
    with patch("src.services.storage.create_client", return_value=MagicMock()):
        return StorageService(url=URL, service_key=SERVICE_KEY, bucket_name=BUCKET)


def test_upload_image_returns_path() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.upload.return_value = {}

    result = service.upload_image(PATH, DATA, CONTENT_TYPE)

    bucket_mock.upload.assert_called_once_with(
        PATH,
        DATA,
        {"contentType": CONTENT_TYPE, "upsert": "true"},
    )
    assert result == PATH


def test_get_public_url_returns_string_directly() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.get_public_url.return_value = "https://cdn.example.com/image.jpg"

    result = service.get_public_url(PATH)

    assert result == "https://cdn.example.com/image.jpg"


def test_get_public_url_extracts_from_dict_response() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.get_public_url.return_value = {
        "data": {"publicUrl": "https://cdn.example.com/image.jpg"}
    }

    result = service.get_public_url(PATH)

    assert result == "https://cdn.example.com/image.jpg"


def test_get_public_url_raises_when_no_url_in_response() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.get_public_url.return_value = {"data": {}}

    with pytest.raises(ValueError, match="public URL"):
        service.get_public_url(PATH)


def test_download_image_returns_bytes() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.download.return_value = DATA

    result = service.download_image(PATH)

    bucket_mock.download.assert_called_once_with(PATH)
    assert result == DATA


def test_download_image_raises_when_not_bytes() -> None:
    service = _make_service()
    bucket_mock = service._client.storage.from_.return_value
    bucket_mock.download.return_value = {"error": "something went wrong"}

    with pytest.raises(TypeError, match="bytes"):
        service.download_image(PATH)
