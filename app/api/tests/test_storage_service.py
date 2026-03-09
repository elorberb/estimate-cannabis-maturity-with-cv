from uuid import uuid4

from src.config import settings
from src.services.storage import SupabaseStorageService


def test_supabase_storage_service_round_trip() -> None:
    service = SupabaseStorageService(
        supabase_url=settings.supabase_url,
        supabase_service_key=settings.supabase_service_key,
        bucket_name=settings.supabase_storage_bucket,
    )

    path = f"test-storage/{uuid4().hex}.txt"
    original = b"test-storage-service"

    returned_path = service.upload_image(path, original, "text/plain")
    assert returned_path == path

    public_url = service.get_public_url(path)
    assert isinstance(public_url, str)
    assert public_url != ""

    downloaded = service.download_image(path)
    assert downloaded == original

