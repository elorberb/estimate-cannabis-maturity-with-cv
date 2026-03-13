from __future__ import annotations

import uuid

from supabase import Client


class StorageService:
    def __init__(self, client: Client, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def upload_image(self, image_bytes: bytes, content_type: str = "image/jpeg") -> str:
        path = f"{uuid.uuid4()}.jpg"
        self._client.storage.from_(self._bucket).upload(
            path=path,
            file=image_bytes,
            file_options={"content-type": content_type},
        )
        public_url = self._client.storage.from_(self._bucket).get_public_url(path)
        return public_url
