from typing import Final

from supabase import Client, create_client


class SupabaseStorageService:
    def __init__(
        self,
        supabase_url: str,
        supabase_service_key: str,
        bucket_name: str,
    ) -> None:
        self._bucket_name: Final[str] = bucket_name
        self._client: Client = create_client(supabase_url, supabase_service_key)

    def upload_image(self, path: str, data: bytes, content_type: str) -> str:
        self._client.storage.from_(self._bucket_name).upload(
            path,
            data,
            {
                "contentType": content_type,
                "upsert": "true",
            },
        )
        return path

    def get_public_url(self, path: str) -> str:
        result = self._client.storage.from_(self._bucket_name).get_public_url(path)
        if isinstance(result, str):
            return result

        data = result.get("data")
        if isinstance(data, dict):
            public_url = data.get("publicUrl")
            if isinstance(public_url, str):
                return public_url

        raise ValueError("Supabase did not return a public URL for the given path.")

    def download_image(self, path: str) -> bytes:
        file_data = self._client.storage.from_(self._bucket_name).download(path)
        if isinstance(file_data, bytes):
            return file_data

        raise TypeError("Supabase download did not return raw bytes.")

