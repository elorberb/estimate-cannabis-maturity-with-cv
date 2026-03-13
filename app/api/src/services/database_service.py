from __future__ import annotations

import datetime
import uuid

from supabase import Client


class DatabaseService:
    def __init__(self, client: Client) -> None:
        self._client = client

    def save_analysis(
        self,
        device_id: str,
        image_url: str,
        annotated_image_url: str | None,
        result_payload: dict,
    ) -> dict:
        record = {
            "id": str(uuid.uuid4()),
            "device_id": device_id,
            "image_url": image_url,
            "annotated_image_url": annotated_image_url,
            "result": result_payload,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        response = self._client.table("analyses").insert(record).execute()
        return response.data[0]

    def get_analysis(self, analysis_id: str) -> dict | None:
        response = self._client.table("analyses").select("*").eq("id", analysis_id).execute()
        if response.data:
            return response.data[0]
        return None

    def list_analyses(self, device_id: str, limit: int = 20) -> list[dict]:
        response = (
            self._client.table("analyses")
            .select("*")
            .eq("device_id", device_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data
