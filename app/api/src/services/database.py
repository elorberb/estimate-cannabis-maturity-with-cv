from typing import Final

from supabase import Client, create_client

ANALYSES_TABLE: Final[str] = "analyses"


class DatabaseService:
    def __init__(
        self,
        url: str,
        service_key: str,
    ) -> None:
        self._client: Client = create_client(url, service_key)
        self._table_name: Final[str] = ANALYSES_TABLE

    def create_analysis(self, values: dict[str, object]) -> dict[str, object]:
        response = self._client.table(self._table_name).insert(values).execute()
        data = response.data
        if not isinstance(data, list) or not data:
            raise ValueError("Insert did not return analysis data.")

        row = data[0]
        if not isinstance(row, dict):
            raise TypeError("Response row in unexpected format.")

        return row

    def get_analysis_by_id(self, analysis_id: str) -> dict[str, object] | None:
        response = (
            self._client.table(self._table_name)
            .select("*")
            .eq("id", analysis_id)
            .limit(1)
            .execute()
        )
        data = response.data
        if not isinstance(data, list) or not data:
            return None

        row = data[0]
        if not isinstance(row, dict):
            raise TypeError("Response row in unexpected format.")

        return row

    def list_analyses_for_device(
        self,
        device_id: str,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        end = offset + max(limit, 0) - 1
        query = (
            self._client.table(self._table_name)
            .select("*")
            .eq("device_id", device_id)
            .order("created_at", desc=True)
        )

        if end >= offset:
            query = query.range(offset, end)

        response = query.execute()
        data = response.data
        if not isinstance(data, list):
            raise TypeError("Response list in unexpected format.")

        rows: list[dict[str, object]] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Response row in unexpected format.")
            rows.append(item)

        return rows

    def update_corrections(
        self,
        analysis_id: str,
        corrections: dict[str, object],
    ) -> dict[str, object] | None:
        response = (
            self._client.table(self._table_name)
            .update({"corrections": corrections})
            .eq("id", analysis_id)
            .execute()
        )
        data = response.data
        if not isinstance(data, list) or not data:
            return None

        row = data[0]
        if not isinstance(row, dict):
            raise TypeError("Response row in unexpected format.")

        return row

