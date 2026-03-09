from typing import Final

from supabase import Client, create_client


class SupabaseDatabaseService:
    def __init__(
        self,
        supabase_url: str,
        supabase_service_key: str,
    ) -> None:
        self._client: Client = create_client(supabase_url, supabase_service_key)
        self._table_name: Final[str] = "analyses"

    def create_analysis(self, values: dict[str, object]) -> dict[str, object]:
        response = self._client.table(self._table_name).insert(values).execute()
        data = response.data
        if not isinstance(data, list) or not data:
            raise ValueError("Supabase did not return inserted analysis data.")

        row = data[0]
        if not isinstance(row, dict):
            raise TypeError("Supabase returned analysis row in unexpected format.")

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
            raise TypeError("Supabase returned analysis row in unexpected format.")

        return row

    def list_analyses_for_device(
        self,
        device_id: str,
        limit: int = 20,
        offset: int = 0,
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
            raise TypeError("Supabase returned analysis list in unexpected format.")

        rows: list[dict[str, object]] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Supabase returned analysis row in unexpected format.")
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
            raise TypeError("Supabase returned analysis row in unexpected format.")

        return row

