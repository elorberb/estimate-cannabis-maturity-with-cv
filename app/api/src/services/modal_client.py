from __future__ import annotations

import asyncio

import modal


class InferenceError(Exception):
    pass


class ModalClient:
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    async def analyze(self, image_bytes: bytes) -> dict:
        try:
            fn = modal.Function.lookup(self._app_name, "TrichomeAnalyzer.analyze")
            result = await asyncio.to_thread(fn.remote, image_bytes)
            return result
        except modal.exception.Error as e:
            raise InferenceError(f"Modal inference failed: {e}") from e
