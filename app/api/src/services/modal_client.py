from __future__ import annotations

import asyncio

import modal

from services.inference_error import InferenceError


class ModalClient:
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    async def analyze(self, image_bytes: bytes) -> dict:
        try:
            function = modal.Function.lookup(self._app_name, "MaturityAnalyzer.analyze")
            return await asyncio.to_thread(function.remote, image_bytes)
        except modal.exception.Error as e:
            raise InferenceError(f"Modal inference failed: {e}") from e
