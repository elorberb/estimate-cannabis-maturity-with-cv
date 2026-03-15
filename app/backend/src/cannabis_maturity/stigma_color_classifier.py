from __future__ import annotations

import importlib.resources
import json

import numpy as np

_data = json.loads(
    importlib.resources.files("cannabis_maturity")
    .joinpath("stigma_color_data.json")
    .read_text(encoding="utf-8")
)
GREEN_DATA: np.ndarray = np.array(_data["green"], dtype=np.float32)
ORANGE_DATA: np.ndarray = np.array(_data["orange"], dtype=np.float32)


class StigmaColorClassifier:
    @staticmethod
    def classify(segment_rgb: np.ndarray) -> tuple[float, float]:
        bg_mask = np.all(segment_rgb < 10, axis=-1) | np.all(segment_rgb > 240, axis=-1)
        flat = segment_rgb.reshape(-1, 3).astype(np.float32)
        valid_indices = np.where(~bg_mask.reshape(-1))[0]
        if len(valid_indices) == 0:
            return 0.0, 0.0

        valid_pixels = flat[valid_indices]
        pixel_expanded = valid_pixels[:, np.newaxis, :]
        dist2_green = np.sum((pixel_expanded - GREEN_DATA[np.newaxis, :, :]) ** 2, axis=2)
        dist2_orange = np.sum((pixel_expanded - ORANGE_DATA[np.newaxis, :, :]) ** 2, axis=2)
        is_green = np.min(dist2_green, axis=1) < np.min(dist2_orange, axis=1)

        green_count = int(np.count_nonzero(is_green))
        orange_count = len(valid_indices) - green_count
        total = green_count + orange_count
        return green_count / total, orange_count / total
