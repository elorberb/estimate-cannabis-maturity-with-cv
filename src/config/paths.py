import os
from pathlib import Path

from src.config.settings import (
    PROCESSED_CANNABIS_PATCHES_DIR,
    PROCESSED_TRICHOME_PATCHES_DIR,
    RAW_IMAGE_DIR,
    WEEKS_DIR,
    ZOOM_TYPES_DIR,
)


class Paths:
    @staticmethod
    def resolve_week_dir(week: str) -> str:
        return WEEKS_DIR.get(week, week)

    @staticmethod
    def resolve_zoom_type_dir(zoom_type: str) -> str:
        return ZOOM_TYPES_DIR.get(zoom_type, zoom_type)

    @staticmethod
    def get_raw_image_path(week: str, zoom_type: str) -> Path:
        return RAW_IMAGE_DIR / Paths.resolve_week_dir(week) / Paths.resolve_zoom_type_dir(zoom_type)

    @staticmethod
    def get_processed_cannabis_path(week: str, zoom_type: str) -> Path:
        return PROCESSED_CANNABIS_PATCHES_DIR / Paths.resolve_week_dir(week) / Paths.resolve_zoom_type_dir(zoom_type)

    @staticmethod
    def get_processed_trichome_path(week: str, zoom_type: str) -> Path:
        return PROCESSED_TRICHOME_PATCHES_DIR / Paths.resolve_week_dir(week) / Paths.resolve_zoom_type_dir(zoom_type)

    @staticmethod
    def find_image_details(image_number: str, base_path: Path = RAW_IMAGE_DIR) -> tuple[str | None, str | None]:
        for week in os.listdir(base_path):
            week_path = os.path.join(base_path, week)
            if not os.path.isdir(week_path):
                continue
            for zoom_type in os.listdir(week_path):
                zoom_path = os.path.join(week_path, zoom_type)
                if not os.path.isdir(zoom_path):
                    continue
                image_path = os.path.join(zoom_path, f"{image_number}.JPG")
                if os.path.exists(image_path):
                    return week, zoom_type
        return None, None

    @staticmethod
    def get_image_path(
        image_name: str, base_path: Path = RAW_IMAGE_DIR, processed_type: str | None = None
    ) -> str | None:
        week, zoom_type = Paths.find_image_details(image_name, base_path)
        if week is None or zoom_type is None:
            return None

        week_dir = Paths.resolve_week_dir(week)
        zoom_type_dir = Paths.resolve_zoom_type_dir(zoom_type)

        if processed_type == "cannabis":
            return os.path.join(PROCESSED_CANNABIS_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
        if processed_type == "trichome":
            return os.path.join(PROCESSED_TRICHOME_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
        return os.path.join(base_path, week_dir, zoom_type_dir, f"{image_name}.JPG")
