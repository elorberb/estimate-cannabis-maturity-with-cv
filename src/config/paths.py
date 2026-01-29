import os
from src.config.settings import (
    RAW_IMAGE_DIR,
    PROCESSED_CANNABIS_PATCHES_DIR,
    PROCESSED_TRICHOME_PATCHES_DIR,
    WEEKS_DIR,
    ZOOM_TYPES_DIR,
)


def resolve_week_dir(week):
    return WEEKS_DIR.get(week, week)


def resolve_zoom_type_dir(zoom_type):
    return ZOOM_TYPES_DIR.get(zoom_type, zoom_type)


def get_raw_image_path(week, zoom_type):
    return RAW_IMAGE_DIR / resolve_week_dir(week) / resolve_zoom_type_dir(zoom_type)


def get_processed_cannabis_path(week, zoom_type):
    return PROCESSED_CANNABIS_PATCHES_DIR / resolve_week_dir(week) / resolve_zoom_type_dir(zoom_type)


def get_processed_trichome_path(week, zoom_type):
    return PROCESSED_TRICHOME_PATCHES_DIR / resolve_week_dir(week) / resolve_zoom_type_dir(zoom_type)


def find_image_details(image_number, base_path=RAW_IMAGE_DIR):
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


def get_image_path(image_name, base_path=RAW_IMAGE_DIR, processed_type=None):
    week, zoom_type = find_image_details(image_name, base_path)
    if week is None or zoom_type is None:
        return None

    week_dir = resolve_week_dir(week)
    zoom_type_dir = resolve_zoom_type_dir(zoom_type)

    if processed_type == "cannabis":
        return os.path.join(PROCESSED_CANNABIS_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
    if processed_type == "trichome":
        return os.path.join(PROCESSED_TRICHOME_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
    return os.path.join(base_path, week_dir, zoom_type_dir, f"{image_name}.JPG")
