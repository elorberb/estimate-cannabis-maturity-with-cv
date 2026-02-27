from datetime import datetime
from pathlib import Path

ROOT_IMAGES_DIR = Path("/sise/home/etaylor/images/")
RAW_IMAGE_DIR = ROOT_IMAGES_DIR / "raw_images"
PROCESSED_IMAGE_DIR = ROOT_IMAGES_DIR / "processed_images"
PROCESSED_CANNABIS_PATCHES_DIR = PROCESSED_IMAGE_DIR / "cannabis_patches"
PROCESSED_TRICHOME_PATCHES_DIR = PROCESSED_IMAGE_DIR / "trichome_patches"

WEEKS_DIR = {
    "week2": "week2_27_04_2023",
    "week3": "week3_03_05_2023",
    "week4": "week4_09_05_2023",
    "week5": "week5_18_05_2023",
    "week6": "week6_22_05_2023",
    "week7": "week7_01_06_2023",
    "week8": "week8_07_06_2023",
    "week9": "week9_15_06_2023",
}

ZOOM_TYPES_DIR = {
    "1xr": "1x_regular",
    "1xfs": "1x_focus_stacking",
    "3xr": "3x_regular",
    "3xfs": "3x_focus_stacking",
}

ANNOTATIONS_CLASS_MAPPINGS = {
    0: "trichome",
    1: "clear",
    2: "cloudy",
    3: "amber",
}

CANNABIS_PATCH_SIZE = 512
DATETIME_FORMAT = "%d-%m-%Y_%H-%M-%S"

GOOD_QUALITY_IMAGES_CSV = Path("/home/etaylor/code_projects/thesis/metadata/good_quality_images.csv")
SEGMENTS_FOLDER = "/home/etaylor/code_projects/thesis/segments"

ULTRALYTICS_RUNS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/runs"
ULTRALYTICS_WEIGHTS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/weights"
ULTRALYTICS_DATASETS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/datasets"


class DateUtils:
    @staticmethod
    def get_datetime_str() -> str:
        return datetime.now().strftime(DATETIME_FORMAT)
