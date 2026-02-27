import cv2
import numpy as np

from src.config.paths import Paths
from src.config.settings import CANNABIS_PATCH_SIZE
from src.data_preparation.image_io import ImageIO
from src.data_preparation.patching import Patching
from src.data_preparation.sharpness import Sharpness


class PreprocessingPipeline:
    @staticmethod
    def filter_sharp_patches(patches: list) -> list:
        mean_sharpness = np.mean([Sharpness.calculate_sharpness(p) for p, _ in patches])
        return [
            (patch, coords)
            for patch, coords in patches
            if Sharpness.calculate_sharpness(patch) > mean_sharpness
        ]

    @staticmethod
    def preprocess_single_image(
        image_or_path: np.ndarray | str,
        image_name: str,
        patch_size: int = 512,
    ) -> list:
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                return []
        else:
            image = image_or_path

        patches = Patching.cut_images(image, patch_size=patch_size)
        return PreprocessingPipeline.filter_sharp_patches(patches)

    @staticmethod
    def run(
        images_source: str,
        patch_size: int = 512,
        saving_dir: str | None = None,
        csv_file_path: str | None = None,
    ) -> dict[str, list]:
        images = ImageIO.read_images(input_path_or_list=images_source)
        images_patches = {}

        for image_name, image in images.items():
            sharp_patches = PreprocessingPipeline.preprocess_single_image(image, image_name, patch_size=patch_size)

            if saving_dir and csv_file_path:
                Patching.save_patches(image_name, sharp_patches, saving_dir, csv_file_path)

            images_patches[image_name] = sharp_patches

        return images_patches


if __name__ == "__main__":
    week = "week9"
    zoom_type = "3xr"
    PreprocessingPipeline.run(
        images_source=str(Paths.get_raw_image_path(week, zoom_type)),
        saving_dir=str(Paths.get_processed_cannabis_path(week, zoom_type)),
        csv_file_path="metadata/cannabis_patches_metadata.csv",
        patch_size=CANNABIS_PATCH_SIZE,
    )
