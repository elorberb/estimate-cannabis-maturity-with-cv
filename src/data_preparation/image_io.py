import os

import cv2
import numpy as np


class ImageIO:
    @staticmethod
    def extract_filename(file_path: str) -> str:
        basename = os.path.basename(file_path)
        name, _ = os.path.splitext(basename)
        return name

    @staticmethod
    def get_image_paths(
        directory: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    ) -> list[str]:
        extensions_lower = tuple(ext.lower() for ext in extensions)
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(extensions_lower)
        ]

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        return cv2.imread(path)

    @staticmethod
    def read_images(
        input_path_or_list: str | list, transform_fn=None
    ) -> dict[str, np.ndarray]:
        if isinstance(input_path_or_list, str):
            paths = ImageIO.get_image_paths(input_path_or_list)
        elif isinstance(input_path_or_list, list):
            paths = input_path_or_list
        else:
            raise ValueError("Input must be a directory path or a list of image paths.")

        images = {}
        for path in paths:
            filename = os.path.basename(path)
            image = ImageIO.read_image(path)
            if transform_fn:
                image = transform_fn(image)
            images[filename] = image

        return images

    @staticmethod
    def save_image(image: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
