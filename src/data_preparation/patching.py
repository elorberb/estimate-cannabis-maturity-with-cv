import os

import cv2
import numpy as np
import pandas as pd


class Patching:
    @staticmethod
    def calculate_padding(image_shape: tuple, patch_size: int) -> tuple[int, int]:
        pad_height = (patch_size - image_shape[0] % patch_size) % patch_size
        pad_width = (patch_size - image_shape[1] % patch_size) % patch_size
        return pad_height, pad_width

    @staticmethod
    def pad_image(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
        return cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    @staticmethod
    def extract_patches(padded_image: np.ndarray, patch_size: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
        patches = []
        for i in range(0, padded_image.shape[0], patch_size):
            for j in range(0, padded_image.shape[1], patch_size):
                patch = padded_image[i:i + patch_size, j:j + patch_size]
                patches.append((patch, (i, j)))
        return patches

    @staticmethod
    def cut_images(image: np.ndarray, patch_size: int = 512) -> list[tuple[np.ndarray, tuple[int, int]]]:
        pad_height, pad_width = Patching.calculate_padding(image.shape, patch_size)
        padded_image = Patching.pad_image(image, pad_height, pad_width)
        return Patching.extract_patches(padded_image, patch_size)

    @staticmethod
    def create_patches_metadata(
        image_name: str, patches: list[tuple[np.ndarray, tuple[int, int]]]
    ) -> pd.DataFrame:
        rows = []
        for i, (_, coords) in enumerate(patches):
            rows.append({
                "patch_name": f"{image_name}_p{i}.png",
                "y": coords[0],
                "x": coords[1],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def save_patches(
        image_name: str,
        patches: list[tuple[np.ndarray, tuple[int, int]]],
        saving_dir: str,
        csv_file_path: str,
    ) -> None:
        os.makedirs(saving_dir, exist_ok=True)
        new_metadata = Patching.create_patches_metadata(image_name, patches)

        for i, (patch, _) in enumerate(patches):
            patch_filename = f"{image_name}_p{i}.png"
            patch_path = os.path.join(saving_dir, patch_filename)
            cv2.imwrite(patch_path, patch)

        if os.path.exists(csv_file_path):
            existing = pd.read_csv(csv_file_path)
            metadata = pd.concat([existing, new_metadata], ignore_index=True)
        else:
            metadata = new_metadata

        metadata.to_csv(csv_file_path, index=False)

    @staticmethod
    def apply_to_images(images_and_names: list, func) -> list:
        return [(func(image), name) for image, name in images_and_names]
