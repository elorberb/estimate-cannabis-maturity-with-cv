import os
import random
import shutil

import cv2
import numpy as np


class BlurDatasetCreator:
    @staticmethod
    def apply_blur(image: np.ndarray, blur_type: str = "gaussian", intensity: int = 5) -> np.ndarray:
        if blur_type == "gaussian":
            return cv2.GaussianBlur(image, (intensity, intensity), 0)
        elif blur_type == "motion":
            kernel = np.zeros((intensity, intensity))
            kernel[int((intensity - 1) / 2), :] = np.ones(intensity)
            kernel /= intensity
            return cv2.filter2D(image, -1, kernel)
        elif blur_type == "median":
            return cv2.medianBlur(image, intensity)
        elif blur_type == "bilateral":
            return cv2.bilateralFilter(image, intensity, 75, 75)
        else:
            raise ValueError("Invalid blur type specified.")

    @staticmethod
    def create_blur_vs_quality(
        input_folder: str,
        output_folder: str,
        blur_types: list[str],
        intensity_range: tuple[int, int] = (5, 15),
        blur_fraction: float = 0.5,
    ) -> None:
        for trichome_class in ["clear", "cloudy", "amber"]:
            class_path = os.path.join(input_folder, trichome_class)
            blur_output_path = os.path.join(output_folder, "dataset_1", trichome_class, "blur")
            quality_output_path = os.path.join(output_folder, "dataset_1", trichome_class, "good_quality")
            os.makedirs(blur_output_path, exist_ok=True)
            os.makedirs(quality_output_path, exist_ok=True)

            files = os.listdir(class_path)
            selected_files = random.sample(files, int(len(files) * blur_fraction))

            for file in files:
                shutil.copy(os.path.join(class_path, file), os.path.join(quality_output_path, file))

            for file in selected_files:
                image = cv2.imread(os.path.join(class_path, file))
                for blur_type in blur_types:
                    intensity = random.randint(*intensity_range)
                    blurred_img = BlurDatasetCreator.apply_blur(image, blur_type, intensity)
                    cv2.imwrite(
                        os.path.join(blur_output_path, f"{blur_type}_{intensity}_{file}"),
                        blurred_img,
                    )

    @staticmethod
    def create_combined_good_quality(input_folder: str, output_folder: str) -> None:
        combined_output_path = os.path.join(output_folder, "dataset_2", "good_quality")
        os.makedirs(combined_output_path, exist_ok=True)
        for trichome_class in ["clear", "cloudy", "amber"]:
            class_path = os.path.join(input_folder, trichome_class)
            for file in os.listdir(class_path):
                shutil.copy(
                    os.path.join(class_path, file),
                    os.path.join(combined_output_path, f"{trichome_class}_{file}"),
                )


if __name__ == "__main__":
    _good_quality_path = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/good_quality"
    _output_base = "/home/etaylor/code_projects/thesis/classification_datasets/output_dataset"

    BlurDatasetCreator.create_blur_vs_quality(
        _good_quality_path, _output_base, ["gaussian", "motion", "median"]
    )
    BlurDatasetCreator.create_combined_good_quality(_good_quality_path, _output_base)
