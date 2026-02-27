import json
import os

import cv2

from src.data_preparation.sharpness import Sharpness


class SharpnessExtractor:
    @staticmethod
    def calculate_and_save(images_dir: str, save_path: str) -> None:
        sharpness_dict = {}
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    sharpness_dict[file] = Sharpness.calculate_sharpness(image)

        with open(save_path, "w") as json_file:
            json.dump(sharpness_dict, json_file, indent=4)


if __name__ == "__main__":
    SharpnessExtractor.calculate_and_save(
        images_dir="/home/etaylor/images/processed_images/cannabis_patches",
        save_path="/home/etaylor/code_projects/thesis/data/metadata/sharpness_per_patch_scores.json",
    )
