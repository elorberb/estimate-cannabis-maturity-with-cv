import csv
import json
import os

import numpy as np


class ExperimentDataUtils:
    @staticmethod
    def get_class_distribution(json_data: dict) -> dict[str, float]:
        return {
            "clear_count": float(np.round(json_data["class_distribution"].get("1", 0), 4)),
            "cloudy_count": float(np.round(json_data["class_distribution"].get("2", 0), 4)),
            "amber_count": float(np.round(json_data["class_distribution"].get("3", 0), 4)),
            "clear_normalized": float(np.round(json_data["normalized_class_distribution"].get("1", 0), 4)),
            "cloudy_normalized": float(np.round(json_data["normalized_class_distribution"].get("2", 0), 4)),
            "amber_normalized": float(np.round(json_data["normalized_class_distribution"].get("3", 0), 4)),
        }

    @staticmethod
    def collect_data_from_json(
        base_path: str, day_folders: list[str]
    ) -> tuple[list[dict], list[dict]]:
        data_per_image: list[dict] = []
        data_per_folder: list[dict] = []

        for day_folder in day_folders:
            day_folder_path = os.path.join(base_path, day_folder)

            for subfolder in ["greenhouse", "lab"]:
                subfolder_path = os.path.join(day_folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                for numbered_folder in os.listdir(subfolder_path):
                    numbered_folder_path = os.path.join(subfolder_path, numbered_folder)
                    if not os.path.isdir(numbered_folder_path):
                        continue

                    json_file_path = os.path.join(numbered_folder_path, "class_distribution.json")
                    if not os.path.exists(json_file_path):
                        continue

                    with open(json_file_path) as f:
                        json_data = json.load(f)

                    folder_entry = {
                        "day": day_folder,
                        "location": subfolder,
                        "number": numbered_folder,
                    }
                    folder_entry.update(ExperimentDataUtils.get_class_distribution(json_data))
                    data_per_folder.append(folder_entry)

                    for image_folder in os.listdir(numbered_folder_path):
                        image_folder_path = os.path.join(numbered_folder_path, image_folder)
                        if not os.path.isdir(image_folder_path) or not image_folder.startswith("IMG_"):
                            continue

                        for img_file in os.listdir(image_folder_path):
                            if not img_file.endswith("_class_distribution.json"):
                                continue

                            img_json_file_path = os.path.join(image_folder_path, img_file)
                            with open(img_json_file_path) as img_f:
                                img_json_data = json.load(img_f)

                            image_entry = {
                                "day": day_folder,
                                "location": subfolder,
                                "number": numbered_folder,
                                "image": image_folder,
                            }
                            image_entry.update(ExperimentDataUtils.get_class_distribution(img_json_data))
                            data_per_image.append(image_entry)

        return data_per_image, data_per_folder

    @staticmethod
    def save_data_to_csv(data: list[dict], output_file: str) -> None:
        if not data:
            return

        fieldnames = list(data[0].keys())

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    @staticmethod
    def validate_image_folders(base_path: str, day_folders: list[str]) -> None:
        current_number = 1

        for day_folder in day_folders:
            day_folder_path = os.path.join(base_path, day_folder)

            for subfolder in ["greenhouse", "lab"]:
                subfolder_path = os.path.join(day_folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                for i in range(current_number, current_number + 30):
                    numbered_folder_path = os.path.join(subfolder_path, str(i))
                    if not os.path.exists(numbered_folder_path):
                        continue
                    if not any(
                        os.path.isfile(os.path.join(numbered_folder_path, f))
                        for f in os.listdir(numbered_folder_path)
                    ):
                        pass
