import csv
import json
import os

import numpy as np

EXP_1_DAY_FOLDERS = [
    "day_1_2024_05_30",
    "day_3_2024_06_06",
    "day_4_2024_06_10",
    "day_5_2024_06_13",
    "day_6_2024_06_17",
    "day_7_2024_06_20",
    "day_9_2024_06_27",
]

EXP_2_DAY_FOLDERS = [
    "day_1_2024_12_05",
    "day_2_2024_12_09",
    "day_3_2024_12_12",
    "day_4_2024_12_17",
    "day_5_2024_12_24",
    "day_6_2024_12_30",
    "day_7_2025_01_06",
    "day_8_2025_01_09",
    "day_9_2025_01_16",
]


class TrichomesResultsCollector:
    @staticmethod
    def load_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def get_class_distribution(json_data: dict) -> dict:
        return {
            "clear_count": np.round(json_data["class_distribution"].get("1", 0), 4),
            "cloudy_count": np.round(json_data["class_distribution"].get("2", 0), 4),
            "amber_count": np.round(json_data["class_distribution"].get("3", 0), 4),
            "clear_normalized": np.round(json_data["normalized_class_distribution"].get("1", 0), 4),
            "cloudy_normalized": np.round(json_data["normalized_class_distribution"].get("2", 0), 4),
            "amber_normalized": np.round(json_data["normalized_class_distribution"].get("3", 0), 4),
        }

    @staticmethod
    def process_image_folder(
        image_folder_path: str,
        day_folder: str,
        subfolder: str,
        numbered_folder: str,
        image_folder: str,
    ) -> dict | None:
        for img_file in os.listdir(image_folder_path):
            if img_file.endswith("_class_distribution.json"):
                json_data = TrichomesResultsCollector.load_json(
                    os.path.join(image_folder_path, img_file)
                )
                entry = {
                    "day": day_folder,
                    "location": subfolder,
                    "number": numbered_folder,
                    "image": image_folder,
                }
                entry.update(TrichomesResultsCollector.get_class_distribution(json_data))
                return entry
        return None

    @staticmethod
    def process_numbered_folder(
        numbered_folder_path: str, day_folder: str, subfolder: str, numbered_folder: str
    ) -> tuple[dict | None, list[dict]]:
        json_file_path = os.path.join(numbered_folder_path, "class_distribution.json")
        if not os.path.exists(json_file_path):
            return None, []

        json_data = TrichomesResultsCollector.load_json(json_file_path)
        folder_entry: dict = {"day": day_folder, "location": subfolder, "number": numbered_folder}
        folder_entry.update(TrichomesResultsCollector.get_class_distribution(json_data))

        image_entries = []
        for image_folder in os.listdir(numbered_folder_path):
            image_folder_path = os.path.join(numbered_folder_path, image_folder)
            if os.path.isdir(image_folder_path) and image_folder.startswith("IMG_"):
                entry = TrichomesResultsCollector.process_image_folder(
                    image_folder_path, day_folder, subfolder, numbered_folder, image_folder
                )
                if entry:
                    image_entries.append(entry)

        return folder_entry, image_entries

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

                    folder_entry, image_entries = TrichomesResultsCollector.process_numbered_folder(
                        numbered_folder_path, day_folder, subfolder, numbered_folder
                    )
                    if folder_entry:
                        data_per_folder.append(folder_entry)
                    data_per_image.extend(image_entries)

        return data_per_image, data_per_folder

    @staticmethod
    def save_data_to_csv(data: list[dict], output_file: str) -> None:
        if not data:
            return

        fieldnames = list(data[0].keys())
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    base_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/results/faster_rcnn_with_yolo"
    output_folder = os.path.join(base_path, "csv_results")
    os.makedirs(output_folder, exist_ok=True)

    data_per_image, data_per_folder = TrichomesResultsCollector.collect_data_from_json(
        base_path, EXP_1_DAY_FOLDERS
    )
    TrichomesResultsCollector.save_data_to_csv(
        data_per_image, os.path.join(output_folder, "collected_class_distribution_per_image.csv")
    )
    TrichomesResultsCollector.save_data_to_csv(
        data_per_folder, os.path.join(output_folder, "collected_class_distribution_per_folder.csv")
    )
