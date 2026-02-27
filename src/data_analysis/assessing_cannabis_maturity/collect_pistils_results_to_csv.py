import csv
import json
import os

import numpy as np

EXP_DAY_FOLDERS = [
    "day_1_2024_05_30",
    "day_3_2024_06_06",
    "day_4_2024_06_10",
    "day_5_2024_06_13",
    "day_6_2024_06_17",
    "day_7_2024_06_20",
    "day_9_2024_06_27",
]


class PistilsResultsCollector:
    @staticmethod
    def load_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_json(data: dict, path: str) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def get_color_ratios_from_dict(json_data: dict) -> dict:
        return {
            "num_pistils": json_data.get("num_pistils", 0),
            "avg_green_ratio": np.round(json_data.get("average_green_ratio", 0), 4),
            "avg_orange_ratio": np.round(json_data.get("average_orange_ratio", 0), 4),
        }

    @staticmethod
    def get_color_ratios_from_list(json_data: list) -> dict:
        num = len(json_data)
        avg_green = np.round(np.mean([e.get("green_ratio", 0) for e in json_data]), 4)
        avg_orange = np.round(np.mean([e.get("orange_ratio", 0) for e in json_data]), 4)
        return {"num_pistils": num, "avg_green_ratio": avg_green, "avg_orange_ratio": avg_orange}

    @staticmethod
    def get_color_ratios(json_data) -> dict:
        if isinstance(json_data, dict) and "num_pistils" in json_data:
            return PistilsResultsCollector.get_color_ratios_from_dict(json_data)
        if isinstance(json_data, list) and json_data:
            return PistilsResultsCollector.get_color_ratios_from_list(json_data)
        return {"num_pistils": 0, "avg_green_ratio": 0, "avg_orange_ratio": 0}

    @staticmethod
    def aggregate_folder_entries(image_entries: list[dict]) -> dict:
        total = sum(e["num_pistils"] for e in image_entries)
        count = len(image_entries)

        if total == 0 or count == 0:
            return {"num_pistils": 0, "avg_green_ratio": 0, "avg_orange_ratio": 0, "num_pistils_normalized": 0}

        avg_green = np.round(sum(e["num_pistils"] * e["avg_green_ratio"] for e in image_entries) / total, 4)
        avg_orange = np.round(sum(e["num_pistils"] * e["avg_orange_ratio"] for e in image_entries) / total, 4)
        normalized = np.round(total / count, 4)

        return {
            "num_pistils": total,
            "avg_green_ratio": avg_green,
            "avg_orange_ratio": avg_orange,
            "num_pistils_normalized": normalized,
        }

    @staticmethod
    def collect_image_entries(numbered_folder_path: str) -> list[dict]:
        entries = []
        for item in os.listdir(numbered_folder_path):
            image_folder_path = os.path.join(numbered_folder_path, item)
            if not os.path.isdir(image_folder_path) or not item.startswith("IMG_"):
                continue

            img_json_path = os.path.join(
                image_folder_path, "pistils_analysis", "aggregated_pistils_color_ratios.json"
            )
            if os.path.exists(img_json_path):
                entries.append(
                    PistilsResultsCollector.get_color_ratios(PistilsResultsCollector.load_json(img_json_path))
                )

        return entries

    @staticmethod
    def aggregate_folder_json(numbered_folder_path: str) -> dict:
        image_entries = PistilsResultsCollector.collect_image_entries(numbered_folder_path)
        aggregated = PistilsResultsCollector.aggregate_folder_entries(image_entries)

        folder_analysis_dir = os.path.join(numbered_folder_path, "pistils_analysis")
        os.makedirs(folder_analysis_dir, exist_ok=True)
        PistilsResultsCollector.save_json(
            aggregated, os.path.join(folder_analysis_dir, "aggregated_pistils_color_ratios.json")
        )

        return aggregated

    @staticmethod
    def process_image_entries(
        numbered_folder_path: str, day_folder: str, subfolder: str, numbered_folder: str
    ) -> list[dict]:
        entries = []
        for image_folder in os.listdir(numbered_folder_path):
            image_folder_path = os.path.join(numbered_folder_path, image_folder)
            if not os.path.isdir(image_folder_path) or not image_folder.startswith("IMG_"):
                continue

            img_json_path = os.path.join(
                image_folder_path, "pistils_analysis", "aggregated_pistils_color_ratios.json"
            )
            if not os.path.exists(img_json_path):
                continue

            img_json = PistilsResultsCollector.load_json(img_json_path)
            entries.append({
                "day": day_folder,
                "location": subfolder,
                "number": numbered_folder,
                "image": image_folder,
                "num_pistils": img_json.get("num_pistils", 0),
                "avg_green_ratio": np.round(
                    img_json.get("average_green_ratio", img_json.get("avg_green_ratio", 0)), 4
                ),
                "avg_orange_ratio": np.round(
                    img_json.get("average_orange_ratio", img_json.get("avg_orange_ratio", 0)), 4
                ),
            })

        return entries

    @staticmethod
    def collect_pistils_data(base_path: str, day_folders: list[str]) -> tuple[list[dict], list[dict]]:
        data_per_image: list[dict] = []
        data_per_folder: list[dict] = []

        for day_folder in day_folders:
            day_folder_path = os.path.join(base_path, day_folder)

            for subfolder in ["greenhouse"]:
                subfolder_path = os.path.join(day_folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    continue

                for numbered_folder in os.listdir(subfolder_path):
                    numbered_folder_path = os.path.join(subfolder_path, numbered_folder)
                    if not os.path.isdir(numbered_folder_path):
                        continue

                    aggregated = PistilsResultsCollector.aggregate_folder_json(numbered_folder_path)
                    data_per_folder.append({
                        "day": day_folder,
                        "location": subfolder,
                        "number": numbered_folder,
                        **aggregated,
                    })

                    data_per_image.extend(
                        PistilsResultsCollector.process_image_entries(
                            numbered_folder_path, day_folder, subfolder, numbered_folder
                        )
                    )

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

    data_per_image, data_per_folder = PistilsResultsCollector.collect_pistils_data(base_path, EXP_DAY_FOLDERS)
    PistilsResultsCollector.save_data_to_csv(
        data_per_image, os.path.join(output_folder, "collected_pistils_results_per_image.csv")
    )
    PistilsResultsCollector.save_data_to_csv(
        data_per_folder, os.path.join(output_folder, "collected_pistils_results_per_folder.csv")
    )
