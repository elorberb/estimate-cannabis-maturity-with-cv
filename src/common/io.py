import json
import os
import re


class IO:
    @staticmethod
    def save_json(data: dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def extract_number_from_filename(filename: str) -> int:
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else -1

    @staticmethod
    def get_image_files(
        folder_path: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    ) -> list[str]:
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(extensions)
        ]
        return sorted(files, key=IO.extract_number_from_filename)

    @staticmethod
    def ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path
