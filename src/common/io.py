import os
import json
import re


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_number_from_filename(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1


def get_image_files(folder_path, extensions=(".png", ".jpg", ".jpeg")):
    files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(extensions)
    ]
    return sorted(files, key=extract_number_from_filename)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
