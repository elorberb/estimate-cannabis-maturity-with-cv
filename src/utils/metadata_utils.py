import csv
import os
from datetime import datetime

from src.config.settings import DATETIME_FORMAT


class MetadataUtils:
    @staticmethod
    def annotation_tracking(csv_file: str, image_number: str, annotator: str) -> None:
        current_time = datetime.now().strftime(DATETIME_FORMAT)
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Image Number", "Annotator", "Time"])
            writer.writerow([image_number, annotator, current_time])
