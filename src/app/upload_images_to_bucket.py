import csv

from google.cloud import storage

from src.config.paths import Paths
from src.config.settings import GOOD_QUALITY_IMAGES_CSV


class ImageUploader:
    @staticmethod
    def upload_files(bucket_name: str) -> None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        with open(GOOD_QUALITY_IMAGES_CSV, newline="") as csvfile:
            filereader = csv.DictReader(csvfile)
            for row in filereader:
                image_number = row["image_number"]
                week, zoom_type = Paths.find_image_details(image_number)
                if week and zoom_type:
                    image_path = Paths.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"
                    blob = bucket.blob(f"images/{image_number}.JPG")
                    blob.upload_from_filename(str(image_path))


if __name__ == "__main__":
    ImageUploader.upload_files("trichome_classification_study_storage")
