import os


class DatasetStats:
    @staticmethod
    def count_jpg_images(base_dir: str) -> int:
        total_images = 0
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("day_"):
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if file.lower().endswith(".jpg"):
                            total_images += 1
        return total_images

    @staticmethod
    def count_images_in_folder(folder_path: str) -> int:
        count = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".jpg"):
                    count += 1
        return count

    @staticmethod
    def count_images_in_each_subfolder(base_folder: str) -> None:
        with os.scandir(base_folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    image_count = DatasetStats.count_images_in_folder(entry.path)
                    print(f"Folder: {entry.name} - {image_count} image(s)")


if __name__ == "__main__":
    _base_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images"
    print(f"Total number of JPG images in experiment 1: {DatasetStats.count_jpg_images(_base_folder)}")

    _base_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images"
    print(f"Total number of JPG images in experiment 2: {DatasetStats.count_jpg_images(_base_folder)}")

    DatasetStats.count_images_in_each_subfolder(
        "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/good_quality"
    )

    _folder_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/stigmas_images_flat"
    print(f"Folder: stigmas dataset - {DatasetStats.count_images_in_folder(_folder_path)} image(s)")
