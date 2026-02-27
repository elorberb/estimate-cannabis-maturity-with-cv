from segments import SegmentsDataset

from src.config.paths import Paths
from src.config.settings import WEEKS_DIR, ZOOM_TYPES_DIR
from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler


class ModelAssistPipeline:
    def __init__(self) -> None:
        self._segments_handler = SegmentsAIHandler()

    def _train_segmentation_model(self, train_dataset_name: str):
        return None

    def _create_new_test_dataset(
        self, image_name: str, week: str, zoom_type: str, single_category: bool = True
    ) -> str:
        dataset_name = f"cannabis_patches_{week}_{zoom_type}_{image_name}"
        description = f"cannabis patches week={week} zoom_type={zoom_type} of image={image_name}."
        task_type = "segmentation-bitmap"

        if single_category:
            task_attributes = {
                "format_version": "0.1",
                "categories": [{"name": "trichome", "id": 0, "color": [65, 117, 5]}],
            }
        else:
            task_attributes = {
                "format_version": "0.1",
                "categories": [
                    {"name": "trichome", "id": 0, "color": [65, 117, 5]},
                    {"name": "clear", "id": 1, "color": [155, 155, 155]},
                    {"name": "cloudy", "id": 2, "color": [255, 255, 255]},
                    {"name": "amber", "id": 3, "color": [245, 166, 35]},
                ],
            }

        test_dataset = f"etaylor/{dataset_name}"
        self._segments_handler.create_dataset(dataset_name, description, task_type, task_attributes)
        return test_dataset

    def _upload_predictions(self, release, model) -> None:
        dataset = SegmentsDataset(release)
        for sample in dataset:
            image = sample["image"]
            segmentation_bitmap, annotations = model(image)
            self._segments_handler.upload_annotation(sample["uuid"], segmentation_bitmap, annotations)

    def run(self, image_name: str, week_key: str, zoom_type_key: str, visualize: bool = False) -> None:
        train_dataset_name = "etaylor/cannabis_patches_all_images"

        if visualize:
            self._segments_handler.visualize_dataset(train_dataset_name)

        model = self._train_segmentation_model(train_dataset_name)

        test_dataset = self._create_new_test_dataset(
            image_name, WEEKS_DIR[week_key], ZOOM_TYPES_DIR[zoom_type_key]
        )
        abs_images_path = f"{Paths.get_processed_cannabis_path(week_key, zoom_type_key)}/{image_name}"

        self._segments_handler.upload_images(test_dataset, abs_images_path)

        release_name = "v0.1"
        self._segments_handler.client.add_release(test_dataset, release_name, "upload predictions to dataset.")
        test_release = self._segments_handler.client.get_release(test_dataset, "v0.1")
        self._upload_predictions(test_release, model)


if __name__ == "__main__":
    pipeline = ModelAssistPipeline()
    pipeline.run("IMG_2129", "week9", "3xr", visualize=True)
