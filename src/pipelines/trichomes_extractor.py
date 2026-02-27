import os

import cv2

from src.common.detection import Detection
from src.common.logging import Logging
from src.common.visualization import Visualization
from src.pipelines.end_to_end.end_to_end_utils import EndToEndUtils

logger = Logging.get_logger(__name__)

CLASS_MAPPING = {1: "clear", 2: "cloudy", 3: "amber"}


class TrichomesExtractor:
    @staticmethod
    def save_detection_results(
        image_path: str,
        predictions: list,
        image_identifier_folder: str,
        margin: float = 0.25,
    ) -> None:
        image_identifier = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image from {image_path}")
            return

        image_height, image_width = image.shape[:2]

        for idx, prediction in enumerate(predictions):
            bbox = prediction.bbox
            x_min = int(bbox.minx)
            y_min = int(bbox.miny)
            x_max = int(bbox.maxx)
            y_max = int(bbox.maxy)

            x_min_ext, y_min_ext, x_max_ext, y_max_ext = Detection.extend_bbox(
                x_min, y_min, x_max, y_max, image_width, image_height, margin
            )
            cropped_img = Detection.crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)

            class_name = CLASS_MAPPING.get(prediction.category.id, "unknown")
            class_folder = os.path.join(image_identifier_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

            result_filename = f"{image_identifier}_trichome_{idx}_bbox_{x_min_ext}_{y_min_ext}_{x_max_ext}_{y_max_ext}.jpg"
            result_path = os.path.join(class_folder, result_filename)
            cv2.imwrite(result_path, cropped_img)
            logger.info(f"Saved detected object to {result_path}")

    @staticmethod
    def extract_trichomes_from_image(
        image_path: str,
        detection_model,
        patch_size: int = 512,
        save_results_path: str = "results",
    ) -> None:
        os.makedirs(save_results_path, exist_ok=True)

        image_identifier = os.path.splitext(os.path.basename(image_path))[0]
        image_identifier_folder = os.path.join(save_results_path, image_identifier)
        os.makedirs(image_identifier_folder, exist_ok=True)

        detection_result = EndToEndUtils.run_sliced_detection(image_path, detection_model, patch_size)
        filtered_predictions = Detection.filter_large_objects(detection_result.object_prediction_list)

        TrichomesExtractor.save_detection_results(image_path, filtered_predictions, image_identifier_folder)
        Visualization.save_visuals(detection_result, image_identifier_folder, image_identifier)

    @staticmethod
    def run(image_path: str, detection_model_config: dict, save_results_path: str, patch_size: int = 512) -> None:
        detection_model = EndToEndUtils.load_detection_model(detection_model_config, patch_size)
        TrichomesExtractor.extract_trichomes_from_image(image_path, detection_model, patch_size, save_results_path)


if __name__ == "__main__":
    _detection_model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }
    _save_results_path = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/extracted_trichomes_images/images_datasets"
    _image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_4_2024_06_10/greenhouse/100/IMG_6182.JPG"

    TrichomesExtractor.run(_image_path, _detection_model_config, _save_results_path)
