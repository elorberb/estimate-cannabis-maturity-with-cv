import argparse
import os
import re
import time

import cv2
import numpy as np
from fastai.vision.all import PILImage, Resize, Transform, load_learner, rgb2hsv
from ultralytics import YOLO

from src.common.detection import Detection
from src.common.logging import Logging
from src.pipelines.end_to_end.end_to_end_utils import EndToEndUtils

logger = Logging.get_logger(__name__)


class RGB2HSV(Transform):
    def encodes(self, img: PILImage):
        return rgb2hsv(img)


class ClassificationPipeline:
    CLASS_COLOR_MAPPING = {
        1: (128, 128, 128),
        2: (255, 255, 255),
        3: (0, 165, 255),
    }

    DETECTION_CLASS_ID_TO_NAME = {1: "Clear", 2: "Cloudy", 3: "Amber"}
    CLASSIFICATION_CLASS_ID_TO_NAME = {0: "amber", 1: "clear", 2: "cloudy"}
    CLASSIFICATION_TO_DETECTION_ID_MAPPING = {0: 3, 1: 1, 2: 2}

    @staticmethod
    def custom_transform(size: int):
        return Resize(size, method="pad", pad_mode="zeros")

    @staticmethod
    def load_classification_model(model_config: dict, model_type: str):
        if model_type == "fastai":
            return load_learner(model_config["checkpoint"])
        if model_type == "yolo":
            return YOLO(model_config["checkpoint"])
        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def classify_bbox_image(cropped_image, classification_model, model_type: str):
        if model_type == "fastai":
            cropped_pil_image = (
                PILImage.create(cropped_image)
                if isinstance(cropped_image, np.ndarray)
                else cropped_image
            )
            pred_class, _, _ = classification_model.predict(cropped_pil_image)
            return pred_class
        if model_type == "yolo":
            results = classification_model(cropped_image)
            return int(results[0].probs.data.argmax())
        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def classify_trichomes(image_path: str, result, classification_model, model_type: str) -> None:
        logger.info("Classifying good quality detected trichomes.")
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        for prediction in result.object_prediction_list:
            x_min = int(prediction.bbox.minx)
            y_min = int(prediction.bbox.miny)
            x_max = int(prediction.bbox.maxx)
            y_max = int(prediction.bbox.maxy)

            x_min_ext, y_min_ext, x_max_ext, y_max_ext = Detection.extend_bbox(
                x_min, y_min, x_max, y_max, image_width, image_height
            )
            cropped_image = Detection.crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)
            pred = ClassificationPipeline.classify_bbox_image(cropped_image, classification_model, model_type)

            if model_type == "fastai":
                pred_lower = pred.lower()
                pred_class_id = list(ClassificationPipeline.CLASSIFICATION_CLASS_ID_TO_NAME.keys())[
                    list(ClassificationPipeline.CLASSIFICATION_CLASS_ID_TO_NAME.values()).index(pred_lower)
                ]
            else:
                pred_class_id = pred

            detection_class_id = ClassificationPipeline.CLASSIFICATION_TO_DETECTION_ID_MAPPING.get(pred_class_id, 0)
            prediction.category.id = detection_class_id
            prediction.category.name = ClassificationPipeline.DETECTION_CLASS_ID_TO_NAME.get(detection_class_id, "Unknown")

    @staticmethod
    def perform_blur_classification(
        image_path: str, predictions: list, blur_classification_model, model_type: str
    ) -> tuple[list, list]:
        logger.info("Filtering out blurry objects.")
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        filtered_predictions = []
        blurry_trichomes = []

        for prediction in predictions:
            x_min = int(prediction.bbox.minx)
            y_min = int(prediction.bbox.miny)
            x_max = int(prediction.bbox.maxx)
            y_max = int(prediction.bbox.maxy)

            x_min_ext, y_min_ext, x_max_ext, y_max_ext = Detection.extend_bbox(
                x_min, y_min, x_max, y_max, image_width, image_height
            )
            cropped_image = Detection.crop_image(image, x_min_ext, y_min_ext, x_max_ext, y_max_ext)

            is_sharp = ClassificationPipeline.classify_bbox_image(
                cropped_image, blur_classification_model, model_type
            )
            if is_sharp == "good_quality" or is_sharp == 1:
                filtered_predictions.append(prediction)
            else:
                blurry_trichomes.append((prediction, cropped_image))

        logger.info(f"Predictions after blur filtering: {len(filtered_predictions)}")
        return filtered_predictions, blurry_trichomes

    @staticmethod
    def process_image(
        image_path: str,
        detection_model,
        classification_models: dict,
        model_type: str,
        patch_size: int,
        perform_blur_classification_flag: bool = False,
    ):
        results = EndToEndUtils.run_sliced_detection(image_path, detection_model, patch_size)
        filtered_predictions = Detection.filter_large_objects(results.object_prediction_list)
        filtered_predictions = Detection.non_max_suppression(filtered_predictions, iou_threshold=0.7)
        results.object_prediction_list = filtered_predictions

        if perform_blur_classification_flag:
            filtered_predictions, _ = ClassificationPipeline.perform_blur_classification(
                image_path,
                results.object_prediction_list,
                classification_models["blur_classification"],
                model_type,
            )
            results.object_prediction_list = filtered_predictions

        ClassificationPipeline.classify_trichomes(
            image_path, results, classification_models["trichome_classification"], model_type
        )

        return results

    @staticmethod
    def process_images_in_folder(
        folder_path: str,
        detection_model,
        classification_models: dict,
        output_dir: str,
        patch_size: int,
        model_type: str,
        blur_classification_flag: bool,
    ) -> None:
        logger.info(f"Processing images in folder: {os.path.basename(folder_path)}")
        folder_number = os.path.basename(folder_path)
        result_dir = os.path.join(output_dir, folder_number)
        os.makedirs(result_dir, exist_ok=True)

        def extract_number(filename: str) -> int:
            match = re.search(r"\d+", filename)
            return int(match.group()) if match else -1

        image_files = sorted(
            [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=extract_number,
        )

        aggregated_results = []
        start_time = time.time()

        for image_file in image_files[1:]:
            image_path = os.path.join(folder_path, image_file)
            base_file_name = os.path.splitext(image_file)[0]
            image_output_dir = os.path.join(result_dir, base_file_name)
            os.makedirs(image_output_dir, exist_ok=True)

            results = ClassificationPipeline.process_image(
                image_path, detection_model, classification_models, model_type, patch_size,
                perform_blur_classification_flag=blur_classification_flag,
            )

            EndToEndUtils.export_sahi_visuals(results, image_output_dir, base_file_name)
            EndToEndUtils.save_coco_results(results, image_output_dir, base_file_name)

            distribution_json_path = os.path.join(image_output_dir, f"{base_file_name}_class_distribution.json")
            EndToEndUtils.compute_and_save_distribution(results.object_prediction_list, distribution_json_path)

            aggregated_results.extend(results.object_prediction_list)

        logger.info(f"Total time for folder: {time.time() - start_time:.2f}s")
        EndToEndUtils.aggregate_distributions(aggregated_results, result_dir)

    @staticmethod
    def process_all_folders(
        parent_folder_path: str,
        detection_model,
        classification_models: dict,
        output_dir: str,
        patch_size: int,
        model_type: str,
        perform_blur_classification_flag: bool,
    ) -> None:
        start_time = time.time()
        subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]

        for folder_path in subfolders:
            logger.info(f"Processing folder: {folder_path}")
            ClassificationPipeline.process_images_in_folder(
                folder_path=folder_path,
                detection_model=detection_model,
                classification_models=classification_models,
                output_dir=output_dir,
                patch_size=patch_size,
                model_type=model_type,
                blur_classification_flag=perform_blur_classification_flag,
            )

        logger.info(f"Total time for all folders: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    detection_model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }
    classification_models_config = {
        "trichome_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichome_image_classification/yolo/fine_tuned/YOLOv8/Medium_dataset_0.pt"
        },
        "blur_classification": {
            "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/blur_image_classification/yolo/fine_tuned/YOLOv8/Nano_dataset_0.pt"
        },
    }

    patch_size = 512
    model_type = "yolo"
    perform_blur_classification_flag = False

    detection_model = EndToEndUtils.load_detection_model(detection_model_config, patch_size)

    blur_model = ClassificationPipeline.load_classification_model(
        classification_models_config["blur_classification"], model_type
    )
    trichome_model = ClassificationPipeline.load_classification_model(
        classification_models_config["trichome_classification"], model_type
    )
    classification_models = {
        "trichome_classification": trichome_model,
        "blur_classification": blur_model,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_input_folder", type=str)
    parser.add_argument("--output_base_folder", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_base_folder, exist_ok=True)

    ClassificationPipeline.process_all_folders(
        parent_folder_path=args.parent_input_folder,
        detection_model=detection_model,
        classification_models=classification_models,
        output_dir=args.output_base_folder,
        patch_size=patch_size,
        model_type=model_type,
        perform_blur_classification_flag=perform_blur_classification_flag,
    )
