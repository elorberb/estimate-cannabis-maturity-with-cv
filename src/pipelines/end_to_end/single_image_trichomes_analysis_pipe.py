import os

from src.common.logging import Logging
from src.pipelines.end_to_end.end_to_end_utils import EndToEndUtils
from src.pipelines.end_to_end.end_to_end_with_classification import ClassificationPipeline

logger = Logging.get_logger(__name__)

DETECTION_MODEL_CONFIG = {
    "model_name": "faster_rcnn_R_50_C4_1x",
    "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
    "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
}

CLASSIFICATION_MODELS_CONFIG = {
    "trichome_classification": {
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichome_image_classification/yolo/fine_tuned/YOLOv8/Medium_dataset_0.pt"
    },
    "blur_classification": {
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/blur_image_classification/yolo/fine_tuned/YOLOv8/Nano_dataset_0.pt"
    },
}

PATCH_SIZE = 512
MODEL_TYPE = "yolo"
PERFORM_BLUR_CLASSIFICATION = False

IMAGE_PATH = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_3_2024_06_06/greenhouse/81/IMG_5296.JPG"
OUTPUT_DIR = "/home/etaylor/code_projects/thesis/src/pipelines/output_results"


class SingleImageAnalysisPipeline:
    @staticmethod
    def run(
        image_path: str,
        output_dir: str,
        patch_size: int = 512,
        model_type: str = "yolo",
        perform_blur_classification: bool = False,
    ) -> None:
        detection_model = EndToEndUtils.load_detection_model(DETECTION_MODEL_CONFIG, patch_size)

        trichome_model = ClassificationPipeline.load_classification_model(
            CLASSIFICATION_MODELS_CONFIG["trichome_classification"], model_type
        )
        blur_model = ClassificationPipeline.load_classification_model(
            CLASSIFICATION_MODELS_CONFIG["blur_classification"], model_type
        )
        classification_models = {
            "trichome_classification": trichome_model,
            "blur_classification": blur_model,
        }

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing image: {image_path}")

        results = ClassificationPipeline.process_image(
            image_path, detection_model, classification_models, model_type, patch_size,
            perform_blur_classification_flag=perform_blur_classification,
        )

        base_file_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_file_name)
        os.makedirs(image_output_dir, exist_ok=True)

        EndToEndUtils.export_sahi_visuals(results, image_output_dir, base_file_name)
        EndToEndUtils.save_coco_results(results, image_output_dir, base_file_name)

        distribution_json_path = os.path.join(image_output_dir, f"{base_file_name}_class_distribution.json")
        EndToEndUtils.compute_and_save_distribution(results.object_prediction_list, distribution_json_path)

        logger.info(f"Processing complete. Results saved to: {image_output_dir}")


if __name__ == "__main__":
    SingleImageAnalysisPipeline.run(IMAGE_PATH, OUTPUT_DIR, PATCH_SIZE, MODEL_TYPE, PERFORM_BLUR_CLASSIFICATION)
