import os
import time
import warnings

from src.common.logging import get_logger
from src.common.detection import filter_large_objects
from src.common.io import get_image_files
from src.pipelines.end_to_end.end_to_end_utils import (
    load_detection_model,
    run_sliced_detection,
    export_sahi_visuals,
    save_coco_results,
    compute_and_save_distribution,
    aggregate_distributions,
)

warnings.filterwarnings(action="ignore")
logger = get_logger(__name__)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def process_single_image(image_path, model, output_dir, patch_size=512):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = ensure_dir(os.path.join(output_dir, filename))

    result = run_sliced_detection(image_path, model, patch_size)
    filtered = filter_large_objects(result.object_prediction_list)
    result.object_prediction_list = filtered

    export_sahi_visuals(result, image_output_dir, filename)
    save_coco_results(result, image_output_dir, filename)

    distribution_path = os.path.join(image_output_dir, f"{filename}_class_distribution.json")
    compute_and_save_distribution(filtered, distribution_path)

    return filtered


def process_folder(folder_path, model, output_dir, patch_size=512):
    folder_name = os.path.basename(folder_path)
    result_dir = ensure_dir(os.path.join(output_dir, folder_name))

    image_files = get_image_files(folder_path)
    all_predictions = []

    start = time.time()
    for image_file in image_files[1:]:
        image_path = os.path.join(folder_path, image_file)
        predictions = process_single_image(image_path, model, result_dir, patch_size)
        all_predictions.extend(predictions)

    logger.info(f"Folder processed in {time.time() - start:.2f}s")
    aggregate_distributions(all_predictions, result_dir)

    return all_predictions


def process_all_folders(parent_folder, model, output_dir, patch_size=512):
    start = time.time()
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for folder in subfolders:
        logger.info(f"Processing: {folder}")
        process_folder(folder, model, output_dir, patch_size)

    logger.info(f"All folders processed in {time.time() - start:.2f}s")


def main():
    model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }

    images_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_3_2024_12_12/lab"
    output_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/results/faster_rcnn/day_3_2024_12_12/lab"

    ensure_dir(output_folder)
    patch_size = 512

    model = load_detection_model(model_config, patch_size)
    process_all_folders(images_folder, model, output_folder, patch_size)


if __name__ == "__main__":
    main()
