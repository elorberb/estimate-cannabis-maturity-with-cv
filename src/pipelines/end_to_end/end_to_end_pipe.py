import os
import time
import warnings

from src.common.detection import Detection
from src.common.io import IO
from src.common.logging import Logging
from src.pipelines.end_to_end.end_to_end_utils import EndToEndUtils

warnings.filterwarnings(action="ignore")
logger = Logging.get_logger(__name__)


class EndToEndPipe:
    @staticmethod
    def process_single_image(image_path: str, model, output_dir: str, patch_size: int = 512) -> list:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, filename)
        os.makedirs(image_output_dir, exist_ok=True)

        result = EndToEndUtils.run_sliced_detection(image_path, model, patch_size)
        filtered = Detection.filter_large_objects(result.object_prediction_list)
        result.object_prediction_list = filtered

        EndToEndUtils.export_sahi_visuals(result, image_output_dir, filename)
        EndToEndUtils.save_coco_results(result, image_output_dir, filename)

        distribution_path = os.path.join(image_output_dir, f"{filename}_class_distribution.json")
        EndToEndUtils.compute_and_save_distribution(filtered, distribution_path)

        return filtered

    @staticmethod
    def process_folder(folder_path: str, model, output_dir: str, patch_size: int = 512) -> list:
        folder_name = os.path.basename(folder_path)
        result_dir = os.path.join(output_dir, folder_name)
        os.makedirs(result_dir, exist_ok=True)

        image_files = IO.get_image_files(folder_path)
        all_predictions = []

        start = time.time()
        for image_file in image_files[1:]:
            image_path = os.path.join(folder_path, image_file)
            predictions = EndToEndPipe.process_single_image(image_path, model, result_dir, patch_size)
            all_predictions.extend(predictions)

        logger.info(f"Folder processed in {time.time() - start:.2f}s")
        EndToEndUtils.aggregate_distributions(all_predictions, result_dir)

        return all_predictions

    @staticmethod
    def process_all_folders(parent_folder: str, model, output_dir: str, patch_size: int = 512) -> None:
        start = time.time()
        subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

        for folder in subfolders:
            logger.info(f"Processing: {folder}")
            EndToEndPipe.process_folder(folder, model, output_dir, patch_size)

        logger.info(f"All folders processed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    _model_config = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/trichomes_detection/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml",
    }

    _images_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_3_2024_12_12/lab"
    _output_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/results/faster_rcnn/day_3_2024_12_12/lab"

    os.makedirs(_output_folder, exist_ok=True)
    _model = EndToEndUtils.load_detection_model(_model_config, 512)
    EndToEndPipe.process_all_folders(_images_folder, _model, _output_folder)
