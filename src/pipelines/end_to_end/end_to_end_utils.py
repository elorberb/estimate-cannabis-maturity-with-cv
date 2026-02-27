import os
import time
import warnings

from PIL import ImageDraw
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src.common.io import IO
from src.common.logging import Logging
from src.common.metrics import Metrics

warnings.filterwarnings(action="ignore")
logger = Logging.get_logger(__name__)


class EndToEndUtils:
    @staticmethod
    def load_detection_model(model_config: dict, patch_size: int = 512, device: str = "cuda:0"):
        logger.info("Loading detection model")
        return AutoDetectionModel.from_pretrained(
            model_type="detectron2",
            model_path=model_config["checkpoint"],
            config_path=model_config["yaml_file"],
            confidence_threshold=0.5,
            image_size=patch_size,
            device=device,
        )

    @staticmethod
    def run_sliced_detection(image_path: str, model, patch_size: int = 512, overlap: int = 0):
        logger.info(f"Running detection on: {os.path.basename(image_path)}")
        start = time.time()
        result = get_sliced_prediction(
            image_path,
            model,
            slice_height=patch_size,
            slice_width=patch_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=True,
        )
        logger.info(f"Detection completed in {time.time() - start:.2f}s")
        return result

    @staticmethod
    def export_sahi_visuals(result, output_dir: str, filename: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        result.export_visuals(
            export_dir=output_dir,
            text_size=1,
            rect_th=2,
            hide_labels=True,
            hide_conf=True,
            file_name=filename,
        )

    @staticmethod
    def draw_uniform_boxes(result, output_dir: str, filename: str, color: tuple = (0, 255, 0)) -> str:
        os.makedirs(output_dir, exist_ok=True)
        image = result.image.copy()
        draw = ImageDraw.Draw(image)

        for pred in result.object_prediction_list:
            bbox = (int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy))
            draw.rectangle(bbox, outline=color, width=2)

        output_path = os.path.join(output_dir, f"{filename}_visuals.jpg")
        image.save(output_path)
        return output_path

    @staticmethod
    def save_coco_results(result, output_dir: str, filename: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{filename}_raw.json")
        IO.save_json(result.to_coco_predictions(), json_path)
        logger.info(f"Results saved: {json_path}")

    @staticmethod
    def compute_and_save_distribution(predictions: list, output_path: str) -> tuple:
        distribution = Metrics.compute_class_distribution(predictions)
        normalized = Metrics.normalize_distribution(distribution)
        combined = {
            "class_distribution": dict(distribution),
            "normalized_class_distribution": normalized,
        }
        IO.save_json(combined, output_path)
        logger.info(f"Distribution saved: {output_path}")
        return distribution, normalized

    @staticmethod
    def aggregate_distributions(predictions: list, output_dir: str) -> tuple:
        return EndToEndUtils.compute_and_save_distribution(
            predictions,
            os.path.join(output_dir, "class_distribution.json"),
        )
