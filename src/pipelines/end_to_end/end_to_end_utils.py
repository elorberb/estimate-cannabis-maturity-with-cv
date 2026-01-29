import os
import time
import warnings
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import ImageDraw

from src.common.logging import get_logger
from src.common.detection import filter_large_objects, extend_bbox, crop_image, non_max_suppression
from src.common.metrics import compute_class_distribution, normalize_distribution, compute_iou
from src.common.io import save_json

warnings.filterwarnings(action="ignore")
logger = get_logger(__name__)


def load_detection_model(model_config, patch_size=512, device="cuda:0"):
    logger.info("Loading detection model")
    return AutoDetectionModel.from_pretrained(
        model_type="detectron2",
        model_path=model_config["checkpoint"],
        config_path=model_config["yaml_file"],
        confidence_threshold=0.5,
        image_size=patch_size,
        device=device,
    )


def run_sliced_detection(image_path, model, patch_size=512, overlap=0):
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


def export_sahi_visuals(result, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    result.export_visuals(
        export_dir=output_dir,
        text_size=1,
        rect_th=2,
        hide_labels=True,
        hide_conf=True,
        file_name=filename,
    )


def draw_uniform_boxes(result, output_dir, filename, color=(0, 255, 0)):
    os.makedirs(output_dir, exist_ok=True)
    image = result.image.copy()
    draw = ImageDraw.Draw(image)

    for pred in result.object_prediction_list:
        bbox = (int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy))
        draw.rectangle(bbox, outline=color, width=2)

    output_path = os.path.join(output_dir, f"{filename}_visuals.jpg")
    image.save(output_path)
    return output_path


def save_coco_results(result, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{filename}_raw.json")
    save_json(result.to_coco_predictions(), json_path)
    logger.info(f"Results saved: {json_path}")


def compute_and_save_distribution(predictions, output_path):
    distribution = compute_class_distribution(predictions)
    normalized = normalize_distribution(distribution)
    combined = {
        "class_distribution": dict(distribution),
        "normalized_class_distribution": normalized,
    }
    save_json(combined, output_path)
    logger.info(f"Distribution saved: {output_path}")
    return distribution, normalized


def aggregate_distributions(predictions, output_dir):
    distribution, normalized = compute_and_save_distribution(
        predictions,
        os.path.join(output_dir, "class_distribution.json")
    )
    return distribution, normalized
