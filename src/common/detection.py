import os

import cv2
import numpy as np

from src.common.metrics import Metrics


class Detection:
    @staticmethod
    def get_bbox_area(bbox) -> float:
        return (bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)

    @staticmethod
    def filter_large_objects(predictions: list, size_threshold_ratio: int = 10) -> list:
        if not predictions:
            return predictions

        sizes = [Detection.get_bbox_area(pred.bbox) for pred in predictions]
        median_size = np.median(sizes)
        threshold = median_size * size_threshold_ratio

        return [pred for pred in predictions if Detection.get_bbox_area(pred.bbox) <= threshold]

    @staticmethod
    def extend_bbox(
        x_min: int, y_min: int, x_max: int, y_max: int,
        image_width: int, image_height: int, margin: float = 0.25,
    ) -> tuple[int, int, int, int]:
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        margin_x = int(margin * bbox_width)
        margin_y = int(margin * bbox_height)

        return (
            max(0, x_min - margin_x),
            max(0, y_min - margin_y),
            min(image_width, x_max + margin_x),
            min(image_height, y_max + margin_y),
        )

    @staticmethod
    def crop_image(image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
        return image[y_min:y_max, x_min:x_max]

    @staticmethod
    def sort_by_confidence(predictions: list) -> list:
        return sorted(predictions, key=lambda x: x.score.value, reverse=True)

    @staticmethod
    def filter_by_iou(predictions: list, reference_bbox, iou_threshold: float) -> list:
        return [p for p in predictions if Metrics.compute_iou(reference_bbox, p.bbox) < iou_threshold]

    @staticmethod
    def non_max_suppression(predictions: list, iou_threshold: float = 0.7) -> list:
        if not predictions:
            return predictions

        sorted_preds = Detection.sort_by_confidence(predictions)
        keep = []

        while sorted_preds:
            highest = sorted_preds.pop(0)
            keep.append(highest)
            sorted_preds = Detection.filter_by_iou(sorted_preds, highest.bbox, iou_threshold)

        return keep

    @staticmethod
    def get_bboxes_from_segmentation(segmentation_bitmap: np.ndarray, margin: int = 5) -> dict:
        unique_labels = np.unique(segmentation_bitmap)
        unique_labels = unique_labels[unique_labels != 0]
        bboxes = {}

        for label in unique_labels:
            label_mask = segmentation_bitmap == label
            coords = np.column_stack(np.where(label_mask))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            y_min = max(y_min - margin, 0)
            x_min = max(x_min - margin, 0)
            y_max = min(y_max + margin, segmentation_bitmap.shape[0])
            x_max = min(x_max + margin, segmentation_bitmap.shape[1])

            bboxes[f"bbox_{label}"] = (y_min, y_max, x_min, x_max)

        return bboxes

    @staticmethod
    def crop_and_store_bboxes(image: np.ndarray, bboxes: dict, save_dir: str) -> dict:
        cut_images = {}
        for label, (y_min, y_max, x_min, x_max) in bboxes.items():
            cut_image = image[y_min:y_max, x_min:x_max]
            cut_images[label] = cut_image
            save_path = os.path.join(save_dir, f"{label}.png")
            cv2.imwrite(save_path, cut_image)
        return cut_images
