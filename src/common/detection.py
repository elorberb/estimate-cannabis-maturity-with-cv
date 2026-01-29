import numpy as np
from src.common.metrics import compute_iou


def get_bbox_area(bbox):
    return (bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)


def filter_large_objects(predictions, size_threshold_ratio=10):
    if not predictions:
        return predictions

    sizes = [get_bbox_area(pred.bbox) for pred in predictions]
    median_size = np.median(sizes)
    threshold = median_size * size_threshold_ratio

    return [pred for pred in predictions if get_bbox_area(pred.bbox) <= threshold]


def extend_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin=0.25):
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    margin_x = int(margin * bbox_width)
    margin_y = int(margin * bbox_height)

    x_min_ext = max(0, x_min - margin_x)
    y_min_ext = max(0, y_min - margin_y)
    x_max_ext = min(image_width, x_max + margin_x)
    y_max_ext = min(image_height, y_max + margin_y)

    return x_min_ext, y_min_ext, x_max_ext, y_max_ext


def crop_image(image, x_min, y_min, x_max, y_max):
    return image[y_min:y_max, x_min:x_max]


def sort_by_confidence(predictions):
    return sorted(predictions, key=lambda x: x.score.value, reverse=True)


def filter_by_iou(predictions, reference_bbox, iou_threshold):
    return [p for p in predictions if compute_iou(reference_bbox, p.bbox) < iou_threshold]


def non_max_suppression(predictions, iou_threshold=0.7):
    if not predictions:
        return predictions

    sorted_preds = sort_by_confidence(predictions)
    keep = []

    while sorted_preds:
        highest = sorted_preds.pop(0)
        keep.append(highest)
        sorted_preds = filter_by_iou(sorted_preds, highest.bbox, iou_threshold)

    return keep
