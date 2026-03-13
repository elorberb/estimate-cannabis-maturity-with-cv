from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from src.models import TrichomeType
from src.trichome_detector import TrichomeDetector


def _make_detector(boxes: np.ndarray, class_ids: list[int]) -> TrichomeDetector:
    detection_model = MagicMock()
    det_result = MagicMock()
    det_result.boxes.xyxy.cpu.return_value.numpy.return_value = boxes
    detection_model.predict.return_value = [det_result]

    classification_model = MagicMock()
    cls_results = []
    for cid in class_ids:
        cls_result = MagicMock()
        probs_data = torch.zeros(3)
        probs_data[cid] = 1.0
        cls_result.probs.data = probs_data
        cls_results.append([cls_result])
    classification_model.side_effect = cls_results

    return TrichomeDetector(detection_model, classification_model)


def test_detect_returns_correct_trichome_types() -> None:
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    detector = _make_detector(boxes, [1])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert len(result.detections) == 1
    assert result.detections[0].trichome_type == TrichomeType.CLEAR


def test_detect_empty_returns_empty_result() -> None:
    boxes = np.empty((0, 4), dtype=np.float32)
    detector = _make_detector(boxes, [])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result.detections == []
    assert result.total_count == 0


def test_distribution_counts_correctly() -> None:
    boxes = np.array([
        [10, 10, 50, 50],
        [60, 10, 90, 50],
        [10, 60, 50, 90],
        [60, 60, 90, 90],
    ], dtype=np.float32)
    detector = _make_detector(boxes, [1, 1, 0, 2])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result.distribution[TrichomeType.CLEAR] == 2
    assert result.distribution[TrichomeType.AMBER] == 1
    assert result.distribution[TrichomeType.CLOUDY] == 1
    assert result.total_count == 4
