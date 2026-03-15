from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from cannabis_maturity.models import TrichomeType
from cannabis_maturity.trichome_detector import TrichomeDetector


def _make_detector(boxes: list[tuple], class_ids: list[int]) -> TrichomeDetector:
    classification_model = MagicMock()
    cls_results = []
    for cid in class_ids:
        cls_result = MagicMock()
        probs_data = torch.zeros(3)
        probs_data[cid] = 1.0
        cls_result.probs.data = probs_data
        cls_results.append([cls_result])
    classification_model.side_effect = cls_results

    with patch("cannabis_maturity.trichome_detector.YOLO"):
        detector = TrichomeDetector("fake_path.pt", classification_model)

    detector._get_boxes = MagicMock(return_value=boxes)
    return detector


def test_detect_returns_correct_trichome_types() -> None:
    detector = _make_detector([(10.0, 10.0, 50.0, 50.0)], [1])
    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(result.detections) == 1
    assert result.detections[0].trichome_type == TrichomeType.CLEAR


def test_detect_empty_returns_empty_result() -> None:
    detector = _make_detector([], [])
    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.detections == []
    assert result.total_count == 0


def test_distribution_counts_correctly() -> None:
    boxes = [(10.0, 10.0, 50.0, 50.0), (60.0, 10.0, 90.0, 50.0), (10.0, 60.0, 50.0, 90.0), (60.0, 60.0, 90.0, 90.0)]
    detector = _make_detector(boxes, [1, 1, 0, 2])
    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.distribution[TrichomeType.CLEAR] == 2
    assert result.distribution[TrichomeType.AMBER] == 1
    assert result.distribution[TrichomeType.CLOUDY] == 1
    assert result.total_count == 4
