from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from cannabis_maturity.stigma_color_classifier import StigmaColorClassifier
from cannabis_maturity.stigma_detector import StigmaDetector


def test_detect_returns_empty_when_no_masks() -> None:
    seg_model = MagicMock()
    result = MagicMock()
    result.masks = None
    seg_model.predict.return_value = [result]

    classifier = StigmaColorClassifier()
    detector = StigmaDetector(seg_model, classifier)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detector.detect(image)
    assert res.detections == []
    assert res.total_count == 0


def test_detect_computes_ratios() -> None:
    seg_model = MagicMock()
    result = MagicMock()
    mask = np.ones((100, 100), dtype=np.float32)
    result.masks.data.cpu.return_value.numpy.return_value = np.array([mask])
    seg_model.predict.return_value = [result]

    classifier = MagicMock(spec=StigmaColorClassifier)
    classifier.classify.return_value = (0.7, 0.3)

    detector = StigmaDetector(seg_model, classifier)
    green_img = np.full((100, 100, 3), (0, 100, 0), dtype=np.uint8)
    res = detector.detect(green_img)
    assert len(res.detections) == 1
    assert res.detections[0].green_ratio == 0.7
    assert res.detections[0].orange_ratio == 0.3
    assert isinstance(res.detections[0].polygon, list)
    assert all(len(pt) == 2 for pt in res.detections[0].polygon)


def test_avg_ratios_computed_correctly() -> None:
    seg_model = MagicMock()
    result = MagicMock()
    mask = np.ones((100, 100), dtype=np.float32)
    result.masks.data.cpu.return_value.numpy.return_value = np.array([mask, mask])
    seg_model.predict.return_value = [result]

    classifier = MagicMock(spec=StigmaColorClassifier)
    classifier.classify.side_effect = [(0.8, 0.2), (0.6, 0.4)]

    detector = StigmaDetector(seg_model, classifier)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detector.detect(image)
    assert res.total_count == 2
    assert abs(res.avg_green_ratio - 0.7) < 1e-9
    assert abs(res.avg_orange_ratio - 0.3) < 1e-9
