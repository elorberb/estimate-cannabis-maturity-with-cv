from __future__ import annotations

import numpy as np

from cannabis_maturity.color_classifier import ColorClassifier


def test_classify_pure_green() -> None:
    color = ColorClassifier._GREEN_DATA[0]
    image = np.full((10, 10, 3), color, dtype=np.uint8)
    green_ratio, orange_ratio = ColorClassifier.classify(image)
    assert green_ratio > 0.9
    assert orange_ratio < 0.1


def test_classify_pure_orange() -> None:
    color = ColorClassifier._ORANGE_DATA[0]
    image = np.full((10, 10, 3), color, dtype=np.uint8)
    green_ratio, orange_ratio = ColorClassifier.classify(image)
    assert orange_ratio > 0.9
    assert green_ratio < 0.1


def test_classify_all_black_returns_zero() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    green_ratio, orange_ratio = ColorClassifier.classify(image)
    assert green_ratio == 0.0
    assert orange_ratio == 0.0


def test_classify_ratios_sum_to_one() -> None:
    top = np.full((5, 10, 3), ColorClassifier._GREEN_DATA[0], dtype=np.uint8)
    bottom = np.full((5, 10, 3), ColorClassifier._ORANGE_DATA[0], dtype=np.uint8)
    image = np.vstack([top, bottom])
    green_ratio, orange_ratio = ColorClassifier.classify(image)
    assert abs(green_ratio + orange_ratio - 1.0) < 1e-9
