from __future__ import annotations

import numpy as np

from cannabis_maturity.stigma_color_classifier import GREEN_DATA, ORANGE_DATA, StigmaColorClassifier


def test_classify_pure_green() -> None:
    image = np.full((10, 10, 3), GREEN_DATA[0].astype(np.uint8), dtype=np.uint8)
    green_ratio, orange_ratio = StigmaColorClassifier.classify(image)
    assert green_ratio > 0.9
    assert orange_ratio < 0.1


def test_classify_pure_orange() -> None:
    image = np.full((10, 10, 3), ORANGE_DATA[0].astype(np.uint8), dtype=np.uint8)
    green_ratio, orange_ratio = StigmaColorClassifier.classify(image)
    assert orange_ratio > 0.9
    assert green_ratio < 0.1


def test_classify_all_black_returns_zero() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    green_ratio, orange_ratio = StigmaColorClassifier.classify(image)
    assert green_ratio == 0.0
    assert orange_ratio == 0.0


def test_classify_ratios_sum_to_one() -> None:
    top = np.full((5, 10, 3), GREEN_DATA[0].astype(np.uint8), dtype=np.uint8)
    bottom = np.full((5, 10, 3), ORANGE_DATA[0].astype(np.uint8), dtype=np.uint8)
    image = np.vstack([top, bottom])
    green_ratio, orange_ratio = StigmaColorClassifier.classify(image)
    assert abs(green_ratio + orange_ratio - 1.0) < 1e-9
