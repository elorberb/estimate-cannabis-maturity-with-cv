"""Pytest fixtures for backend tests."""

import numpy as np
import pytest

from models import (
    AnalysisResult,
    BoundingBox,
    TrichomeDetection,
    TrichomeDistribution,
    TrichomeType,
)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)


@pytest.fixture
def sample_detection(sample_bbox):
    """Create a sample detection."""
    return TrichomeDetection(
        bbox=sample_bbox,
        trichome_type=TrichomeType.CLOUDY,
        confidence=0.85,
    )


@pytest.fixture
def sample_detections():
    """Create a list of sample detections."""
    return [
        TrichomeDetection(
            bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50),
            trichome_type=TrichomeType.CLEAR,
            confidence=0.9,
        ),
        TrichomeDetection(
            bbox=BoundingBox(x_min=60, y_min=60, x_max=100, y_max=100),
            trichome_type=TrichomeType.CLOUDY,
            confidence=0.85,
        ),
        TrichomeDetection(
            bbox=BoundingBox(x_min=110, y_min=110, x_max=150, y_max=150),
            trichome_type=TrichomeType.CLOUDY,
            confidence=0.88,
        ),
        TrichomeDetection(
            bbox=BoundingBox(x_min=160, y_min=160, x_max=200, y_max=200),
            trichome_type=TrichomeType.AMBER,
            confidence=0.75,
        ),
    ]


@pytest.fixture
def sample_distribution():
    """Create a sample distribution."""
    return TrichomeDistribution(clear_count=20, cloudy_count=60, amber_count=20)


@pytest.fixture
def sample_result(sample_detections):
    """Create a sample analysis result."""
    return AnalysisResult(
        detections=sample_detections,
        image_path="/path/to/test_image.jpg",
    )


@pytest.fixture
def sample_image():
    """Create a sample test image (numpy array)."""
    # Create a simple 512x512 RGB image
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def temp_image_file(tmp_path, sample_image):
    """Create a temporary image file."""
    import cv2

    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), sample_image)
    return image_path
