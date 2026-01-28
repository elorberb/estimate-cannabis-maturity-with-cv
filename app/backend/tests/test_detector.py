"""Tests for app.backend.detector module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.backend.models import BoundingBox, TrichomeDetection, TrichomeType
from app.backend.detector import TrichomeDetector


class TestTrichomeDetectorInit:
    """Tests for TrichomeDetector initialization."""

    def test_init_single_stage(self):
        """Test single-stage detector initialization."""
        detector = TrichomeDetector(
            detection_model_path="/path/to/model.pt",
            confidence_threshold=0.6,
        )
        assert detector.confidence_threshold == 0.6
        assert not detector.is_two_stage

    def test_init_two_stage(self):
        """Test two-stage detector initialization."""
        detector = TrichomeDetector(
            detection_model_path="/path/to/detect.pt",
            classification_model_path="/path/to/classify.pt",
        )
        assert detector.is_two_stage


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_no_overlap(self):
        """Test IoU for non-overlapping boxes."""
        box1 = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        box2 = BoundingBox(x_min=100, y_min=100, x_max=150, y_max=150)
        iou = TrichomeDetector._compute_iou(box1, box2)
        assert iou == 0.0

    def test_full_overlap(self):
        """Test IoU for identical boxes."""
        box1 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        box2 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        iou = TrichomeDetector._compute_iou(box1, box2)
        assert iou == 1.0

    def test_partial_overlap(self):
        """Test IoU for partially overlapping boxes."""
        box1 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        box2 = BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150)
        iou = TrichomeDetector._compute_iou(box1, box2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.1428...
        assert 0.14 < iou < 0.15


class TestFilterLargeObjects:
    """Tests for filtering large objects."""

    def test_filter_large_objects(self):
        """Test that large outliers are filtered."""
        detector = TrichomeDetector(detection_model_path="/path/to/model.pt")

        detections = [
            TrichomeDetection(
                bbox=BoundingBox(x_min=0, y_min=0, x_max=40, y_max=40),  # Area: 1600
                trichome_type=TrichomeType.CLEAR,
                confidence=0.9,
            ),
            TrichomeDetection(
                bbox=BoundingBox(x_min=50, y_min=50, x_max=90, y_max=90),  # Area: 1600
                trichome_type=TrichomeType.CLOUDY,
                confidence=0.85,
            ),
            TrichomeDetection(
                bbox=BoundingBox(
                    x_min=100, y_min=100, x_max=500, y_max=500
                ),  # Area: 160000 (outlier)
                trichome_type=TrichomeType.AMBER,
                confidence=0.7,
            ),
        ]

        # Median area: 1600, threshold: 1600 * 10 = 16000
        # 160000 > 16000, so it should be filtered
        filtered = detector._filter_large_objects(detections)
        assert len(filtered) == 2

    def test_no_filtering_when_all_similar(self):
        """Test no filtering when all objects are similar size."""
        detector = TrichomeDetector(detection_model_path="/path/to/model.pt")

        detections = [
            TrichomeDetection(
                bbox=BoundingBox(x_min=0, y_min=0, x_max=40, y_max=40),
                trichome_type=TrichomeType.CLEAR,
                confidence=0.9,
            ),
            TrichomeDetection(
                bbox=BoundingBox(x_min=50, y_min=50, x_max=90, y_max=90),
                trichome_type=TrichomeType.CLOUDY,
                confidence=0.85,
            ),
        ]

        filtered = detector._filter_large_objects(detections)
        assert len(filtered) == 2


class TestNonMaxSuppression:
    """Tests for non-maximum suppression."""

    def test_nms_removes_overlapping(self):
        """Test NMS removes overlapping detections."""
        detector = TrichomeDetector(detection_model_path="/path/to/model.pt")

        detections = [
            TrichomeDetection(
                bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
                trichome_type=TrichomeType.CLEAR,
                confidence=0.9,
            ),
            TrichomeDetection(
                bbox=BoundingBox(x_min=10, y_min=10, x_max=110, y_max=110),  # High overlap
                trichome_type=TrichomeType.CLOUDY,
                confidence=0.8,
            ),
        ]

        filtered = detector._non_max_suppression(detections, iou_threshold=0.5)
        # Should keep only the higher confidence detection
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9

    def test_nms_keeps_non_overlapping(self):
        """Test NMS keeps non-overlapping detections."""
        detector = TrichomeDetector(detection_model_path="/path/to/model.pt")

        detections = [
            TrichomeDetection(
                bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
                trichome_type=TrichomeType.CLEAR,
                confidence=0.9,
            ),
            TrichomeDetection(
                bbox=BoundingBox(x_min=100, y_min=100, x_max=150, y_max=150),  # No overlap
                trichome_type=TrichomeType.CLOUDY,
                confidence=0.8,
            ),
        ]

        filtered = detector._non_max_suppression(detections, iou_threshold=0.5)
        assert len(filtered) == 2


class TestAnalyzeWithMock:
    """Tests for analyze method with mocked YOLO models."""

    def test_analyze_with_mock_model(self, temp_image_file):
        """Test analyze method with mocked YOLO."""
        # Setup mock detection results
        mock_model = MagicMock()

        # Create mock tensor-like objects that support .cpu().numpy()
        def make_tensor_mock(value):
            mock = MagicMock()
            mock.cpu.return_value.numpy.return_value = np.array(value)
            return mock

        # Create mock detection result
        mock_result = MagicMock()
        mock_boxes = MagicMock()

        # Mock the tensor arrays with cpu().numpy() support
        mock_boxes.xyxy = [make_tensor_mock([10, 10, 50, 50])]
        mock_boxes.conf = [make_tensor_mock(0.9)]
        mock_boxes.cls = [make_tensor_mock(1)]  # Clear
        mock_boxes.__len__ = lambda self: 1
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        # Create detector and inject mock directly
        detector = TrichomeDetector(detection_model_path="/path/to/model.pt")
        detector._detection_model = mock_model  # Inject mock directly

        result = detector.analyze(temp_image_file)

        assert result.distribution is not None
        assert mock_model.called


class TestClassMappings:
    """Tests for class mapping dictionaries."""

    def test_classification_mapping(self):
        """Test classification model output mapping."""
        mapping = TrichomeDetector.CLASSIFICATION_TO_TYPE
        assert mapping[0] == TrichomeType.AMBER
        assert mapping[1] == TrichomeType.CLEAR
        assert mapping[2] == TrichomeType.CLOUDY

    def test_detection_mapping(self):
        """Test detection model class mapping."""
        mapping = TrichomeDetector.DETECTION_CLASS_TO_TYPE
        assert mapping[0] is None  # Generic trichome
        assert mapping[1] == TrichomeType.CLEAR
        assert mapping[2] == TrichomeType.CLOUDY
        assert mapping[3] == TrichomeType.AMBER
