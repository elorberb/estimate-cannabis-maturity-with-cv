import numpy as np
import pytest
from unittest.mock import MagicMock

from models import BoundingBox, StigmaDetection, StigmaAnalysisResult
from stigma_detector import StigmaDetector


class TestStigmaDetectorInit:
    def test_init(self):
        detector = StigmaDetector(model_path="/path/to/model.pt")
        assert detector.confidence_threshold == 0.5
        assert detector.device == "cuda:0"

    def test_init_custom_params(self):
        detector = StigmaDetector(
            model_path="/path/to/model.pt",
            confidence_threshold=0.7,
            device="cpu",
        )
        assert detector.confidence_threshold == 0.7
        assert detector.device == "cpu"


class TestStigmaDetection:
    def test_creation(self):
        bbox = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
        detection = StigmaDetection(
            bbox=bbox,
            confidence=0.9,
            orange_ratio=0.6,
            green_ratio=0.4,
        )
        assert detection.orange_ratio == 0.6
        assert detection.green_ratio == 0.4

    def test_to_dict(self):
        bbox = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
        detection = StigmaDetection(
            bbox=bbox,
            confidence=0.9,
            orange_ratio=0.65,
            green_ratio=0.35,
        )
        d = detection.to_dict()
        assert d["orange_ratio"] == 0.65
        assert d["green_ratio"] == 0.35
        assert d["confidence"] == 0.9


class TestStigmaAnalysisResult:
    def test_creation(self):
        result = StigmaAnalysisResult(image_path="/test.jpg")
        assert result.image_path == "/test.jpg"
        assert len(result.detections) == 0

    def test_auto_compute_overall_ratios(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
        detections = [
            StigmaDetection(bbox=bbox, confidence=0.9, orange_ratio=0.8, green_ratio=0.2),
            StigmaDetection(bbox=bbox, confidence=0.8, orange_ratio=0.6, green_ratio=0.4),
        ]
        result = StigmaAnalysisResult(detections=detections)
        assert result.overall_orange_ratio == pytest.approx(0.7)
        assert result.overall_green_ratio == pytest.approx(0.3)

    def test_to_dict(self):
        bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
        detections = [
            StigmaDetection(bbox=bbox, confidence=0.9, orange_ratio=0.7, green_ratio=0.3),
        ]
        result = StigmaAnalysisResult(detections=detections, image_path="/test.jpg")
        d = result.to_dict()
        assert d["num_stigmas"] == 1
        assert d["image_path"] == "/test.jpg"
        assert "overall_orange_ratio" in d


class TestColorClassification:
    def test_classify_colors(self):
        detector = StigmaDetector(model_path="/path/to/model.pt")

        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[20:40, 20:40] = [0, 165, 255]  # Orange in BGR
        img_bgr[60:80, 60:80] = [0, 255, 0]    # Green in BGR

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        mask[60:80, 60:80] = 1

        orange_ratio, green_ratio = detector._classify_colors(img_bgr, mask)

        assert orange_ratio >= 0.0
        assert green_ratio >= 0.0
        assert orange_ratio + green_ratio <= 1.01

    def test_classify_colors_empty_mask(self):
        detector = StigmaDetector(model_path="/path/to/model.pt")
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        orange_ratio, green_ratio = detector._classify_colors(img_bgr, mask)

        assert orange_ratio == 0.0
        assert green_ratio == 0.0


class TestAnalyzeWithMock:
    def test_analyze_with_mock(self, temp_image_file):
        mock_model = MagicMock()

        mock_result = MagicMock()
        mock_result.masks = None
        mock_result.boxes = MagicMock()
        mock_model.return_value = [mock_result]

        detector = StigmaDetector(model_path="/path/to/model.pt")
        detector._model = mock_model

        result = detector.analyze(temp_image_file)

        assert isinstance(result, StigmaAnalysisResult)
        assert len(result.detections) == 0
