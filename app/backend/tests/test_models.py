"""Tests for app.backend.models module."""

import pytest

from app.backend.models import (
    AnalysisResult,
    BoundingBox,
    TrichomeDetection,
    TrichomeDistribution,
    TrichomeType,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_creation(self, sample_bbox):
        """Test basic bounding box creation."""
        assert sample_bbox.x_min == 100
        assert sample_bbox.y_min == 100
        assert sample_bbox.x_max == 200
        assert sample_bbox.y_max == 200

    def test_width_and_height(self, sample_bbox):
        """Test width and height properties."""
        assert sample_bbox.width == 100
        assert sample_bbox.height == 100

    def test_area(self, sample_bbox):
        """Test area calculation."""
        assert sample_bbox.area == 10000

    def test_center(self, sample_bbox):
        """Test center calculation."""
        center_x, center_y = sample_bbox.center
        assert center_x == 150
        assert center_y == 150

    def test_extend(self, sample_bbox):
        """Test bounding box extension."""
        extended = sample_bbox.extend(margin=0.25, image_width=500, image_height=500)
        # 25% of 100 = 25 pixels margin
        assert extended.x_min == 75
        assert extended.y_min == 75
        assert extended.x_max == 225
        assert extended.y_max == 225

    def test_extend_clipped_to_image(self):
        """Test that extension clips to image boundaries."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        extended = bbox.extend(margin=0.5, image_width=100, image_height=100)
        assert extended.x_min == 0  # Clipped at boundary
        assert extended.y_min == 0  # Clipped at boundary


class TestTrichomeType:
    """Tests for TrichomeType enum."""

    def test_values(self):
        """Test enum values."""
        assert TrichomeType.CLEAR == 1
        assert TrichomeType.CLOUDY == 2
        assert TrichomeType.AMBER == 3

    def test_names(self):
        """Test enum names."""
        assert TrichomeType.CLEAR.name == "CLEAR"
        assert TrichomeType.CLOUDY.name == "CLOUDY"
        assert TrichomeType.AMBER.name == "AMBER"


class TestTrichomeDetection:
    """Tests for TrichomeDetection dataclass."""

    def test_creation(self, sample_detection):
        """Test detection creation."""
        assert sample_detection.trichome_type == TrichomeType.CLOUDY
        assert sample_detection.confidence == 0.85

    def test_to_dict(self, sample_detection):
        """Test conversion to dictionary."""
        d = sample_detection.to_dict()
        assert d["type"] == "cloudy"
        assert d["type_id"] == 2
        assert d["confidence"] == 0.85
        assert d["bbox"]["x_min"] == 100


class TestTrichomeDistribution:
    """Tests for TrichomeDistribution dataclass."""

    def test_creation(self, sample_distribution):
        """Test distribution creation."""
        assert sample_distribution.clear_count == 20
        assert sample_distribution.cloudy_count == 60
        assert sample_distribution.amber_count == 20

    def test_total_count(self, sample_distribution):
        """Test total count calculation."""
        assert sample_distribution.total_count == 100

    def test_ratios(self, sample_distribution):
        """Test ratio calculations."""
        assert sample_distribution.clear_ratio == 0.2
        assert sample_distribution.cloudy_ratio == 0.6
        assert sample_distribution.amber_ratio == 0.2

    def test_empty_distribution(self):
        """Test empty distribution handling."""
        empty = TrichomeDistribution()
        assert empty.total_count == 0
        assert empty.clear_ratio == 0.0
        assert empty.cloudy_ratio == 0.0
        assert empty.amber_ratio == 0.0

    def test_to_dict(self, sample_distribution):
        """Test conversion to dictionary."""
        d = sample_distribution.to_dict()
        assert d["counts"]["clear"] == 20
        assert d["counts"]["total"] == 100
        assert d["percentages"]["cloudy"] == 60.0

    def test_from_detections(self, sample_detections):
        """Test creation from detection list."""
        dist = TrichomeDistribution.from_detections(sample_detections)
        assert dist.clear_count == 1
        assert dist.cloudy_count == 2
        assert dist.amber_count == 1


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_creation(self, sample_result):
        """Test result creation."""
        assert len(sample_result.detections) == 4
        assert sample_result.image_path == "/path/to/test_image.jpg"

    def test_auto_distribution(self, sample_detections):
        """Test automatic distribution calculation."""
        result = AnalysisResult(detections=sample_detections)
        assert result.distribution is not None
        assert result.distribution.total_count == 4

    def test_to_dict(self, sample_result):
        """Test conversion to dictionary."""
        d = sample_result.to_dict()
        assert d["total_detections"] == 4
        assert d["image_path"] == "/path/to/test_image.jpg"
        assert "distribution" in d
        assert "detections" in d

    def test_empty_result(self):
        """Test empty result handling."""
        result = AnalysisResult()
        assert len(result.detections) == 0
        assert result.distribution is None
