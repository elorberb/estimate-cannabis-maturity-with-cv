"""Tests for src.utils module."""

import numpy as np
import pytest
import cv2

from utils import (
    draw_detections,
    load_image,
    resize_image,
    save_image,
    validate_image,
)


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_valid_image(self, temp_image_file):
        """Test loading a valid image file."""
        image = load_image(temp_image_file)
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Height, Width, Channels

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent file raises ValueError."""
        with pytest.raises(ValueError, match="Could not load image"):
            load_image(tmp_path / "nonexistent.jpg")


class TestSaveImage:
    """Tests for save_image function."""

    def test_save_image(self, tmp_path, sample_image):
        """Test saving an image."""
        output_path = tmp_path / "output.jpg"
        save_image(sample_image, output_path)
        assert output_path.exists()

    def test_save_creates_directory(self, tmp_path, sample_image):
        """Test that save creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "output.jpg"
        save_image(sample_image, output_path)
        assert output_path.exists()


class TestDrawDetections:
    """Tests for draw_detections function."""

    def test_draw_detections(self, sample_image, sample_result):
        """Test drawing detections on image."""
        output = draw_detections(sample_image, sample_result)
        assert output.shape == sample_image.shape
        # Verify it's a copy, not the original
        assert output is not sample_image

    def test_draw_detections_with_labels(self, sample_image, sample_result):
        """Test drawing detections with labels."""
        output = draw_detections(sample_image, sample_result, draw_labels=True)
        assert output.shape == sample_image.shape


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_large_image(self):
        """Test resizing a large image."""
        large_image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        resized = resize_image(large_image, max_size=1000)
        assert max(resized.shape[:2]) == 1000

    def test_resize_maintains_aspect_ratio(self):
        """Test that resize maintains aspect ratio."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)  # 1:2 ratio
        resized = resize_image(image, max_size=400)
        h, w = resized.shape[:2]
        assert w == 400
        assert h == 200  # Maintains 1:2 ratio

    def test_no_resize_if_small(self):
        """Test that small images are not resized."""
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = resize_image(small_image, max_size=500)
        assert resized.shape == small_image.shape


class TestValidateImage:
    """Tests for validate_image function."""

    def test_validate_valid_image(self, temp_image_file):
        """Test validating a valid image."""
        info = validate_image(temp_image_file)
        assert info["valid"] is True
        assert info["width"] == 512
        assert info["height"] == 512

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validating nonexistent file."""
        info = validate_image(tmp_path / "nonexistent.jpg")
        assert info["valid"] is False
        assert "does not exist" in info["error"]

    def test_validate_unsupported_format(self, tmp_path):
        """Test validating unsupported file format."""
        bad_file = tmp_path / "file.txt"
        bad_file.write_text("not an image")
        info = validate_image(bad_file)
        assert info["valid"] is False
        assert "Unsupported format" in info["error"]
