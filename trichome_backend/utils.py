"""Utility functions for image processing and visualization."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .models import AnalysisResult, TrichomeType

logger = logging.getLogger(__name__)

# Color mapping for visualization (BGR format for OpenCV)
TYPE_COLORS_BGR = {
    TrichomeType.CLEAR: (128, 128, 128),  # Grey
    TrichomeType.CLOUDY: (255, 255, 255),  # White
    TrichomeType.AMBER: (0, 165, 255),  # Orange
}

# Color mapping (RGB format)
TYPE_COLORS_RGB = {
    TrichomeType.CLEAR: (128, 128, 128),  # Grey
    TrichomeType.CLOUDY: (255, 255, 255),  # White
    TrichomeType.AMBER: (255, 165, 0),  # Orange
}


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format)

    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save an image to file.

    Args:
        image: Image as numpy array (BGR format)
        output_path: Path to save image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Image saved to {output_path}")


def draw_detections(
    image: np.ndarray,
    result: AnalysisResult,
    draw_labels: bool = False,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.

    Args:
        image: Input image (BGR format)
        result: AnalysisResult containing detections
        draw_labels: Whether to draw class labels
        line_thickness: Thickness of bounding box lines

    Returns:
        Image with drawn detections
    """
    output = image.copy()

    for detection in result.detections:
        color = TYPE_COLORS_BGR.get(detection.trichome_type, (255, 255, 255))
        bbox = detection.bbox

        # Draw rectangle
        pt1 = (int(bbox.x_min), int(bbox.y_min))
        pt2 = (int(bbox.x_max), int(bbox.y_max))
        cv2.rectangle(output, pt1, pt2, color, line_thickness)

        # Draw label if requested
        if draw_labels:
            label = detection.trichome_type.name.lower()
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, 1
            )

            # Draw background rectangle
            cv2.rectangle(
                output,
                (pt1[0], pt1[1] - text_height - 5),
                (pt1[0] + text_width, pt1[1]),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                output,
                label,
                (pt1[0], pt1[1] - 5),
                font,
                font_scale,
                (0, 0, 0),
                1,
            )

    return output


def visualize_result(
    image_path: Union[str, Path],
    result: AnalysisResult,
    output_path: Optional[Union[str, Path]] = None,
    draw_labels: bool = False,
) -> np.ndarray:
    """
    Create visualization of detection results.

    Args:
        image_path: Path to original image
        result: AnalysisResult to visualize
        output_path: Optional path to save visualization
        draw_labels: Whether to draw class labels

    Returns:
        Visualization image
    """
    image = load_image(image_path)
    visualization = draw_detections(image, result, draw_labels=draw_labels)

    if output_path:
        save_image(visualization, output_path)

    return visualization


def crop_detection(
    image: np.ndarray,
    result: AnalysisResult,
    detection_index: int,
    margin: float = 0.0,
) -> np.ndarray:
    """
    Crop a single detection from the image.

    Args:
        image: Input image (BGR format)
        result: AnalysisResult containing detections
        detection_index: Index of detection to crop
        margin: Optional margin to add around crop (ratio)

    Returns:
        Cropped image region
    """
    if detection_index >= len(result.detections):
        raise IndexError(f"Detection index {detection_index} out of range")

    detection = result.detections[detection_index]
    bbox = detection.bbox

    if margin > 0:
        h, w = image.shape[:2]
        bbox = bbox.extend(margin, w, h)

    crop = image[
        int(bbox.y_min) : int(bbox.y_max),
        int(bbox.x_min) : int(bbox.x_max),
    ]

    return crop


def resize_image(
    image: np.ndarray,
    max_size: int = 1024,
    maintain_aspect: bool = True,
) -> np.ndarray:
    """
    Resize image to fit within max_size while maintaining aspect ratio.

    Args:
        image: Input image
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if maintain_aspect:
        if max(h, w) <= max_size:
            return image

        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = new_h = max_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_image(image_path: Union[str, Path]) -> dict:
    """
    Validate an image file and return basic info.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image info or error
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return {"valid": False, "error": "File does not exist"}

    if not image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return {"valid": False, "error": f"Unsupported format: {image_path.suffix}"}

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"valid": False, "error": "Could not decode image"}

        h, w, c = image.shape
        return {
            "valid": True,
            "path": str(image_path),
            "width": w,
            "height": h,
            "channels": c,
            "size_bytes": image_path.stat().st_size,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
