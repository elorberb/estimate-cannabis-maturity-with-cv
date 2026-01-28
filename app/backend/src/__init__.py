"""
Trichome Backend Package

A simple YOLO-based backend for detecting and classifying cannabis trichomes
to calculate maturity distribution.

Example usage:
    from detector import TrichomeDetector
    from distribution import get_maturity_assessment

    detector = TrichomeDetector(detection_model_path="model.pt")
    result = detector.analyze("image.jpg")
    print(result.distribution.to_dict())
"""

__version__ = "0.1.0"
