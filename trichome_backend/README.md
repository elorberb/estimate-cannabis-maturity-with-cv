# Trichome Backend

A simple YOLO-based backend for detecting and classifying cannabis trichomes to calculate maturity distribution.

## Features

- **YOLO-based detection**: Uses Ultralytics YOLO models for trichome detection
- **Trichome classification**: Classifies trichomes as Clear, Cloudy, or Amber
- **Distribution calculation**: Computes the percentage distribution of trichome types
- **Maturity assessment**: Provides harvest recommendations based on distribution
- **Flexible architecture**: Supports both single-stage and two-stage detection

## Installation

```bash
cd trichome_backend
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from trichome_backend import TrichomeDetector, get_maturity_assessment

# Initialize detector with your YOLO model
detector = TrichomeDetector(
    detection_model_path="path/to/yolo_detection_model.pt",
    classification_model_path="path/to/yolo_classification_model.pt",  # Optional for two-stage
)

# Analyze an image
result = detector.analyze("cannabis_image.jpg")

# Get distribution
print(result.distribution.to_dict())
# Output: {'counts': {'clear': 45, 'cloudy': 120, 'amber': 35, 'total': 200},
#          'percentages': {'clear': 22.5, 'cloudy': 60.0, 'amber': 17.5}}

# Get maturity assessment
assessment = get_maturity_assessment(result.distribution)
print(assessment['stage'])  # 'peak'
print(assessment['recommendation'])  # 'Optimal harvest window for maximum THC potency.'
```

## API Reference

### TrichomeDetector

Main class for trichome detection and classification.

```python
detector = TrichomeDetector(
    detection_model_path="model.pt",        # Required: YOLO detection model
    classification_model_path=None,          # Optional: YOLO classification model
    confidence_threshold=0.5,                # Detection confidence threshold
    device="cuda:0",                         # Device: 'cuda:0', 'cpu', etc.
    bbox_extension_margin=0.25,              # Margin for bbox extension
)

result = detector.analyze(
    image="image.jpg",                       # Image path or numpy array
    filter_large_objects=True,               # Remove outlier detections
    apply_nms=True,                          # Apply non-maximum suppression
    nms_iou_threshold=0.7,                   # IoU threshold for NMS
)
```

### Data Models

- **TrichomeType**: Enum with CLEAR=1, CLOUDY=2, AMBER=3
- **BoundingBox**: Represents detection bounding box
- **TrichomeDetection**: Single detected trichome with type and confidence
- **TrichomeDistribution**: Distribution of trichome types (counts and percentages)
- **AnalysisResult**: Complete analysis result with detections and distribution

### Utility Functions

```python
from trichome_backend import (
    aggregate_distributions,    # Combine multiple distributions
    get_maturity_assessment,    # Get harvest recommendations
    save_distribution,          # Save distribution to JSON
    load_distribution,          # Load distribution from JSON
    visualize_result,           # Draw detections on image
)
```

## Trichome Types

| Type   | ID | Description |
|--------|------|-------------|
| Clear  | 1    | Early stage, immature trichomes |
| Cloudy | 2    | Peak THC potency |
| Amber  | 3    | Degrading THC, more sedative |

## Maturity Stages

| Stage | Description | Recommendation |
|-------|-------------|----------------|
| Early | >50% clear | Too early for harvest |
| Developing | Mix of clear/cloudy | Approaching harvest window |
| Peak | >60% cloudy, <15% amber | Optimal THC potency |
| Mature | 40-60% cloudy, 15-30% amber | Balanced effects |
| Late | >30% amber | Past peak, sedative effects |

## Running Tests

```bash
cd trichome_backend
pytest tests/ -v
```

## License

MIT License
