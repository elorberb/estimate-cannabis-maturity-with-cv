# Trichome Backend

A simple YOLO-based backend microservice for detecting and classifying cannabis trichomes to calculate maturity distribution.

## Features

- **YOLO-based detection**: Uses Ultralytics YOLO models for trichome detection
- **Trichome classification**: Classifies trichomes as Clear, Cloudy, or Amber
- **Distribution calculation**: Computes the percentage distribution of trichome types
- **Maturity assessment**: Provides harvest recommendations based on distribution

## Project Structure

```
backend/
├── src/                  # Source code
│   ├── __init__.py
│   ├── models.py         # Data models
│   ├── detector.py       # TrichomeDetector class
│   ├── distribution.py   # Distribution calculations
│   ├── utils.py          # Utilities
│   └── config.py         # Configuration
├── tests/                # Tests
├── pyproject.toml
├── .gitignore
└── README.md
```

## Installation

```bash
cd app/backend
uv sync
```

## Quick Start

```python
from detector import TrichomeDetector
from distribution import get_maturity_assessment

# Initialize detector
detector = TrichomeDetector(
    detection_model_path="path/to/yolo_model.pt",
    classification_model_path="path/to/classify_model.pt",  # Optional
)

# Analyze an image
result = detector.analyze("cannabis_image.jpg")

# Get distribution
print(result.distribution.to_dict())
# {'counts': {'clear': 45, 'cloudy': 120, 'amber': 35, 'total': 200},
#  'percentages': {'clear': 22.5, 'cloudy': 60.0, 'amber': 17.5}}

# Get maturity assessment
assessment = get_maturity_assessment(result.distribution)
print(assessment['recommendation'])
```

## Running Tests

```bash
cd app/backend
uv run pytest tests/ -v
```

## Trichome Types

| Type   | ID | Description |
|--------|------|-------------|
| Clear  | 1    | Early stage, immature |
| Cloudy | 2    | Peak THC potency |
| Amber  | 3    | Degrading THC, sedative |

## License

MIT License
