# Cannabis Analysis Backend

YOLO-based backend for cannabis maturity assessment through trichome and stigma detection.

## Features

- **Trichome Detection**: Classifies trichomes as Clear, Cloudy, or Amber
- **Stigma Detection**: Detects pistil/stigma for maturity assessment
- **Distribution Calculation**: Computes percentage distributions
- **Maturity Assessment**: Harvest recommendations based on analysis

## Project Structure

```
backend/
├── src/
│   ├── models.py              # Pydantic models
│   ├── trichome_detector.py   # TrichomeDetector
│   ├── stigma_detector.py     # StigmaDetector
│   ├── distribution.py        # Distribution utils
│   ├── utils.py               # Image utilities
│   └── config.py              # Configuration
├── tests/
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd app/backend
uv sync
```

## Usage

```python
from trichome_detector import TrichomeDetector
from stigma_detector import StigmaDetector

# Trichome analysis
trichome_detector = TrichomeDetector(detection_model_path="model.pt")
result = trichome_detector.analyze("image.jpg")
print(result.distribution.to_dict())

# Stigma detection
stigma_detector = StigmaDetector(model_path="stigma_model.pt")
stigmas = stigma_detector.detect("image.jpg")
```

## Running Tests

```bash
cd app/backend
uv run pytest tests/ -v
```

## License

MIT License
