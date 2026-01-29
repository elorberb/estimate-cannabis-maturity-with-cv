# Estimating Cannabis Flower Maturity in Greenhouse Conditions using Computer Vision

[![Paper](https://img.shields.io/badge/Paper-SSRN-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5528623)

This repository contains the implementation of our research on automated cannabis flower maturity estimation using computer vision and deep learning techniques.

## Publication

**Estimating Cannabis Flower Maturity in Greenhouse Conditions using Computer Vision**
*Etay Lorberboym, Silit Lazare, Polina Golshmid, Guy Shani*
[Read the full paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5528623)

<p align="center">
  <img src="readme_images/cannabis flowers.jpg" alt="Cannabis Flower" width="400" height="400">
</p>

## Table of Contents

1. [Abstract](#abstract)
2. [Motivation](#motivation)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Dataset](#dataset)
7. [Installation](#installation)
8. [Usage](#usage)

## Abstract

The maturity of cannabis flowers at harvest critically influences cannabinoid yield and product quality. However, conventional assessment methods rely on subjective visual inspection of trichomes and stigmas, making them inherently inconsistent. This research presents an automated framework that integrates computer vision and deep learning techniques to objectively evaluate cannabis flower maturity.

High-resolution macro images were acquired using low-cost, smartphone-based systems under both greenhouse and laboratory conditions. A two-stage analysis pipeline was implemented:
1. **Trichome Analysis**: A fine-tuned Faster R-CNN model detects and localizes trichomes, while a YOLO-based classifier categorizes them into clear, cloudy, or amber classes.
2. **Stigma Analysis**: A YOLO-based instance segmentation model delineates stigmas to compute color ratios as additional indicators of maturity.

## Motivation

### Why Cannabis Flower Maturity Matters

Cannabis flower maturity is a critical factor in determining the quality and potency of the final product. As cannabis flowers mature, their trichomes undergo a notable color transformation, shifting from clear to cloudy, and finally to amber. This color change is a vital indicator of the flower's maturity stage and directly impacts the chemical composition and potential effects of the final cannabis product.

<p align="center">
  <img src="readme_images/trichome color change cut.png" alt="Trichome Color Change">
</p>

### The Challenge

The current standard for assessing cannabis flower maturity involves manual inspection using a loupe (magnifier). This method requires observers to closely inspect the trichomes on the cannabis flower to determine their color and clarity. However, this approach is:
- Highly subjective
- Time-consuming
- Leads to significant variation between different observers

<p align="center">
  <img src="readme_images/current measurment aproach using loupe.jpg" alt="Manual Inspection Using a Loupe" width="300" height="300">
</p>

### Our Solution

We employ advanced image analysis techniques using a smartphone with a macro lens and computer vision algorithms to provide a more objective and deterministic approach to assessing maturity.

<p align="center">
  <img src="readme_images/our approach for the measurment using phone.jpg" alt="Our Approach Using Phone" width="400" height="400">
</p>

## Methodology

Our framework implements a comprehensive two-stage pipeline for cannabis maturity assessment:

### 1. Image Acquisition and Preprocessing

- **Hardware**: iPhone 14 Pro with 10X magnifying macro lens
- **Image Segmentation**: Images are segmented into patches (512x512 pixels)
- **Quality Filtering**: Edge-based sharpness metric filters out low-quality/blurry regions

### 2. Trichome Analysis Pipeline

The trichome analysis consists of two stages:

**Stage 1 - Detection**: Fine-tuned Faster R-CNN (Detectron2) model detects and localizes individual trichomes in image patches.

**Stage 2 - Classification**: YOLO-based classifier categorizes detected trichomes into three maturity classes:
- **Clear**: Immature trichomes (translucent)
- **Cloudy**: Peak maturity (milky/opaque)
- **Amber**: Over-mature (amber/brown coloration)

### 3. Stigma Analysis Pipeline

A parallel analysis pipeline for stigma (pistil) evaluation:
- **Segmentation**: YOLOv8 instance segmentation model delineates individual stigmas
- **Color Analysis**: HSV-based color classification computes ratios of:
  - White (young stigmas)
  - Orange/Brown (mature stigmas)
  - Green (background/leaves)

### 4. Maturity Score Computation

The final maturity assessment combines:
- Trichome class distribution (clear/cloudy/amber ratios)
- Stigma color ratios (white/orange)
- Aggregated metrics across multiple images per plant

## Project Structure

```
thesis/
├── src/
│   ├── app/                          # Streamlit web application
│   │   ├── pages/                    # Multi-page app components
│   │   │   ├── 1_Experiment_Tutorial.py
│   │   │   ├── 2_Experiment.py
│   │   │   └── 3_Post_Questionnaire.py
│   │   ├── streamlit_utils.py
│   │   └── ☘️_Introduction.py
│   │
│   ├── classification/               # Trichome classification models
│   │   ├── fastai/                   # FastAI-based classifiers
│   │   │   └── train_model.py        # ResNet, VGG, DenseNet training
│   │   ├── dinov2/                   # DINOv2 feature extraction
│   │   └── classification_datasets/  # Dataset preparation utilities
│   │
│   ├── segmentation/                 # Detection and segmentation models
│   │   ├── framework_handlers/       # Model framework wrappers
│   │   │   ├── detectron2_handler.py # Faster R-CNN (Detectron2)
│   │   │   └── sam_handler.py        # Segment Anything Model
│   │   ├── train_scripts/            # Training scripts
│   │   │   ├── train_detectron2_model.py
│   │   │   ├── train_ultralytics_model.py
│   │   │   └── tune_ultralytics_model.py
│   │   └── evaluation/               # Model evaluation utilities
│   │
│   ├── stigmas_detection/            # Stigma segmentation and analysis
│   │   ├── segmentation/             # YOLO segmentation models
│   │   └── pistils_analysis_pipe.py  # Color ratio computation
│   │
│   ├── pipelines/                    # End-to-end processing pipelines
│   │   ├── end_to_end/               # Full analysis pipelines
│   │   │   ├── end_to_end_pipe.py    # Main trichome pipeline
│   │   │   ├── end_to_end_pistils.py # Stigma analysis pipeline
│   │   │   └── end_to_end_with_classification.py
│   │   ├── stigma_segmentation_pipe.py
│   │   ├── trichomes_extractor.py
│   │   └── preprocessing_pipeline.py
│   │
│   ├── data_preparation/             # Image preprocessing
│   │   ├── image_loader.py
│   │   ├── patch_cutter.py           # Image patch extraction
│   │   └── sharpness_assessment.py   # Quality filtering metrics
│   │
│   ├── data_analysis/                # Results analysis
│   │   └── assessing_cannabis_maturity/
│   │
│   ├── datasets_and_annotations/     # Dataset management
│   │   ├── annotation_handler.py
│   │   └── segmentsai_handler.py     # Segments.ai integration
│   │
│   └── utils/                        # Shared utilities
│
├── config.py                         # Configuration settings
├── requirements/                     # Dependencies
└── readme_images/                    # Documentation images
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cannabis Maturity Assessment                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Image Acquisition                                   │
│                   (Smartphone + 10X Macro Lens)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Preprocessing                                       │
│              ┌─────────────────┐    ┌─────────────────┐                     │
│              │  Patch Cutting  │ -> │ Sharpness Filter │                    │
│              │   (512x512)     │    │  (Edge-based)    │                    │
│              └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│       Trichome Pipeline           │  │        Stigma Pipeline            │
│  ┌─────────────────────────────┐  │  │  ┌─────────────────────────────┐  │
│  │   Faster R-CNN Detection    │  │  │  │  YOLOv8 Segmentation        │  │
│  │      (Detectron2)           │  │  │  │                             │  │
│  └─────────────┬───────────────┘  │  │  └─────────────┬───────────────┘  │
│                ▼                  │  │                ▼                  │
│  ┌─────────────────────────────┐  │  │  ┌─────────────────────────────┐  │
│  │   YOLO Classification       │  │  │  │  HSV Color Analysis         │  │
│  │  (Clear/Cloudy/Amber)       │  │  │  │  (White/Orange/Green)       │  │
│  └─────────────────────────────┘  │  │  └─────────────────────────────┘  │
└───────────────────────────────────┘  └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Maturity Score Aggregation                            │
│           (Trichome Distribution + Stigma Color Ratios)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Dataset

The dataset was created in partnership with **Rohama Greenhouse** and spans the final 5 weeks of cannabis flower growth.

### Data Collection
- **Device**: iPhone 14 Pro with 10X magnifying lens
- **Conditions**: Greenhouse and laboratory environments
- **Coverage**: Multiple plants tracked throughout the maturation period

<p align="center">
  <img src="readme_images/images collection process.png" alt="Data Collection Process">
</p>

### Annotation
Labeling was performed through the [Segments.ai](https://segments.ai/) interface, enabling accurate and consistent annotation of:
- Individual trichomes with maturity class labels
- Stigma boundaries for instance segmentation

## Installation

```bash
# Clone the repository
git clone https://github.com/elorberb/thesis.git
cd thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/requirements.txt
```

## Usage

### Running the End-to-End Pipeline

```python
from src.pipelines.end_to_end.end_to_end_pipe import process_all_folders

# Configure model paths
model_config = {
    "model_name": "faster_rcnn_R_50_C4_1x",
    "checkpoint": "path/to/checkpoint.pth",
    "yaml_file": "path/to/config.yaml",
}

# Process images
process_all_folders(
    parent_folder_path="path/to/images",
    detection_model=model,
    output_base_dir="path/to/output",
    patch_size=512
)
```

### Running Stigma Analysis

```python
from src.pipelines.stigma_segmentation_pipe import run_stigma_segmentation_pipeline
from ultralytics import YOLO

# Load model
stigma_model = YOLO("path/to/stigma_model.pt")

# Analyze image
segmented_objects, ratios = run_stigma_segmentation_pipeline(
    image_path="path/to/image.jpg",
    model=stigma_model,
    save_dir="path/to/output"
)
```

### Training Models

**Trichome Detection (Detectron2)**:
```bash
python src/segmentation/train_scripts/train_detectron2_model.py
```

**Trichome Classification (FastAI)**:
```bash
python src/classification/fastai/train_model.py resnet50 10
```

**Stigma Segmentation (YOLO)**:
```bash
python src/segmentation/train_scripts/train_ultralytics_model.py
```

## Citation

If you use this code or our methodology in your research, please cite:

```bibtex
@article{lorberboym2025cannabis,
  title={Estimating Cannabis Flower Maturity in Greenhouse Conditions using Computer Vision},
  author={Lorberboym, Etay and Lazare, Silit and Golshmid, Polina and Shani, Guy},
  journal={SSRN},
  year={2025},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5528623}
}
```

## License

This project is for research purposes. Please contact the authors for commercial use inquiries.
