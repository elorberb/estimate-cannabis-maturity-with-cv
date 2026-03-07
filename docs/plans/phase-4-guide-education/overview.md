# Phase 4: Lens Guide, Photo Quality & Education

## Goal

Improve user success rate and engagement with macro photography guidance, automatic image quality validation, and educational content about cannabis maturity science.

## Scope

- Lens Guide screen with recommended macro lenses by price tier
- Image quality check (sharpness validation before inference)
- Learn/Education screen with trichome science content adapted from published paper
- Server-side content API (update without app releases)

## New Screens & Features

**Screen 13: Lens Guide**
- Hero: "You need a macro lens" + comparison image
- Recommended lenses by tier:
  - Budget ($10-20): Clip-on 10X macro lenses
  - Mid ($20-50): Higher quality glass clip-ons
  - Best: iPhone 14 Pro+ built-in macro mode (no accessory needed)
- "How to attach" visual guide
- Example photos: good macro trichome images
- Accessible from Home screen and Camera screen

**Screen 14: Learn / Education**
- Sections:
  - **Trichome Basics**: What are trichomes, why they matter
  - **Color Stages**: Clear -> Cloudy -> Amber and what each means
  - **Stigma Color**: Green -> Orange maturity indicator
  - **Harvest Timing**: How to decide based on desired effects
  - **Using the App**: Photo tips, reading results
- Content from published paper, simplified for growers
- Illustrated with example images

**Image Quality Check (Camera Enhancement)**
- Sharpness check before submission (Laplacian variance method)
- Reuse logic from `src/data_preparation/sharpness_filter.py`
- Warning if too blurry, with tips and retake option
- User can proceed anyway

## Content Strategy

Content served as JSON from API endpoints, allowing updates without app releases:
- `GET /api/v1/content/lens-guide`
- `GET /api/v1/content/education`

## Duration

Weeks 11-12

## Dependencies

Requires Phase 1 complete (quality check integrates with camera screen). Independent of Phases 2-3 content-wise.
