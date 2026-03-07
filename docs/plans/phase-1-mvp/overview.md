# Phase 1: MVP — "Snap & Analyze"

## Goal

Deliver a working mobile app where cannabis growers can take a macro photo and get instant maturity assessment with trichome/stigma analysis and harvest recommendations.

## Scope

- FastAPI server with `/analyze` endpoint
- Modal GPU function wrapping existing TrichomeDetector + StigmaDetector
- Supabase for image storage + results database
- React Native (Expo) app with 6 screens: Home, Camera, Analyzing, Results, Review/Correct, History
- No auth — anonymous device_id for history tracking
- User correction flow for trichome + stigma classifications

## Screens

- **Home**: Analyze button + recent history preview
- **Camera/Upload**: Expo Camera + gallery picker
- **Analyzing**: Loading animation with status text
- **Results**: Annotated image, distribution stats, maturity badge, recommendation
- **Review/Correct**: Grid of cropped detections, tap to correct classifications
- **History**: Past analyses list

See [Screens & UX](../2026-03-07-screens.md) for detailed designs.

## Architecture

See [Architecture](../2026-03-07-architecture.md) for full system diagram and project structure.

## Data Model

MVP uses the `analyses` table with anonymous `device_id` tracking. See [Data Model](../2026-03-07-data-model.md) for schema.

## Duration

Weeks 1-5

## Verification

1. **Backend**: `cd app/api && uv run pytest tests/ -v`
2. **Modal**: `modal run modal/inference.py` — test with sample image
3. **Mobile**: `cd app/mobile && npx expo start` — test on device/simulator
4. **Integration**: Upload image via app -> verify results match direct Python inference
5. **Corrections**: Submit corrections -> verify stored in Supabase alongside originals
