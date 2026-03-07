# Phase 3: Multi-Plant Management

## Goal

Allow growers to register plants, link analyses to specific plants, and track maturity progression over time with timeline visualization, harvest prediction, exports, and push notifications.

## Scope

- Plant CRUD (create, read, update, soft-delete)
- Link analyses to plants
- Maturity timeline chart (stacked area chart: clear/cloudy/amber over time)
- Harvest prediction algorithm (trend-based with confidence indicator)
- PDF + CSV export
- Push notifications (scheduled check reminders + maturity alerts)

## New Screens

**Screen 9: Plants List**
- Plants grouped by active/harvested
- Each card: name, strain, last analysis thumbnail, maturity badge
- "Add Plant" FAB button

**Screen 10: Add/Edit Plant**
- Form: name, strain (autocomplete), planted date, notes
- Save / Cancel

**Screen 11: Plant Detail + Timeline**
- Plant info header
- Maturity Chart: stacked area chart (clear/cloudy/amber over time)
- Harvest Prediction Banner: "Estimated X-Y days to peak potency"
- Analysis Cards Timeline: scrollable list of analysis cards
- Quick Analyze button

**Screen 12: Export/Report**
- PDF report: plant info + chart + analysis summary
- CSV export: raw data per analysis
- Share via native share sheet

See [Screens & UX](../2026-03-07-screens.md) for detailed designs.

## Harvest Prediction Algorithm

Simple trend-based prediction using the last N analyses:
1. Compute rate of change for cloudy% and amber% over time
2. If cloudy% increasing and amber% low -> predict days until cloudy peaks (>60%)
3. If amber% increasing -> predict days until thresholds (15% balanced, 30% late)
4. Display as: "Estimated X-Y days to [peak potency / balanced effects]"
5. Confidence: high (5+ analyses), medium (3-4), low (<3, insufficient data)

## Push Notifications

- **Scheduled check reminders**: User sets check schedule per plant (e.g., every 2 days)
- **Maturity alerts**: When prediction detects approaching peak/threshold
- **Implementation**: Expo Push Notifications + backend cron job

## Data Model & API

See [Data Model](../2026-03-07-data-model.md#phase-3-plant-management-additions) for schema and endpoints.

## Duration

Weeks 7-10

## Dependencies

Requires Phase 2 complete (auth needed for user-owned plants).
