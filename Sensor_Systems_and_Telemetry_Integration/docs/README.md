# Documentation Index

This folder contains the technical documentation for the Sensor Systems & Telemetry Integration module.

## Core documentation

- `BUILD_AND_TEST_RESULTS.md` — recorded build counts and test interface.
- `VALIDATION_RULES.md` — validation rules used by the ingestion and packet pipeline.
- `sensor_telemetry/TELEMETRY_SCHEMA.md` — packet structure and field definitions.
- `sensor_telemetry/VALIDATION_AND_DATA_QUALITY.md` — quality checks, missing values, and provenance rules.
- `sensor_telemetry/DATASET_CATALOG.md` — source dataset catalogue and coverage.
- `sensor_telemetry/RESOURCE_REFERENCE.md` — source/resource references.
- `sensor_telemetry/FRONTEND_INTEGRATION.md` — ML, frontend, and telemetry integration status and shared channel contract.
- `sensor_telemetry/FRONTEND_HANDOFF.md` — frontend-facing handoff notes.
- `sensor_telemetry/DEMO_AND_PRESENTATION_NOTES.md` — demo caveats and presentation guidance.
- `sensor_telemetry/SUBMISSION_CHECKLIST.md` — module completion and outstanding external dependencies.

The existing `sensor_telemetry/` path is retained for compatibility with the packaged documentation links. It is an internal legacy folder name, not the official name of the project responsibility.

- [PSI-100 SAFFIRE-III integration](sensor_telemetry/PSI100_SAFFIRE_III_INTEGRATION.md) — source workbook packetization, channel registry and timestamp/overlap rules.
- [SAME-R timestamp and channel update](sensor_telemetry/SAME_R_CHANNELS_AND_TIMESTAMPS.md) — validated UTC promotion and channel caveats.
