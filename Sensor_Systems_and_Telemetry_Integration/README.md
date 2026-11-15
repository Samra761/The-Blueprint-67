# Flame in Freefall — Sensor Systems & Telemetry Integration

**Package version:** 2.1.5 (data-quality fixes and verification; see [`CHANGELOG.md`](CHANGELOG.md))  
**Project:** NASA Flame in Freefall  
**Module:** Sensor Systems & Telemetry Integration

## Purpose

This module provides a provenance-preserving path from supplied combustion-research datasets to normalized telemetry-style observations and wide packets that downstream project components can consume. It covers ingestion, research-variable mapping, validation, packet generation, exports, and documented handoff adapters.

It does **not** own the project-wide database/API, frontend application, or machine-learning models. Those remain with their respective project leads.

## What is included

- A 69-entry canonical **research-variable** registry and seven dataset profiles. These are research variables, not a verified inventory of physical sensor hardware or sensor IDs.
- CSV, JSON, XLSX, and XLS ingestion, validation, provenance retention, and wide-packet generation.
- Python interfaces returning dictionary and DataFrame outputs, with JSON/CSV export.
- A frontend adapter and integration example, tests, and technical documentation.
- Supplemental PSI source-table inventory and packetization layer.
- A SAME-R handoff containing the original HKP CSV inputs, normalized wide packets, channel mapping contract, and ML preprocessing reference.

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Research-variable registry and dataset profiles |
| `src/` | Base import, packet builder, and build commands |
| `data/processed/` | Processed source tables and base outputs |
| `output/all_sources/` | Generated multi-source packet exports and summary |
| `integration_update/` | Supplemental PSI tables, inventories, packets, and source review |
| `ml_lead_handoff/` | SAME-R source files, normalized packets, and ML handoff contract |
| `frontend_adapter/` | Frontend payload adapter and example integration |
| `tests/` | Automated tests |
| `docs/` | Schema, validation, dataset, build, and integration documentation |
| `reference_data_engineering/` | Clearly labelled team templates; synthetic templates are not hardware documentation |

## Quick start

Run commands from this directory. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Rebuild the base seven-source outputs (applies the documented source corrections first; the build is deterministic, so a rebuild reproduces the packaged files byte for byte):

```bash
python src/apply_source_corrections.py
python src/run_all_sources.py
```

Run the base telemetry and frontend-adapter tests:

```bash
python -m pytest -q tests frontend_adapter/test_adapter.py
```

Run the supplemental PSI importer tests:

```bash
PYTHONPATH=integration_update/src python -m pytest -q integration_update/tests
```

## Recorded build outputs

The packaged baseline build records **10,627 mapped observations**, **1,519 wide packets**, and **zero importer errors**. Those 1,519 packets are untimestamped source-row packets because the source tables do not provide sample-level timestamps. The supplemental PSI layer contains **22 original CSV tables**, **2,009 source-row packets**, and **16,042 numeric candidates**. These layers overlap; do not add their packet counts as if they represented unique observations.

The SAME-R handoff contains **43 original HKP CSV files**, **58,386 normalized packets**, and **41 tests**. The ML loader's documented filtering uses 40 files after excluding duplicate copies and a test in which the heater did not fire. The packets are normalized raw inputs, not engineered model features. ML-owned baseline correction, rolling-window features, ratios, and model inference must be performed by the ML pipeline.

## Data and timestamp rules

- Preserve original source values, units, source identifiers, provenance, and quality flags.
- Use `timestamp_utc: null` when a genuine sample-level UTC timestamp is not available or its semantics have not been approved. Do not turn a test day, GMT label, or elapsed-time value into UTC without documented agreement.
- P-Trak `-999` is a missing-value sentinel, not zero; normalized `ptrak` is null with a quality flag.
- DustTrak B is internally consistent (see `ml_lead_handoff/SAME_R_DUSTTRAK_CHECK.json`), but smoke-specific calibration is not established; do not claim calibrated smoke concentration. DustTrak A readings are not usable as mass concentrations.
- Corrections to processed values are made only through `config/source_corrections.json` (with evidence) and `src/apply_source_corrections.py`; the original NASA tables are never edited.
- Keep the 52 ground-reference rows distinguishable from microgravity observations.
- Do not treat the 69 research-variable registry entries or synthetic sensor template as confirmed physical sensor channels.
- Do not silently resolve ambiguous source units, headers, or values.

## Integration ownership

- **Sensor Systems & Telemetry Integration:** ingestion, mapping, validation, provenance, normalized packet exports, and handoff adapters.
- **Machine Learning:** ML preprocessing, feature engineering, training, inference, and evaluation.
- **Frontend:** application UI, dashboard, visualization, and replay/feed consumption.
- **Data Engineering:** canonical upstream database/API, acquisition, schema governance, and source-resolution rules.

The current handoff documents ML-to-Frontend integration and data compatibility between the ML inputs and normalized SAME-R packets. The Frontend package is not yet directly consuming this module's packet output. Do not claim a complete live end-to-end integration until that connection has been implemented and tested.

## Documentation map

- [`docs/README.md`](docs/README.md) — documentation index.
- [`docs/sensor_telemetry/FRONTEND_INTEGRATION.md`](docs/sensor_telemetry/FRONTEND_INTEGRATION.md) — current integration status, channel contract, caveats, and open decisions.
- [`docs/sensor_telemetry/TELEMETRY_SCHEMA.md`](docs/sensor_telemetry/TELEMETRY_SCHEMA.md) — packet schema.
- [`docs/sensor_telemetry/VALIDATION_AND_DATA_QUALITY.md`](docs/sensor_telemetry/VALIDATION_AND_DATA_QUALITY.md) and [`docs/VALIDATION_RULES.md`](docs/VALIDATION_RULES.md) — validation policy.
- [`integration_update/README.md`](integration_update/README.md) — supplemental PSI data layer.
- [`ml_lead_handoff/README.md`](ml_lead_handoff/README.md) — SAME-R handoff contents and limitations.
- [`frontend_adapter/README.md`](frontend_adapter/README.md) — frontend adapter usage.

## Scope note

This repository folder is the Sensor Systems & Telemetry Integration contribution. It is not a replacement for the other project leads' repositories or deliverables.


## v2.1.6 integration update

- Added the three original PSI-100 SAFFIRE-III exports under `data/raw/psi100/`.
- Added a reproducible PSI-100 workbook packet builder and normalized 11,450-row / 44-channel RawData stream.
- Verified that the workbook RawData sheet matches the detailed cleaned CSV across all 47 mapped columns and 11,450 rows; these are not counted twice.
- Promoted SAME-R `source_epoch_s` to `timestamp_utc` only after all 58,386 source GMT labels matched exactly. Original epoch and elapsed-time fields are preserved.
- Extended the frontend adapter to support channel-based packets and JSONL loading; added a Streamlit reference consumer for JSON/JSONL.
- Existing 7-source research-table packet layer and ML-owned preprocessing/model files remain separate and unchanged in ownership.
- Remaining sensor-side evidence gaps: physical hardware identity, units not stated in source headers, and DustTrak smoke calibration. Actual dashboard wiring remains frontend-owned.

Latest v2.1.6 regression test result: 23 passed. See `docs/BUILD_AND_TEST_RESULTS.md`.
