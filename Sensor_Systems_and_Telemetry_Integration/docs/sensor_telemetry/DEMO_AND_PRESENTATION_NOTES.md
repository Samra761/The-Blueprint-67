# Demo/presentation notes for Sensor Systems & Telemetry Integration

## 30-second explanation

“My role is Sensor Systems and Telemetry Integration. I built the data layer that takes the supplied combustion research tables, maps approved fields to a canonical variable registry, validates readings and metadata, and converts them into standardized wide-format packets. Each packet contains all available variables for a source record or timestamp. I preserve source references, quality flags and measurement types so the dashboard and AI modules can consume traceable data.”

## Show during demo

1. Open `config/variable_registry.json` and show a canonical ID such as `FLEX::pressure_kpa`.
2. Run `python src/run_all_sources.py`.
3. Open `output/all_sources/build_summary.json` to show the seven processed sources and counts.
4. Open `output/all_sources/all_sources_wide_packets.json` and show `readings`, `units`, `reading_metadata` and `source_row_metadata`.
5. Explain why timestamps remain null for experiment-level records without sample timestamps.
6. Show the frontend adapter output or the frontend handoff guide.

## Honest limitation statement

“These supplied tables are mostly reduced or experiment-level results rather than a single continuous sensor stream. I therefore preserve experiment context and source traceability, and I do not fabricate live timestamps or claim derived research variables are physical sensor channels.”
