# Sensor Systems & Telemetry Integration — Scope and architecture

## Responsibility

Build the trusted bridge from supplied NASA combustion research tables to downstream application/AI consumers. This role is data integration, not frontend UI implementation and not ML model ownership.

## Pipeline

1. **Source intake:** read CSV, JSON or Excel table.
2. **Source profile:** identify the investigation/source table and its record/experiment identifiers.
3. **Variable registry:** map each approved source column to a stable canonical `variable_id`, display name, unit and measurement class.
4. **Observation extraction:** emit numeric registered variables while retaining source-row context.
5. **Validation:** check required identifiers, finite numeric values, registry membership, measurement class, duplicate observation IDs and timestamp parseability.
6. **Packetization:** group all readings for a given source/experiment/PSI/run/source timestamp into one wide packet. If the sample timestamp is absent, use a source-row key and leave timestamp null.
7. **Export:** JSON envelope plus flattened wide CSV/DataFrame.
8. **Handoff:** frontend adapter exposes variable metadata, packets, provenance and latest packet without inventing timestamps.

## Dataset coverage

- SAME / SAME-R smoke-particle reduced data — PSI-101/102.
- BASS test log — PSI-26.
- Unified microgravity flammability master — curated cross-investigation table.
- FLEX droplet combustion — PSI-69.
- BASS-II flame-spread microgravity table — PSI-25.
- SPICE smoke-point table — PSI-107.
- SAFFIRE I–III experimental table — PSI-98/99/100.

The unified master overlaps investigation-specific source records. Do not sum its records with the source tables as though they were independent experiments.

## Important distinction

The 69 entries in `variable_registry.json` are canonical research variables. They include reported experimental variables and derived results. They are not a verified inventory of thermocouples, gas analysers, cameras or hardware sensor IDs. The separate `reference_data_engineering/sensor_dictionary_synthetic_template.json` is a development template and explicitly describes synthetic channels.
