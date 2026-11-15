# PSI-100 SAFFIRE-III sensor telemetry integration

## Integrated source files

All three PSI-100 exports are preserved unchanged in `data/raw/psi100/`:

- `PSI-100_Raw_Data_Raw_telemetry_file.csv` — 3,972 rows, 47 columns.
- `PSI-100_Raw_Data_Saffire_Data_Table2017.161.00.42.03-EU.csv` — 11,450 rows, 47 columns.
- `PSI-100_Raw_Data_Saffire-III_1_Hz_Telemetry_Data.xlsx` — 6 worksheets.

The workbook `RawData` worksheet is the canonical combined stream. It contains 11,450 populated records and 44 telemetry channels. The normalized packet output contains 11,450 packets: 11,449 with a source-derived UTC timestamp and one invalid terminal timestamp retained with a quality flag.

## Source reconciliation and overlap control

The workbook `RawData` sheet and the detailed cleaned CSV have the same 11,450 data rows. A 47-column mapping was checked across every row; all values match within numeric tolerance 1e-7. The detailed CSV is therefore treated as a corroborating export, not as an independent second stream.

The workbook also contains `CygnusData`, `SaffireData`, `CygnusSmokeDetectorData`, `CygnusPressureData`, and `CygnusTemperatureData`. These are views/projections of the combined workbook information. They are inventoried but are not stacked into the canonical packet count, avoiding double counting.

The separate raw telemetry CSV has 3,972 rows and coarse scientific-notation Cygnus timestamps. It is preserved unchanged but is not used as the canonical time-indexed stream.

## Timestamp handling

- `CygnusTimestamp` is interpreted as Unix epoch milliseconds only when the value is within a plausible epoch-millisecond range.
- `timestamp_utc` is generated as ISO-8601 UTC from that source field. The original `source_epoch_ms` remains in every packet.
- One terminal source row has an extreme invalid `CygnusTimestamp`; it is preserved with `timestamp_utc: null` and `invalid_cygnus_timestamp_source_value`.
- `RelativeSeconds` is preserved exactly as named. Its scale conflicts with explicit millisecond relative-time fields elsewhere in the workbook; no unit conversion is applied.
- Duplicate timestamp values are retained: the build found 7 duplicate timestamp rows and 11,442 unique valid source timestamps among 11,449 timestamped rows.

## Channel mapping

44 source channel columns from `RawData` are mapped to stable IDs under `SAFFIRE3::`. Source header text is retained. Units are populated only when the source header explicitly states a unit; otherwise the registry unit is null. Every channel is labelled `telemetry_channel`, but `is_instrument_sensor` is false because exact hardware identities and calibration are not confirmed by this package.

## Outputs

- `data/processed/saffire_iii_psi100/PSI100_RawData_packets.jsonl`
- `data/processed/saffire_iii_psi100/PSI100_RawData_normalized.csv`
- `data/processed/saffire_iii_psi100/channel_registry.json`
- `data/processed/saffire_iii_psi100/build_summary.json`
- `metadata/PSI100_channel_registry.csv`
- `metadata/PSI100_workbook_sheet_inventory.csv`
- `metadata/PSI100_source_equivalence.json`

Rebuild normalized outputs with:

```bash
python src/build_psi100_saffire_iii_packets.py
```

## Still requiring official evidence or team action

- Confirm the physical hardware identities represented by the Radiometer, Anemometer, SmokeWire, igniter, and Cygnus channel labels.
- Confirm engineering units for channels whose source headers do not state a unit.
- Obtain official calibration/transfer functions before interpreting raw signal values as calibrated physical quantities.
- Confirm the interpretation of `RelativeSeconds` with the NASA source documentation.
- Frontend owner must wire the adapter into the actual dashboard and test replay/rendering.

Raw-file sizes and SHA-256 hashes are recorded in `metadata/PSI100_RAW_FILE_INVENTORY.csv`; all three files match the manifest checksums supplied with the Data Engineering handoff.


## NASA instrumentation context (external documentation)

NASA's Saffire hardware paper describes the diagnostic suite as including four calibrated radiometers, two flow anemometers, fan tachometer readings, and voltage/current histories from the smoke wire and igniters. This corroborates those *channel families* in the workbook, but it does not establish the exact model/serial identity or calibration transfer function for each column in this exported file.

- Ruff & Urban, *Operation and Development Status of the Spacecraft Fire Experiments (Saffire)*, NASA Glenn Research Center / International Conference on Environmental Systems (2017), especially the diagnostics and telemetry descriptions: https://ntrs.nasa.gov/api/citations/20170000230/downloads/20170000230.pdf?attachment=true

The workbook channel names are therefore retained verbatim. Do not infer unlisted units or calibrations from the paper's high-level diagnostics description.
