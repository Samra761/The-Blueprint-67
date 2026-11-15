# Supplemental PSI Source-Table Layer

This directory contains the supplemental PSI experimental-table intake and source-row packet layer. It complements the base processed-source pipeline; it does not replace the canonical data-engineering repository or database.

## Contents

- `data/raw_psi_experimental_tables/` — 22 original PSI CSV tables retained as supplied.
- `PSI_SOURCE_TABLE_INVENTORY.csv` — source file inventory, row counts, hashes, and source references.
- `PSI_FIELD_INVENTORY.csv` — 231 source fields with literal unit hints where available.
- `PSI_SOURCE_ROW_PACKETS.jsonl` — 2,009 source-row packets; timestamps remain null where no sample timestamp exists.
- `PSI_NUMERIC_CANDIDATES.csv` — 16,042 numeric scalar candidates for review; these include settings, identifiers, outcomes, and derived values and are not all sensor measurements.
- `src/psi_source_table_importer.py` — supplemental importer.
- `tests/` — importer tests.
- `ML_LEAD_SAME_R_ADAPTER.json` — reference to the SAME-R channel mapping contract.

## Test command

From the module root:

```bash
PYTHONPATH=integration_update/src python -m pytest -q integration_update/tests
```

## Interpretation cautions

Source tables are not all time-series telemetry. Some are test logs, sample summaries, indexes, or condition-range tables. Numeric extraction does not establish that a field is a physical sensor channel. Preserve source names, units, provenance, and unresolved semantics. Do not infer sample timestamps, calibration, or sensor IDs. Source layers may overlap; do not sum their row/packet counts as unique observations.
