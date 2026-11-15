# Sensor Lead Handoff — v2.1.6

## What is ready

### SAME-R sensor stream
- 43 original HKP CSVs remain preserved byte-for-byte.
- 58,386 normalized SAME-R packets across 41 tests remain available in `ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl`.
- All 58,386 `source_epoch_s` values were validated against the NASA GMT `time_string` field (zero mismatches).
- `timestamp_utc` is now ISO-8601 UTC derived from `source_epoch_s`; original epoch and elapsed time remain preserved.
- The P-Trak `-999` sentinel remains null and quality-flagged. The frontend adapter now attaches that channel-specific flag only to `ptrak`.

### PSI-100 SAFFIRE-III
- All three original exports are included unchanged under `data/raw/psi100/`.
- Workbook `RawData` is normalized into 11,450 packets and 44 channel variables.
- 11,449 packets have valid source-derived UTC timestamps; one invalid terminal timestamp is preserved and flagged.
- The workbook RawData sheet and detailed cleaned CSV match across 47 mapped columns and all 11,450 rows. They are not counted twice.
- Other workbook sheets are inventoried as projections/views, not additional independent datasets.

### Frontend adapter
- Supports existing wide-packet JSON and channel-packet JSONL.
- Supports the SAME-R and PSI-100 packet structures.
- Preserves nulls, source time, elapsed time, source row, material-label verification, and quality flags.
- Streamlit reference consumer accepts JSON and JSONL uploads.

## Validation
Run from this directory:

```bash
PYTHONPATH=integration_update/src python -m pytest -q tests frontend_adapter/test_adapter.py integration_update/tests
```

Result in this build: **23 passed**.

Rebuild PSI-100 normalized packets:

```bash
python src/build_psi100_saffire_iii_packets.py
```

Revalidate/promote SAME-R timestamps (the script refuses to write if any GMT label mismatches):

```bash
python src/promote_same_r_source_epoch_utc.py
```

## Frontend integration examples

```python
from frontend_adapter.telemetry_frontend_adapter import load_and_adapt_jsonl

same_r = load_and_adapt_jsonl(
    "ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl"
)

saffire_iii = load_and_adapt_jsonl(
    "data/processed/saffire_iii_psi100/PSI100_RawData_packets.jsonl",
    "data/processed/saffire_iii_psi100/channel_registry.json"
)
```

The adapter returns `nasa.telemetry.frontend.v1`. Frontend owner should render `payload['packets']`, use `payload['latest']` only when timestamped, and keep source quality flags visible. This package does not alter the actual Flame in Freefall dashboard; wiring it into the UI and end-to-end UI tests remain frontend-owned.

## Items still requiring Sensor Lead / official source confirmation
1. Confirm physical hardware identity for NASA header labels, including Radiometer/Anemometer/SmokeWire/igniter and Cygnus channels.
2. Confirm units for channels whose source headers do not state units.
3. Obtain official DustTrak B smoke-specific calibration/transfer function before showing calibrated mass concentration. Until then it is a relative reading.
4. Confirm the PSI-100 `RelativeSeconds` scale with NASA documentation. It is preserved as supplied; no conversion was applied.
5. Confirm whether any hardware tags in `SAME_R_CHANNEL_INVENTORY.csv` have documented instrument meanings. They remain verbatim and uninterpreted.
6. Coordinate canonical API path and schema version with Data Engineering.
7. Ask Frontend Lead to connect the adapter to the actual dashboard and test historical replay. Do not call historical files live telemetry.

## Ownership boundaries
- Sensor Integration: ingestion, mapping, validation, packetization, provenance and adapter.
- ML Lead: feature engineering, model training, inference and evaluation.
- Frontend Lead: dashboard UI, replay controls and display.
- Data Engineering: canonical source repository/API and source acquisition.
