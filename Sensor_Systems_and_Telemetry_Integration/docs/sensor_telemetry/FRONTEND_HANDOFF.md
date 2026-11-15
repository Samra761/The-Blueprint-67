# Frontend handoff

The frontend owner should consume `frontend_adapter/telemetry_frontend_adapter.py` or the combined packet JSON directly.

```python
from frontend_adapter.telemetry_frontend_adapter import to_frontend_payload, write_frontend_json
import json

packet_data = json.load(open('output/all_sources/all_sources_wide_packets.json', encoding='utf-8'))
payload = to_frontend_payload(packet_data)
write_frontend_json(packet_data, 'output/all_sources/frontend_payload.json')
```

The adapter exposes packet count, timestamped/untimestamped counts, variable metadata, packet provenance and latest packet. Reading metadata includes `value_type` and `quality_flags`.

## UI requirements

- Show investigation, source file, experiment/PSI identifiers and quality flags.
- Label values as reported or derived using `value_type`/`measurement_class`.
- Display source date/test time as experiment context, not a sample timestamp.
- If `timestamp_utc` is null, say “Sample timestamp not supplied”; do not put the computer clock into the data field.
- Keep simulated/demo KPIs visually labelled as demo until replaced by imported source values.
- Do not show a continuous live line chart for experiment-level rows without actual sample timestamps.
- Preserve gravity-level labels and distinguish the 52 ground-reference records from microgravity observations.

The supplied Flame in Freefall dashboard currently includes illustrative cards/charts and random chart updates. This Sensor Systems & Telemetry Integration package does not alter that UI; it supplies the validated data interface for its owner.


## SAME-R and SAFFIRE-III JSONL feeds (v2.1.6)

The adapter now accepts JSONL channel packets as well as the original wide-packet JSON envelope.

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

Both outputs use `nasa.telemetry.frontend.v1`. Preserve `quality_flags`, `timestamp_source`, `source_epoch_s` / `source_epoch_ms`, and elapsed time. SAME-R `timestamp_utc` is derived from a validated Unix epoch source field; PSI-100 timestamps are derived from `CygnusTimestamp` epoch milliseconds when plausible. Invalid source timestamps remain null.

The Streamlit reference consumer now accepts `.json` and `.jsonl` uploads. The frontend lead still owns wiring this adapter into the actual Flame in Freefall dashboard.
