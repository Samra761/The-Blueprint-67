# Telemetry schema — wide packet v1.1

> The packet schema version (1.1) is independent of the package version (2.1.5).

Top-level envelope:

```json
{
  "schema_version": "1.1",
  "format": "wide_packet_per_timestamp",
  "source_count": 7,
  "packet_count": 1519,
  "grouping_policy": "...",
  "timestamp_policy": "...",
  "data_semantics": "...",
  "packets": []
}
```

Each packet includes:

- `packet_id`
- `experiment_id`, `psi_id`, `run_id`
- `timestamp_utc` (ISO-8601 only when supplied as a sample timestamp; otherwise null)
- `time_basis` (`source_timestamp` or `no_timestamp_source_row`)
- `source_file`, `source_reference`, `record_ids`
- `source_row_metadata` (record/test/date/gravity/material/source labels when present)
- `readings`: map from canonical variable ID to numeric value
- `units`, `variable_names`, `measurement_classes`
- `reading_metadata`: per-variable `value_type`, `quality_flags`, source row/reference and `is_instrument_sensor: false`

Flattened CSV has one row per packet, packet metadata columns, and one column per canonical variable ID. Empty values remain blank/null.

## Grouping

If a genuine `timestamp_utc` field exists, records group by source file + experiment + PSI + run + timestamp. If absent, each original source row becomes its own packet. A test date or elapsed time is not automatically a timestamp.
