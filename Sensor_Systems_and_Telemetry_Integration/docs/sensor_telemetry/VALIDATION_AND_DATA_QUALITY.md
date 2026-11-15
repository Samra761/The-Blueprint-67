# Validation and data-quality rules

The importer and validator enforce:

- Required experiment, variable, unit and source identifiers.
- Canonical variable ID must exist in the registry.
- Measurement class must be one of `reported_experimental_variable`, `experimental_setting`, or `derived_result`.
- Values must parse as finite numeric values; NaN and infinity are rejected.
- A supplied `timestamp_utc` must parse as a timestamp.
- Duplicate observation IDs are rejected.
- Unregistered numeric columns are not silently added as variables; they remain outside the canonical measurement output.
- Missing source values are omitted, not replaced with zero.
- `quality_flags` and `value_type` are preserved as per-reading metadata.

## Timestamp semantics

A source may contain a test date, GMT label, elapsed time or test-point index. These are not interchangeable. Only a field documented as an individual sample timestamp can populate `timestamp_utc`. For the supplied processed tables, the build produces untimed source-row packets. The dashboard may show experiment/test dates separately as contextual metadata.

## Limitations

The current validator does not certify physical sensor calibration, instrument identity, sampling rate, physical plausibility bounds or unit correctness beyond the registry mapping. Those require official instrumentation/data dictionary evidence. It also does not resample irregular time series or synthesize missing readings.
