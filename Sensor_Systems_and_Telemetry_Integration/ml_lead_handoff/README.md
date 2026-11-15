# ML lead SAME-R handoff layer (Sensor Systems & Telemetry Integration integration)

This folder adds the ML lead’s original SAME-R HKP CSV inputs and a Sensor Systems & Telemetry Integration packetized handoff. Raw CSVs are preserved byte-for-byte. The ML lead retains ownership of preprocessing, models and evaluation.

## Files
- `raw_same_r_hkp/`: all 43 original CSVs by material folder, byte-for-byte. The ML loader uses 40 (it drops 2 duplicate copies and 1 test where the heater never fired; see the inventory).
- `SAME_R_WIDE_PACKETS.jsonl`: one packet per TestNumber + Time sample. Time fields: `source_epoch_s` (CSV `Time`, Unix epoch seconds), `time_string` (GMT label) and `elapsed_test_time_s` (CSV `ElapsedTestTime(Secs)`). `timestamp_utc` stays null pending team approval to promote `source_epoch_s`.
- `ML_LEAD_SAME_R_ADAPTER.json`: canonical channel map and rules.
- `SAME_R_RAW_FILE_INVENTORY.csv`: source file sizes and SHA-256.
- `feature_columns.json`: ML lead’s saved feature-column contract.
- `ML_LEAD_FEATURE_PREPROCESSING_REFERENCE.py.txt`: ML lead’s preprocessing source for integration reference.
- `SAME_R_CHANNEL_INVENTORY.csv`: all 98 HKP columns as named in the NASA files, with header units, verbatim bracketed hardware tags, value ranges and ML-input mapping (built by `src/build_same_r_channel_inventory.py`). Tags are not interpreted.
- `SAME_R_DUSTTRAK_CHECK.json`: DustTrak A/B internal-consistency check. DustTrak B is consistent (~20 mg/m³ per volt, r = 0.95); DustTrak A readings are not usable concentrations. Not a calibration.

**Important:** These packets contain mapped raw channel values, not the engineered feature vectors expected by the trained models. The ML lead’s baseline correction, rolling-window features and classifier ratios must be applied exactly before inference. The CSV `Time` column is Unix epoch seconds and agrees with the GMT `TimeString` in every packet; the seconds-since-test-start column is `ElapsedTestTime(Secs)`. P-Trak `-999` is a sentinel: the original CSV remains unchanged; normalized `ptrak` is null with a quality flag.
