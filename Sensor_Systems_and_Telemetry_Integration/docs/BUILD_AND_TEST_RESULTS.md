# Wide packet build and test results

- Input observations: 10627
- Wide packets: 1519
- Wide DataFrame shape: 1519 rows × 78 columns
- Timestamped packets: 0
- Untimestamped source-row packets: 1519

All supplied source observations in this build lack timestamp values. Consequently, timestamped packet count is zero; each packet retains `timestamp_utc: null` and groups the variables from one source row. This respects the requirement without fabricating time.

The function `build_wide_packets` returns `(Dict, DataFrame)`. `export_packets_json` dumps the Dict envelope as JSON. The CLI writes JSON and a wide CSV automatically.

## 2.1.5 build and test results

- `python src/apply_source_corrections.py --check`: 70 corrected cells, 0 pending.
- `python src/run_all_sources.py`: 7 sources, 10,627 observations, 1,519 packets, 0 import errors, all validations passed. It now also writes `all_sources_wide_packets.csv` and `frontend_payload.json`.
- Reproducibility: two rebuilds with different `PYTHONHASHSEED` values produce byte-identical outputs.
- `python -m pytest -q tests frontend_adapter/test_adapter.py`: 17 passed.
- `PYTHONPATH=integration_update/src python -m pytest -q integration_update/tests`: 3 passed.
- `python src/build_same_r_channel_inventory.py`: 98 channels, 24 with bracketed hardware tags, 8 ML inputs; DustTrak B reading vs analog r = 0.954 at ~20.0 mg/m³ per volt.


## 2.1.6 sensor integration update

- PSI-100 SAFFIRE-III: 3 original source exports included unchanged.
- Canonical workbook `RawData`: 11,450 packets, 44 source channels; 11,449 valid source-derived UTC timestamps; 1 invalid terminal timestamp retained and flagged.
- Workbook `RawData` matches the detailed cleaned CSV on all 47 mapped columns and all 11,450 rows (numeric tolerance 1e-7). They are not double-counted.
- SAME-R: all 58,386 `source_epoch_s` values match the source GMT label and are promoted to ISO-8601 UTC; 0 mismatches.
- Frontend adapter now supports `channels` packets and JSONL loading, with source channel metadata and quality flags.
- Full local test command: `PYTHONPATH=integration_update/src python -m pytest -q tests frontend_adapter/test_adapter.py integration_update/tests` — **23 passed**.
- Full-data adapter smoke test: SAME-R 58,386/58,386 timestamped; PSI-100 11,450 packets, 11,449 timestamped, 1 untimestamped.
