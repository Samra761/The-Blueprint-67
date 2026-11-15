# Sensor Systems & Telemetry Integration status — package release 2.1.5

## 2.1.5 changes
- SAFFIRE sample thickness corrected for 7 samples (Saffire-II 2-4/2-5/2-6: 37 → 0.37 mm; Saffire-I: 3.7 → 0.37 mm; Saffire-III: 3.3 → 0.33 mm), with evidence in `config/source_corrections.json`.
- Rebuilds are now byte-for-byte reproducible (field order no longer depends on Python's set ordering).
- BASS/BASS-II oxygen confirmed to be percentage-scale everywhere; no change needed.
- New `ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv` (98 source channels) and `SAME_R_DUSTTRAK_CHECK.json`.
- Tests: 17 base + 3 supplemental, all passing.
- Record and packet counts are unchanged.

## Completed deliverables retained
The original Sensor Systems & Telemetry Integration baseline is retained in this release and remains executable:
- 69-variable research registry, seven dataset profiles, generic importer, validation, packet builder, exports and frontend adapter.
- 10,627 mapped observations; 1,519 packets; 7/7 source validation reports passed; 12/12 tests passed in its previous local build (17/17 in 2.1.5).
- Packet rule: one packet per source row when there is no sample timestamp. No timestamp is fabricated.

## New source-table layer in this update
- PSI experimental-table CSVs preserved: 22
- Non-empty source-row packets: 2009
- Source field definitions inventoried: 231
- Numeric scalar candidates recorded: 16042
- Numeric candidates are not all sensors; many are settings, identifiers, outcomes, or derived values.
- Blank cells remain missing; explicit `none`/`n/a` remain not applicable; `OR`, `-999`, `?`, and spreadsheet error tokens remain flagged.
- SI values are added only when a source unit is explicit or the supplied PSI-69 correction is specifically applied. Original value and unit are retained.

## How to use it
1. Keep `flame-in-freefall-main` as the team's canonical ingestion/API owner.
2. Use the prior Sensor Systems & Telemetry Integration package's adapter for frontend and model handoff.
3. Use `integration_update/PSI_SOURCE_ROW_PACKETS.jsonl` for the 22 summary/test tables when a source-row view is needed.
4. For actual time-series smoke data, use the SAME-R HKP raw files and the ML lead adapter contract, not the PSI-101 109-row experimental test log.
5. For SAFFIRE-III time-series data, use the separate PSI-100 telemetry exports parsed by the data-engineering repo, not the 2-row experimental table.
6. Coordinate the canonical variable IDs with the data engineer; do not fork the registry.

## ML lead handoff
The ML lead's SAME-R handoff contains 43 original CSVs, 41 tests and 58,386 one-second records across five material labels. The adapter JSON maps the eight reported model inputs to SAME-R's raw column names. All 43 original ML-lead HKP CSVs are now included under `ml_lead_handoff/raw_same_r_hkp/`, with a checksummed inventory and 58,386 normalized wide packets. The ML lead's feature-column contract and preprocessing source are included as references. The normalized raw packets are not yet the engineered feature matrix consumed by trained models; the ML lead's exact preprocessing must run before inference.

## Ownership
- Sensor Systems & Telemetry Integration: ingest, map, validate, preserve provenance, emit packets and model/frontend-ready data.
- Data engineer: canonical upstream database/API, acquisition, original-paper resolution and cross-dataset comparison rules.
- ML lead: owns training/inference and model evaluation; Sensor Systems & Telemetry Integration supplies data.
- Frontend/dashboard owner: consumes the adapter/API and preserves the chosen visual aesthetic.

## Reproducible test commands
Run from the package root:
- Base suites: `python -m pytest -q tests frontend_adapter/test_adapter.py`
- Supplemental importer suite: `PYTHONPATH=integration_update/src python -m pytest -q integration_update/tests`

The ML SAME-R raw handoff is included in this archive: all 43 original CSV files are preserved, 40 are used by the ML lead loader rules, and 58,386 normalized raw-channel packets are provided. The adapter is a data contract; trained-model inference and end-to-end model execution have not been run or claimed.


Package version note: 2.1.5 changes processed values only where `config/source_corrections.json` documents a correction. Original NASA source files are unchanged.
