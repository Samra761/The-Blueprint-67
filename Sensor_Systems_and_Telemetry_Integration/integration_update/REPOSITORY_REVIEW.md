# Sensor Systems & Telemetry Integration integration review — all supplied repositories

## What each ZIP actually contributes

| Uploaded archive | Contents established by inspection | Sensor Systems & Telemetry Integration use |
|---|---|---|
| `Bebo-ZeroG-X-main.zip` | Streamlit dashboard (`app.py`), `physics_models.py`, RAG engine and three small dashboard CSVs (BASS extinction limits, SAFFIRE pressure/O2, SoFIE PMMA burn rates). | Treat as a dashboard/UI reference. Do not mistake its three dashboard CSVs for the full sensor archive. Keep its existing visual design; consume the established data/API contract. |
| `flame-in-freefall-main.zip` | Mature data-engineering repository: PSI investigation/file registries, 72-file acquisition manifest, 13 canonical Parquet schemas/tables, CSV/XLS/XLSX parsers, dedicated PSI-100 telemetry and PSI-101 SAME-R parsers, validation, provenance, DuckDB views and read-only FastAPI API. Its committed registry snapshot lists 24 combustion investigations, 16,354 archive files and 22 experimental tables. Raw full telemetry archives are outside Git; only small authentic parser fixtures are committed. | This is the upstream data-engineering source of truth. Reuse its canonical schema, accession IDs, channel maps and API. Do not build a competing database, registry or API. |
| `flamme-main.zip` | 22 downloaded PSI experimental-table CSVs, source catalogue/manifests, long-format cleaned table, and FLEX / CFI-G ML scripts and generated model artifacts. | Provides the actual original PSI summary/test tables included in this update's source-row packet export. These tables are not all time-series sensor telemetry. |
| `fluxfire-Orbital.Mind-main.zip` | Independent evidence workspace for PSI-25 BASS-II, PSI-98 SAFFIRE-I and PSI-69 FLEX; provenance, source references, field-matched comparisons and import staging. | Use its provenance discipline and source references as a reference. It is not a substitute for the complete telemetry ingestion layer. |
| `NASA_flame_in_freefall-main.zip` | Next.js frontend / exploration app, investigation catalogue, rank/compare/gap UI and demo answers. Its README describes 20 investigation records and a heuristic scoring system. | Frontend/reference only. Do not interpret its ranking score as a measured sensor value or validated risk model. |

## Existing Sensor Systems & Telemetry Integration baseline retained

The prior verified Sensor Systems & Telemetry Integration package is preserved in this updated ZIP. It already contains:
- 69 canonical research-variable entries across seven supplied processed tables;
- seven source profiles and CSV/JSON/XLSX/XLS importer;
- one wide packet per experiment/source row when no sample timestamp exists;
- Python Dict/DataFrame interface and JSON/CSV export;
- validation, provenance, frontend adapter and tests;
- 10,627 mapped observations and 1,519 wide packets;
- 12/12 local tests passed in the prior build (17/17 in 2.1.5).

Those 69 entries are research variables, not confirmed physical sensor channels. The prior build had 0 sample-timestamped packets; its 1,519 packets are explicitly untimed source-row packets.

## What this update adds

- All 22 original PSI experimental-table CSVs from `flamme-main.zip`, preserved byte-for-byte.
- A table inventory with accession, original filename, row counts, encoding, SHA-256 and PSI source link.
- A field inventory of every source header and literal unit hint.
- A JSONL wide packet for each non-empty source row. Original cell strings, missingness flags, source row number, accession, PSI link and file checksum are retained.
- A numeric-candidate CSV with explicit SI conversions only where the unit is unambiguous. Numeric candidates are not automatically promoted to sensor readings.
- An adapter contract for the ML lead's previously reviewed SAME-R ZIP, mapping its eight reported model inputs to the exact SAME-R HKP source columns used by the data-engineering repository.

## Do not double-count

The 22 PSI experimental tables overlap investigation families and source content already represented in the previous seven-table Sensor Systems & Telemetry Integration package and in the data-engineering repository. This update stores them as a separate source-table layer. Do not add its packet/observation counts to the prior 1,519 packets or 10,627 observations as if all were unique experiments.

## Data boundaries

1. Experiment settings, derived outcomes, instrument channels and metadata remain distinct.
2. The PSI-101 experimental table is a sample/test log; the 98-column SAME-R HKP CSVs are the actual time-series channel data.
3. The PSI-100 experimental table is a sample summary; 1 Hz SAFFIRE-III telemetry is in separate telemetry exports.
4. PSI-68 FLEX-2 is a condition-range table, not individual measurements.
5. PSI-47 DAFT's experimental table is a session/instrument log, not a time-series combustion dataset.
6. PSI-142 is an index to figures/publications, not a measurement table.
7. Source dates, test-day numbers, GMT labels and elapsed settings are not promoted to UTC timestamps without documented semantics.
8. No live telemetry, physical sensor IDs, calibration constants or sample rates are invented.

## Open data-quality items to carry forward

### Resolved in 2.1.5

- **SAFFIRE sample thickness — corrected.** The processed tables stored Saffire-II samples 2-4, 2-5 and 2-6 as 37 mm (NASA PSI-99 prints `037 mm`). NASA's own results paper (Ferkul et al., ICES-2017, NTRS 20170008805) gives 0.37 mm for all three, so the processed value is now 0.37 mm. The same check showed the Saffire-I (3.7 mm) and Saffire-III (3.3 mm) SIBAL fabric values were also wrong by 10×: the NASA tables put a millimetre-scale number under a `cm` label, while every publication gives SIBAL fabric as about 0.32–0.37 mm. Those are now 0.37 mm and 0.33 mm. Every change, its evidence and its confidence level is listed in `config/source_corrections.json` and applied by `src/apply_source_corrections.py`. Corrected rows carry a `thickness_corrected` or `thickness_unit_corrected` quality flag. The original NASA tables are unchanged.
- **BASS oxygen scale — verified, no change needed.** All BASS (PSI-26) and BASS-II (PSI-25) oxygen values in the processed tables and observations are on the percentage scale (15.9–22.7 vol% and 16.8–20.9 vol%), labelled `vol%`, matching the NASA header `% by vol`. FLEX keeps a separate `x_O2` field that is explicitly a fraction. A test now fails if any `vol%` oxygen value falls outside 1–100.
- **SAME-R DustTrak — checked, partly resolved.** `ml_lead_handoff/SAME_R_DUSTTRAK_CHECK.json` shows that DustTrak B is internally consistent: its reading tracks its own analog output (r = 0.95) at a steady ~20 mg/m³ per volt and reads 0 when powered off. Its analog output clips at 5 V (about 100 mg/m³), and −10 V rows are faults. DustTrak A readings are coded values that do not track its analog output and are not usable as mass concentrations; it is not an ML input. Smoke-specific calibration is still not established here, so `dtb` stays a relative mass-monitor reading until the NIST/NASA factors from the SAME publications are applied.
- **SAME-R channel identities — documented from source.** `ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv` lists all 98 SAME-R housekeeping columns exactly as named in the NASA files, with units from the headers, the 24 bracketed hardware tags (for example `[AT08]`, `[PN23]`) copied verbatim, value ranges and the 8 ML-input mappings. The tags are not interpreted. The −10 V values seen in raw ISS/STS/ion channels occur only in idle rows (test number 0), which the packets already exclude.

### Still open

- The supplied team update states FLEX burning-rate unit is confirmed as mm²/s; this update applies that correction only to PSI-69's burning-rate field and retains the source header verbatim.
- Saffire-II sample 2-6 air flow: the NASA table says 20 cm/s, Ferkul et al. ICES-2017 says 25 cm/s and NTRS 20170001636 says 30 cm/s. The sources disagree, so the NASA table value is kept.
- `integration_update/PSI_NUMERIC_CANDIDATES.csv` remains a literal parse of the original NASA tables (it still shows `037 mm` as 37 mm). Use the corrected processed tables for analysis.
- Smoke-specific DustTrak calibration factors (see above).
- Meaning of the bracketed SAME-R hardware tags requires NASA instrumentation documentation.
