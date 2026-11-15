# Changelog

## 2.1.5

### Fixed
- **SAFFIRE sample thickness (7 samples).** Processed values were too large by 10× or 100×:
  - Saffire-II 2-4, 2-5, 2-6: 37 mm → **0.37 mm**. The NASA PSI-99 table prints `037 mm`. Confirmed by Ferkul et al., ICES-2017 (NTRS 20170008805).
  - Saffire-I S1, S2: 3.7 mm → **0.37 mm**. The PSI-98 table puts 0.37 under a `cm` header, but published SIBAL fabric thickness is 0.32–0.37 mm.
  - Saffire-III S1, S2: 3.3 mm → **0.33 mm**. The PSI-100 table prints `.33 cm`, but NTRS 20170001636 gives SIBAL as 0.33 mm.
  - All evidence is in `config/source_corrections.json`. Corrections are applied by `src/apply_source_corrections.py`, which is idempotent and has a `--check` mode. Each corrected value carries a quality flag. Original NASA tables are unchanged.
- **Non-reproducible builds.** Field order in `source_row_metadata` depended on Python's set ordering, so rebuilds differed byte-for-byte. It now follows source column order, and rebuilds are byte-identical.

### Added
- `ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv`: all 98 SAME-R housekeeping channels as named in the NASA files (units, 24 verbatim hardware tags, ranges, ML-input mapping).
- `ml_lead_handoff/SAME_R_DUSTTRAK_CHECK.json`: DustTrak A/B consistency check.
- `src/build_same_r_channel_inventory.py`, `src/apply_source_corrections.py`, `config/source_corrections.json`.
- `src/run_all_sources.py` now also regenerates `all_sources_wide_packets.csv` and `frontend_payload.json`.
- Tests: corrections fully applied, SAFFIRE fabric/silicone thickness under 2 mm, `vol%` oxygen on percentage scale, channel inventory matches the ML adapter, deterministic metadata order (17 base tests, up from 12).

### Verified (no change needed)
- BASS and BASS-II oxygen values are percentage-scale (`vol%`) throughout, matching the NASA headers.
- Raw −10 V values in SAME-R detector channels occur only in idle rows (test number 0), which the packets already exclude.

### Unchanged
- 10,627 observations, 1,519 base packets, 2,009 PSI source-row packets, 58,386 SAME-R packets.


## 2.1.6
- Added PSI-100 SAFFIRE-III raw exports and normalized RawData telemetry packets (44 channels).
- Added workbook/cleaned-CSV equivalence validation and worksheet overlap inventory.
- Validated and promoted SAME-R Unix epoch seconds to ISO-8601 UTC across 58,386 packets with zero GMT-label mismatches.
- Extended frontend adapter for channel packets and JSONL input; expanded tests and Streamlit example.
- Preserved unresolved hardware identity, unit and calibration limitations explicitly.
