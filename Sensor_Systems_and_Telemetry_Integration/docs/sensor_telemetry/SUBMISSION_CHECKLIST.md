# Sensor Systems & Telemetry Integration submission checklist

- [x] Canonical variable registry (69 variables)
- [x] Seven dataset profiles and source tables
- [x] CSV/JSON/Excel input reader
- [x] Source-column to canonical variable mapping
- [x] Required-field, numeric, duplicate-ID and timestamp validation
- [x] Preserve value type, quality flags, source references and row context
- [x] Wide packet generation (timestamp grouping; source-row fallback)
- [x] Dict + DataFrame function interface
- [x] JSON + CSV export
- [x] All-seven-source rebuild command
- [x] Frontend adapter and handoff contract
- [x] Unit tests
- [x] Dataset catalog, schema and limitations documentation
- [x] SAME-R channel names, units and hardware tags documented verbatim from the NASA source files (`ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv`)
- [x] SAFFIRE thickness corrected with evidence; BASS oxygen scale verified; DustTrak consistency checked (2.1.5)
- [x] Deterministic, byte-reproducible rebuild
- [ ] Meaning of the SAME-R bracketed hardware tags, and DustTrak smoke calibration factors, confirmed from NASA/NIST instrumentation documentation
- [ ] Agree final canonical IDs and API path with Data Engineer/backend owner
- [ ] Frontend owner plugs the output into the actual dashboard

The open items require team coordination or official evidence; they are not silently assumed complete.


## v2.1.6 additions
- [x] PSI-100 SAFFIRE-III source exports included unchanged
- [x] PSI-100 RawData worksheet normalized to 11,450 packets and 44 source channels
- [x] Workbook RawData ↔ detailed cleaned CSV equivalence checked across 47 mapped columns
- [x] SAME-R epoch-to-UTC conversion validated against all 58,386 GMT labels
- [x] Frontend adapter supports SAME-R / SAFFIRE-III channel packets and JSONL input
- [x] Adapter tests cover channel packets, source metadata and JSONL loading
- [ ] Physical hardware identities and unlabelled units confirmed from official NASA documentation
- [ ] DustTrak smoke-specific calibration confirmed
- [ ] Frontend lead wires the adapter into the actual dashboard and verifies replay/rendering
