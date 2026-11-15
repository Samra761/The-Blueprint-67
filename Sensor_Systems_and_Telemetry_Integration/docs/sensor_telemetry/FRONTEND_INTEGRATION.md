# Frontend / ML / Sensor Systems & Telemetry Integration integration contract

Status checked against: the ML lead's final zip (`ML Flame in freefall`) and the frontend lead's `app.py` (Streamlit, single embedded HTML page) with its CSV bundle and engineering report.

## 1. How the three pieces connect

| Link | Status |
|---|---|
| ML ↔ Frontend | Integrated. The frontend ran the ML lead's saved models/feature columns and replays the outputs. |
| ML ↔ Sensor Systems & Telemetry Integration | Compatible. The 43 raw SAME-R CSVs and the 58,386 packets match the ML lead's data row for row; `feature_columns.json` is identical. |
| Sensor Systems & Telemetry Integration ↔ Frontend | Adapter now supports both SAME-R channel packets and PSI-100 SAFFIRE-III packets. The frontend owner still needs to wire the adapter into the actual dashboard; the uploaded dashboard has not been changed here. |

## 2. Shared channel contract (SAME-R, 8 channels)

| ML feature / packet key | Frontend key | Source CSV column | Unit | Status |
|---|---|---|---|---|
| `iss_obs` | `io` | `ISSDetectorObscurationMeasurement(Volts)` | V | uncalibrated voltage |
| `iss_sct` | `is` | `ISSDetectorScatterMeasurement(Volts)` | V | uncalibrated voltage |
| `sts` | `st` | `STSDetectorMeasurement(Volts)` | V | uncalibrated voltage |
| `ion_a` | `ia` | `IonDetectorAMeasurement(Volts)` | V | uncalibrated voltage |
| `ion_b` | `ib` | `IonDetectorBMeasurement(Volts)` | V | uncalibrated voltage |
| `ptrak` | `pk` | `PTrakMeasurement(PtPerCC)` | pt/cc | `-999` = sentinel → null + flag |
| `dtb` | `db` | `DustTrakBMeasurement(MgPerM3)` | mg/m³ | validity/calibration not documented |
| `plume_t` | `pt` | `GasPlumeTemperature(DegC)` | °C | see §5 |

Single source of truth for the mapping: `ml_lead_handoff/ML_LEAD_SAME_R_ADAPTER.json`.

## 3. Packet fields (`ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl`)

`test_id`, `source_epoch_s`, `elapsed_test_time_s`, `timestamp_utc` (null), `time_string`, `material`, `candidate_materials`, `label_verified`, `heating_in_progress`, `source_file`, `channels{8 keys}`, `quality_flags[]`.

- One packet per (test, second). 58,386 packets, 41 tests.
- Packets are mapped raw values, not model features. Baseline correction, rolling windows and ratios stay with the ML lead's code.

## 4. Timestamp rule — SAME-R promotion validated

- `source_epoch_s` is the CSV `Time` column (Unix epoch seconds). It equals the GMT `time_string` in all 58,386 packets.
- All 58,386 pairs were checked; there were zero mismatches. `timestamp_utc` is now populated as ISO-8601 UTC from `source_epoch_s`; the original epoch field and `time_string` remain preserved.
- `elapsed_test_time_s` is `ElapsedTestTime(Secs)` (seconds since test start) and remains a separate elapsed-time field.
- Conversion provenance is recorded in each packet (`timestamp_source`, `time_basis`) and in `ml_lead_handoff/ML_LEAD_SAME_R_ADAPTER.json`.

## 5. Claims the UI must caveat

- **Early-warning lead time.** The dashboard shows "AI model +3 s, ISS alarm ~+128 s, Lead ~125 s". In all 41 tests the trigger is `plume_t`. The ML lead's own report shows that without plume temperature the median lead over the ISS alarm is about 1 s. Show the no-plume result alongside, and note that plume temperature may react to the heater itself (not confirmed).
- **Classifier early identification.** Test-level accuracy is about 37% at 120 s and about 70% at 450–600 s; 27/38 tests correct overall (leave-one-test-out).
- **DustTrak.** DustTrak B is internally consistent (`ml_lead_handoff/SAME_R_DUSTTRAK_CHECK.json`) but not calibrated for smoke; present `dtb` as a relative mass-monitor reading, not a calibrated concentration. Analog values at 5 V are saturated.
- **Unlabelled / unverified tests.** Duplicate-file tests have `label_verified: false`; they must not be shown as confirmed materials.

## 6. Dataset reconciliation

| Frontend file | Sensor Systems & Telemetry Integration raw table | Result |
|---|---|---|
| `PSI-98_Experimental table_SAFFIRE-1.csv` | `integration_update/data/raw_psi_experimental_tables/` (PSI-98) | Byte-identical |
| `PSI-69_Experimental table_FLEX.csv` (274 rows) | PSI-69 | Byte-identical |
| `PSI-25_Experimental table_BASS-II.csv` (129 rows) | PSI-25 | Byte-identical. Sensor Systems & Telemetry Integration's processed BASS-II table has 52 rows; the two counts are different layers, do not compare them. |
| `ACME_CIR_Extinction_Data.csv`, `FLEX2_Cool_Flame_Droplet_Data.csv` | PSI-20/22/23 (ACME), PSI-68 (FLEX-2) | **Cannot be verified.** The real tables hold test dates, burner sizes, fuels and condition ranges, not per-test burn times or extinction diameters. FLEX2 file cites PSI-70; the Sensor Systems & Telemetry Integration table is PSI-68. Keep the UNVERIFIED badge, or remove them from the demo. |
| `JAXA_FLARE_Kibo_Flight_Data.csv`, `SOFIE_Flight_Data.csv`, `NASA_*`, `SAFFIRE_II_to_VI_*`, other bundle CSVs | not in Sensor Systems & Telemetry Integration | Not checked. |

## 7. Replaying from Sensor Systems & Telemetry Integration packets (optional upgrade)

1. Load packets, group by `test_id`, sort by `source_epoch_s`.
2. Heater-on time = first packet with `heating_in_progress = 1`; relative time = `source_epoch_s` minus that.
3. Baseline = mean of the 60 s before heater-on (the frontend's current rule).
4. Treat `ptrak: null` as missing (do not plot as 0).
5. The frontend's 2 s resample and 0–999 quantisation are display-only; keep raw values for tooltips and exports.

## 8. Open items

- [ ] Team decision on `timestamp_utc` (§4).
- [ ] Add the no-plume lead-time caveat to the Early warning card (§5).
- [ ] Decide whether ACME-CIR / FLEX2 stay in the demo (§6).
- [ ] Optionally switch the replay to Sensor Systems & Telemetry Integration packets (§7).
- [ ] Pin the scikit-learn version (frontend report: RF 71.1% vs ML report 73.7%).


## 9. PSI-100 SAFFIRE-III telemetry integration (v2.1.6)

- Three original PSI-100 exports are preserved under `data/raw/psi100/`.
- The workbook `RawData` sheet is the canonical combined telemetry stream: 11,450 packets, 44 channels, 11,449 source-derived UTC timestamps, and one invalid terminal timestamp retained with a quality flag.
- `RawData` matches the detailed cleaned CSV on all 47 mapped columns and all 11,450 rows; the CSV is a corroborating export, not a second independent packet stream.
- The workbook's `CygnusData`, `SaffireData`, and split detector/pressure/temperature sheets are projections/views. Do not add their rows to the canonical packet count.
- Units are populated only when explicitly present in source headers. Header labels do not establish verified hardware identity or calibration.
- `RelativeSeconds` is preserved verbatim because its scale conflicts with the workbook's explicit millisecond relative-time fields; no silent unit correction is applied.
- One terminal row has an invalid extreme `CygnusTimestamp`; it is retained, has `timestamp_utc: null`, and carries `invalid_cygnus_timestamp_source_value`.
- Rebuild with `python src/build_psi100_saffire_iii_packets.py`.
