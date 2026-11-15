# SAME-R channel and timestamp integration update

## Channels

The ML lead's 8 model input channels remain mapped as documented in `ml_lead_handoff/ML_LEAD_SAME_R_ADAPTER.json`:

| Packet key | Source CSV header | Source unit label |
|---|---|---|
| `iss_obs` | `ISSDetectorObscurationMeasurement(Volts)` | V |
| `iss_sct` | `ISSDetectorScatterMeasurement(Volts)` | V |
| `sts` | `STSDetectorMeasurement(Volts)` | V |
| `ion_a` | `IonDetectorAMeasurement(Volts)` | V |
| `ion_b` | `IonDetectorBMeasurement(Volts)` | V |
| `ptrak` | `PTrakMeasurement(PtPerCC)` | pt/cc |
| `dtb` | `DustTrakBMeasurement(MgPerM3)` | mg/m³ source label; not smoke-calibrated |
| `plume_t` | `GasPlumeTemperature(DegC)` | °C |

The packet channels are raw telemetry, not the ML lead's engineered feature matrix. ML preprocessing and model ownership remain with the ML lead.

## Timestamp validation

`source_epoch_s` is the source CSV `Time` field in Unix epoch seconds. All 58,386 values were converted to UTC and checked against the source GMT `time_string` in `YYYY/DOY/HH:MM:SS` format. There were zero mismatches.

The packet now includes:
- `timestamp_utc`: ISO-8601 UTC derived from `source_epoch_s`.
- `timestamp_source`: `source_epoch_s_unix_seconds`.
- `time_basis`: `source_epoch_s_unix_utc`.
- `source_epoch_s`: unchanged original epoch value.
- `elapsed_test_time_s`: unchanged seconds since test start.
- `time_string`: unchanged source GMT label.

The conversion is reproducible using `python src/promote_same_r_source_epoch_utc.py`. The script checks all source GMT labels before replacing the packet file.

## Limitations retained

- P-Trak `-999` remains null in normalized packets with a quality flag; raw source CSVs remain unchanged.
- DustTrak B is internally consistent but does not have a smoke-specific calibration in the supplied evidence. Treat it as a relative reading, not calibrated smoke mass concentration.
- Bracketed hardware tags in the 98-column inventory remain verbatim; their physical meaning is not inferred.
- The plume-temperature channel triggers all 41 early-warning detections in the reviewed ML result. The no-plume median lead is about 1 second; display this caveat alongside the headline early-warning result.


## Instrument interpretation evidence

NASA PSI-101 describes SAME-R's diagnostics and notes that M-pass replaced one of the commercial diagnostic enclosures that included a DustTrak and ionization detector. NASA/NIST publications describe P-Trak as a condensation particle counter and DustTrak as a light-scattering mass monitor; DustTrak responses are material-dependent and require material-specific calibration/correction. The supplied package does not include a validated SAME-R DustTrak B smoke calibration factor, so `dtb` remains a relative/un-calibrated channel.

- NASA PSI-101 (SAME-R investigation): https://psi.nasa.gov/physci/repo/data/investigations/PSI-101
- Yang et al., *Evaluation of Spacecraft Smoke Detector Performance in the Low-Gravity Environment* (2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC7918997/
- NASA DAFT documentation for P-Trak: https://www.nasa.gov/wp-content/uploads/2025/07/dust-and-aerosol-measurement-feasibility-test-daft.pdf
