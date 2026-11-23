# Task 4: Per-sensor sensitivity breakdown (SAME, 8 channels)

Script: `src/sensor_sensitivity.py` (run from the project root). Input: `Sensor_Systems_and_Telemetry_Integration/ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl` (41 tests, 58,386 packets).

```bash
python src/sensor_sensitivity.py                  # LOTO + 8-fold, detector + classifier (~1.5 h on 2 vCPU)
python src/sensor_sensitivity.py --split loto     # main result only
python src/sensor_sensitivity.py --no-classifier  # detector only (~12 min LOTO)
python src/sensor_sensitivity.py --reuse          # reuse saved results, only compute what is missing
python src/sensor_sensitivity.py --check          # verify against anomaly.fit_detector
```

## Method
- **Detector:** the per-sensor Isolation Forest from `src/anomaly.py`, used unchanged: trained on clean air only, threshold at q = 1.0 of inner out-of-fold clean-air scores, 3 s persistence. It is evaluated with one channel only, with one channel removed, with all 8, and on all 255 channel subsets.
- **t0 = heater on.** SAME heats samples to pyrolysis and there is no open-flame ignition, so latency is t_detect − heater-on.
- **Standard alarms:** ISS photoelectric scatter detector and ionisation detector A, calibrated to the same clean-air quantile, as in `anomaly.py`.
- **Event-level metrics:** each test contributes one fire event (TP if alerted after t0) and one clean-air period (FP if any false alarm). Accuracy = (TP+TN)/(2·41), F1 = 2TP/(2TP+FP+FN). False alarms are counted per hour of clean air.
- **Material classifier:** Random Forest with the parameters from `src/classifier.py`, on 38 labelled tests, using the test-level decision from 120–600 s windows. A feature is used only if every channel it needs is in the subset. LightGBM was not installed in the run environment.
- **Splits:** LOTO (41 folds, main) and 8-fold GroupKFold (check against the README).

## Reproducibility check
- The cached subset construction matches `anomaly.fit_detector` exactly (`--check`: max difference 0).
- On 8-fold GroupKFold, the full detector's alert time is identical to `reports/anomaly/per_test.csv` for all 41 tests (41/41 detected, median 3 s, P90 5 s).
- One extra clean-air false alarm (test 64) appears with scikit-learn 1.9, giving 0.74 vs 0.37 alarms/h.
- The RF classifier gets 27/38 here vs 28/38 in the README (the one-test difference is attributed to the scikit-learn version).

## Main table (LOTO): see `sensor_table_loto.md` / `.csv`
Each row is one channel on its own.

| Sensor | Detected | Det. F1 | FA/h | Median delay (s) | P90 (s) | Beats ISS alarm | Material F1 alone |
|---|---|---|---|---|---|---|---|
| Gas plume temp. | 41/41 | 0.99 | 0.37 | 3 | 5 | 41/41 | 0.59 |
| ISS obscuration | 41/41 | 0.99 | 0.37 | 131 | 198 | 31/41 | 0.51 |
| ISS scatter | 40/41 | 0.98 | 1.11 | 134 | 252 | 1/41 | 0.41 |
| DustTrak B | 41/41 | 1.00 | 0.00 | 136 | 259 | 11/41 | 0.24 |
| Ionisation A | 32/41 | 0.86 | 0.37 | 136 | 253 | 0/41 | 0.35 |
| Ionisation B | 40/41 | 0.98 | 1.11 | 142 | 738 | 4/41 | 0.24 |
| P-Trak | 41/41 | 0.99 | 0.37 | 147 | 665 | 2/41 | 0.15 |
| STS | 1/41 | 0.05 | 0.00 | 1016 | 1016 | 0/41 | 0.24 |
| **Full 8-channel** | 41/41 | 0.98 | 0.74 | 3 | 5 | 41/41 | 0.69 |
| ISS scatter alarm (ref.) | 41/41 | 0.99 | 0.74 | 134 | 250 | – | – |
| Ionisation A alarm (ref.) | 37/41 | 0.94 | 0.37 | 136 | 221 | – | – |

## Findings
1. **Detection speed comes from the gas plume thermocouple.** On its own it matches the full model (3 s median, 5 s P90, 41/41) and beats the ISS alarm in every test. Removing it raises the median to 131 s (the `without_plume_t` row in `all_subsets_loto.csv`). Removing any other channel changes nothing.
2. **Among smoke sensors, ISS obscuration is earliest** (131 s, ahead of the ISS scatter alarm in 31/41 tests). The other smoke channels all cluster at 134–147 s, i.e. they see the same smoke arriving at the same time, so they are redundant for speed. STS is effectively unusable as a detector (1/41).
3. **The smoke channels are what tell you *what is burning*.** Temperature alone identifies the material with F1 0.59. Greedy forward selection (`greedy_selection_loto.csv`) gives: T 0.59 → T+OBS 0.78 → +P-Trak 0.77 → +Ion B 0.79. All 8 channels give 0.69, i.e. extra channels add noise for the classifier.
4. **Recommended telemetry subset:** gas plume temp. + ISS obscuration (+ Ion B / P-Trak optionally). Two channels keep 100 % of detection speed (3 s), the same false-alarm rate (0.37/h), and a higher material F1 than all 8 channels (0.78 vs 0.69). This is 75 % less telemetry (8 vs 32 B/s at 1 Hz float32), 2 Isolation Forests instead of 8, and 8 detector features instead of 32.

## Caveats (state these in the paper)
- The plume thermocouple sits right above the SAME heated sample, so part of its 3 s response is the heater itself. In a real cabin a temperature sensor is rarely this close to the source. The smoke-only latency (~131 s) is the conservative figure.
- With one fire event and one clean period per test, detection F1 saturates near 1. **Latency is the discriminating metric.** Differences in false alarms are only 0–3 events across about 2.7 h of clean air and are not statistically meaningful.
- The subset and greedy choices were selected on the same CV results, so their classifier F1 is optimistic. 38 labelled tests means each test is about 2.6 % accuracy.
- 8-fold results (`*_gkf8.*`) agree with LOTO. The 8-fold greedy pass was stopped early and is not included.

## Files
- `sensor_table_{loto,gkf8}.{md,csv}`: the 8-row table (incl. leave-one-out columns `loo_*`)
- `reference_rows_*.csv`: full model + standard ISS alarms
- `all_subsets_*.csv`: detection metrics for all 255 channel subsets
- `best_subset_per_size_*.csv`, `minimal_subset_*.json`: smallest subsets + telemetry budget
- `classifier_*.csv`, `greedy_selection_loto.csv`: material classifier per subset
- `per_test_*.csv.gz`: every test × subset alert time and false alarms
- `figures/latency_per_sensor_loto.png` (300 DPI, + PDF): **main paper figure**
- `figures/greedy_selection_loto.png` (300 DPI, + PDF): channels vs delay and material F1
