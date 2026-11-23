# Sensor ablation and thermal-lag simulation (SAME, 8-fold)

Script: `src/thermal_lag.py` (run from the project root). Input: `Sensor_Systems_and_Telemetry_Integration/ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl` (41 tests).

```bash
python src/thermal_lag.py          # full run (~6 min on 8 CPUs)
python src/thermal_lag.py --check  # self-checks only
python src/thermal_lag.py --plots  # rebuild tables and figures from saved results
```

## Question
The detector alerts a median of 3 s after heater-on, but the gas plume thermocouple (`plume_t`), which sits right above the heated sample, triggers every alert. How well does the detector work if that sensor is missing, or responds 15, 30 or 60 s late?

## Headline results (q = 1.0, 41/41 tests detected in every row)

| Scenario | Median delay, heater-aligned | False alarms/h | Median delay, arming (causal) | False alarms/h |
|---|---|---|---|---|
| Baseline (no delay) | 3 s | 0.37 (1) | 3 s | 2.11 (5) |
| `plume_t` removed | 131 s | 0.37 (1) | 130 s | 4.64 (11) |
| Shift 15 s | 18 s | 0.37 (1) | 18 s | 2.95 (7) |
| Shift 30 s | 33 s | 0.74 (2) | 33 s | 2.53 (6) |
| Shift 60 s | 63 s | 0.74 (2) | 63 s | 1.69 (4) |
| Slow sensor, τ = 15 s | 131 s | 0.37 (1) | 7 s | 4.64 (11) |
| Slow sensor, τ = 30 s | 131 s | 0.37 (1) | 59 s | 5.07 (12) |
| Slow sensor, τ = 60 s | 131 s | 0.37 (1) | 7 s | 4.64 (11) |
| ISS scatter alarm (reference) | 134 s | 0.74 (2) | 133 s | 0.42 (1) |

Brackets: number of false alarms, in 2.71 h (heater-aligned) or 2.37 h (arming) of clean air. Confidence intervals, P90 and maximum delays are in `summary_table.md`.

## Findings
1. **Without `plume_t` the early-warning advantage is gone.** The median delay is 131 s, about 1 s ahead of the ISS scatter alarm. The 7 smoke sensors alone beat that alarm in 33/41 tests, but by very little.
2. **A pure transport delay costs exactly the delay.** 15, 30 and 60 s give 18, 33 and 63 s, under both baselines. Even at 60 s the detector is ahead of the ISS alarm in all 41 tests (median lead 71 s).
3. **The fused detector does not handle a slow sensor, although the heat is visible.** With the heater-aligned baseline, every τ falls back to the no-temperature result (131 s, 0/41 alerts from `plume_t`). A plain threshold on the same slowed temperature still detects 41/41 in 3 s with one false alarm. The cause is score saturation:
   - An Isolation Forest score stops growing once a reading is outside the training range, so a 50 °C rise scores about the same as a small drift.
   - The slow sensor filters out noise, so ordinary drift in unseen clean air looks abnormal and the clean-air reference level rises (0.65 to 0.73–0.76 raw).
   - The normalised heating peak then drops (1.16 to 0.99–1.04) below the shared threshold (1.07), which the smoke sensors set.
4. **With the arming baseline the slow-sensor result is unstable, not good.** The peak and the threshold are almost equal (about 1.06–1.07 vs 1.06), so the median jumps between 7 and 59 s with no physical order, and P90 stays at 129–193 s. Do not quote the 7 s figures alone.
5. **The false-alarm curves barely depend on the temperature channel** (`figures/fa_rate_curves.png`). With the heater-aligned baseline, every scenario stays inside the baseline's 95 % band across the sweep, except the ablation at q = 0.95–0.96 (below it). With the arming baseline, the 30 s shift is above the band at two settings (q = 0.995 and 0.9965) and the 60 s shift is also raised at mid settings. Removing `plume_t` lowers false alarms slightly at sensitive settings (36.9 to 28.8 per hour at q = 0.95), but the paired interval includes zero. At q = 1.0 the smoke sensors set the rate.
6. **The arming baseline costs false alarms.** Without knowing the heater time, the baseline detector gives 2.11 false alarms/h instead of 0.37, all in tests 62, 97 and 107. With `plume_t` removed this becomes 4.64/h, higher than the ISS scatter alarm (0.42/h).
7. **A slightly stricter threshold helps where there is headroom.** At 1.02 × the q = 1.0 threshold, the heater-aligned baseline, 15 s and 30 s shifts have no false alarms and unchanged delays. At 1.05 ×, the arming baseline and all three shifts have none either. These margins were read off the same tests, so they need confirming on new data.

## Recommendation
- Report 131 s as the smoke-only figure, and 18–63 s as the range for a sensor with a 15–60 s transport delay. The 3 s figure only applies to a sensor at the source.
- Do not rely on the maximum over Isolation Forest scores for a slow thermometer. Give temperature its own rule (a threshold or rate-of-rise check in parallel) or an unbounded score. Test that rule with the arming baseline: the plain threshold detects 41/41 in 4–5 s there, but with 8.4, 3.8 and 2.5 false alarms/h for τ = 15, 30 and 60 s.

## Method
- **Detector:** unchanged per-sensor Isolation Forest protocol from `src/anomaly.py` (clean-air training, 8-fold GroupKFold by test, inner 4-fold calibration, 3 s persistence). Retrained and recalibrated for every scenario.
- **Delay models:** pure shift `y(t) = x(t − D)` and first-order lag `y += (x − y)(1 − exp(−1/τ))` at 1 Hz. Applied to the raw temperature before baseline correction and rolling features. The heater-on time does not move.
- **Two baselines:** *heater-aligned* (last 120 s before heater-on) matches the published results but uses the event time. *Arming* (first 30 s after power-on, from Task 5's `src/evaluate_chained.py`) is causal. Nothing is trained or scored while arming.
- **Curves:** 15 calibration quantiles from 0.95 to 1.0, then 1.02–1.20 × the q = 1.0 threshold. Thresholds come from training folds only.
- **Uncertainty:** exact Poisson intervals for false alarms, test-level bootstrap (2000) for curves, medians and paired differences.
- **Material labels** in the per-test file use the audited Task 3 manifest. They do not affect the detector.

## Checks
- Baseline and ablation reproduce `reports/anomaly/per_test.csv` at q = 1.0: alert times identical in 41/41 tests each, same single false alarm (test 97), same ISS alarm times.
- Features equal `src/features.py` bit for bit (heater-aligned) and `src/evaluate_chained.py` (arming, 3 tests checked).
- A 0 s delay returns the input unchanged. Step responses are exact. Randomising all data after a cut leaves earlier values bit-identical.
- Versions are in `run_info.json` (scikit-learn 1.7.2). Task 4 and Task 5 used 1.9, where one extra clean-air false alarm appears (0.74/h instead of 0.37/h), so compare false alarms only within one version.

## Caveats
- **Very little clean air.** One false alarm is about 0.4/h. Differences of 1–3 alarms are noise.
- **Filled-in history.** A delayed sensor needs readings from before the recording and inside recording gaps. These are filled by reflecting the test's own neighbouring readings. The first version froze the last reading through gaps. This was changed after seeing results, because the frozen signal has zero variance, which the detector itself flags. The frozen version is kept as `--hold` scenarios. It mainly changes the 30 s shift with the heater-aligned baseline (median 87 s instead of 33 s, through the same saturation effect). The 15 s and 60 s medians stay within 1 s. With the arming baseline, the slow-sensor medians become 125–128 s.
- **Delay is only part of distance.** A distant sensor also sees a weaker, more mixed signal, which is not modelled. All delayed results are therefore optimistic, including the plain-threshold 3 s.
- **Chance alerts.** At sensitive settings some "detections" are false alarms that happened to fall in the window. `chance_alert_share` in `curves.csv` estimates this: 49 % of tests for the ablation at q = 0.95, 2 % at q = 1.0.

## Files
- `summary_table.md`: full q = 1.0 tables and the saturation diagnostics
- `curves.csv`: every scenario, baseline and threshold setting
- `per_fold.csv`: the same per fold; `per_test.csv.gz`: per test
- `fa_difference_vs_baseline.csv`: paired false-alarm differences
- `score_headroom_per_test.csv`: heating peak vs threshold per test
- `delay_by_material.csv`, `run_info.json`
- `figures/` (PNG and PDF): `fa_rate_curves` (main figure), `fa_rate_per_fold`, `delay_vs_false_alarms`, `delay_vs_simulated_lag`

In the files, scenario names starting with `only_` and `thr_` are the temperature-only diagnostics.
