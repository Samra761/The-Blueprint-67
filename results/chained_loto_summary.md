# Chained causal LOTO evaluation (SAME, 1 Hz)

Generated 2026-11-23T19:11:32+00:00; 41 LOTO folds (complete: True); seed 0; scikit-learn 1.9.0; runtime 4619.3 s.

Baselines: `arming` = causal, first 30 s after power-on (primary); `heater_oracle` = repo baseline (last 120 s before heater-on; needs the event time, so not deployable). Thresholds at q = 1.0 on training-fold clean air.

## Detection (Stage 1 = end-to-end trigger)

| Baseline | N ticks | Detected | Median latency (s) | P90 (s) | FAR / clean h | ISS scatter detected | ISS median (s) | ISS FAR / h |
|---|---|---|---|---|---|---|---|---|
| arming | 1 | 41/41 | 1.0 | 3.0 | 1.27 | 41/41 | 131.0 | 0.42 |
| arming | 2 | 41/41 | 2.0 | 4.0 | 0.00 | 41/41 | 133.0 | 0.42 |
| arming | 3 | 41/41 | 3.0 | 5.0 | 2.11 | 41/41 | 133.0 | 0.42 |
| heater_oracle | 1 | 41/41 | 80.0 | 191.0 | 0.37 | 40/41 | 133.0 | 1.11 |
| heater_oracle | 2 | 41/41 | 2.0 | 4.0 | 1.11 | 40/41 | 133.0 | 1.11 |
| heater_oracle | 3 | 41/41 | 3.0 | 5.0 | 2.22 | 41/41 | 134.0 | 0.74 |

Repo-reported standalone Stage 1 (8-fold, heater baseline, N=3): 41/41 detected, median 3.0 s, P90 5.0 s, FAR 0.37/h; ISS 134.0 s, 0.74/h.

## Material identification

Oracle Stage 2 = heater-aligned windows starting 120-570 s, probabilities averaged per test (repo protocol). Chained = Stage 2 run only on ticks with an active Stage 1 alarm.

| Baseline / labels | Oracle acc | Oracle macro-F1 | Oracle acc excl. 720 s-aged |
|---|---|---|---|
| arming/psi101 | 0.756 (31/41) | 0.747 | 0.838 (31/37) |
| arming/repo | 0.711 (27/38) | 0.704 | 0.765 (26/34) |
| heater_oracle/psi101 | 0.756 (31/41) | 0.746 | 0.838 (31/37) |
| heater_oracle/repo | 0.684 (26/38) | 0.658 | 0.765 (26/34) |

| Baseline / labels / N | First-alert acc | Until-600 s acc | Steady-state acc | Steady macro-F1 | Steady acc excl. aged | Tick acc transient | Tick acc steady | Share of errors in transient |
|---|---|---|---|---|---|---|---|---|
| arming/psi101/n1 | 0.220 (9/41) | 0.610 | 0.737 (28/38) | 0.721 | 0.800 | 0.298 | 0.708 | 0.72 |
| arming/psi101/n2 | 0.244 (10/41) | 0.610 | 0.732 (30/41) | 0.715 | 0.811 | 0.298 | 0.706 | 0.68 |
| arming/psi101/n3 | 0.268 (11/41) | 0.707 | 0.805 (33/41) | 0.800 | 0.892 | 0.291 | 0.755 | 0.59 |
| arming/repo/n1 | 0.211 (8/38) | 0.553 | 0.743 (26/35) | 0.739 | 0.812 | 0.279 | 0.698 | 0.74 |
| arming/repo/n2 | 0.211 (8/38) | 0.553 | 0.737 (28/38) | 0.729 | 0.824 | 0.283 | 0.701 | 0.70 |
| arming/repo/n3 | 0.237 (9/38) | 0.711 | 0.763 (29/38) | 0.754 | 0.853 | 0.275 | 0.758 | 0.61 |
| heater_oracle/psi101/n1 | 0.341 (14/41) | 0.600 | 0.703 (26/37) | 0.661 | 0.765 | 0.258 | 0.709 | 0.55 |
| heater_oracle/psi101/n2 | 0.293 (12/41) | 0.707 | 0.775 (31/40) | 0.760 | 0.838 | 0.281 | 0.640 | 0.61 |
| heater_oracle/psi101/n3 | 0.268 (11/41) | 0.732 | 0.825 (33/40) | 0.820 | 0.892 | 0.285 | 0.648 | 0.57 |
| heater_oracle/repo/n1 | 0.342 (13/38) | 0.676 | 0.706 (24/34) | 0.668 | 0.774 | 0.299 | 0.726 | 0.57 |
| heater_oracle/repo/n2 | 0.289 (11/38) | 0.658 | 0.784 (29/37) | 0.778 | 0.853 | 0.305 | 0.653 | 0.62 |
| heater_oracle/repo/n3 | 0.263 (10/38) | 0.684 | 0.811 (30/37) | 0.812 | 0.882 | 0.310 | 0.650 | 0.57 |

## Per-tick CPU latency (single thread, streaming implementation)

CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel. Streamed tests: [58, 69, 82, 96].

| Ticks | n | Mean (ms) | P50 (ms) | P99 (ms) | Max (ms) |
|---|---|---|---|---|---|
| all_armed_ticks | 6193 | 85.13 | 58.71 | 269.11 | 547.52 |
| stage1_only | 3970 | 80.86 | 51.39 | 244.82 | 547.52 |
| stage1_plus_stage2 | 2223 | 92.75 | 64.28 | 282.24 | 458.81 |

## Leakage self-checks

- Future perturbation (features at or before the cut bit-identical): tests [58, 75, 91].
- Streaming vs batch, test 58: max |score diff| 0.0e+00, alarm agreement 1.0000, max scaled window feature diff 9.0e-08, max |proba diff| 0.007, Stage 2 argmax agreement 1.0000.
- Streaming vs batch, test 69: max |score diff| 0.0e+00, alarm agreement 1.0000, max scaled window feature diff 9.7e-09, max |proba diff| 0.000, Stage 2 argmax agreement 1.0000.
- Streaming vs batch, test 82: max |score diff| 0.0e+00, alarm agreement 1.0000, max scaled window feature diff 2.7e-08, max |proba diff| 0.000, Stage 2 argmax agreement 1.0000.
- Streaming vs batch, test 96: max |score diff| 0.0e+00, alarm agreement 1.0000, max scaled window feature diff 3.7e-08, max |proba diff| 0.002, Stage 2 argmax agreement 1.0000.
- Window feature differences are scaled by each feature's largest |value| in the test (min 1). They are pandas rolling round-off: the online add/remove algorithm leaves a residue of about 1e-7 of the channel magnitude where the exact value is 0 (e.g. std of a constant channel), which can move a few RF trees across near-zero split thresholds.
