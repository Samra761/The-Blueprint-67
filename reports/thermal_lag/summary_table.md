**Heater-aligned baseline (published protocol), q = 1.0, 8-fold GroupKFold, 41 tests**

| Scenario | Detected | Median delay (s) [95 % CI] | P90 (s) | Max (s) | Lead vs ISS (s) | Beats ISS | Triggered by plume_t | False alarms | FA/h [95 % Poisson CI] |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (no delay) | 41/41 | 3 [3, 4] | 5 | 6 | 131 | 41/41 | 41/41 | 1 | 0.37 [0.01, 2.06] |
| plume_t removed | 41/41 | 131 [129, 140] | 251 | 837 | 1 | 33/41 | 0/41 | 1 | 0.37 [0.01, 2.06] |
| Shift 15 s | 41/41 | 18 [18, 18] | 20 | 22 | 116 | 41/41 | 41/41 | 1 | 0.37 [0.01, 2.06] |
| Shift 30 s | 41/41 | 33 [33, 33] | 35 | 37 | 101 | 41/41 | 41/41 | 2 | 0.74 [0.09, 2.67] |
| Shift 60 s | 41/41 | 63 [63, 63] | 65 | 67 | 71 | 41/41 | 40/41 | 2 | 0.74 [0.09, 2.67] |
| Slow sensor, τ = 15 s | 41/41 | 131 [129, 140] | 251 | 837 | 1 | 33/41 | 0/41 | 1 | 0.37 [0.01, 2.06] |
| Slow sensor, τ = 30 s | 41/41 | 131 [129, 140] | 251 | 837 | 1 | 33/41 | 0/41 | 1 | 0.37 [0.01, 2.06] |
| Slow sensor, τ = 60 s | 41/41 | 131 [129, 140] | 251 | 837 | 1 | 33/41 | 0/41 | 1 | 0.37 [0.01, 2.06] |
| ISS scatter alarm (ref.) | 41/41 | 134 [130, 191] | 250 | 839 | – | – | – | 2 | 0.74 [0.09, 2.67] |
| Ionisation A alarm (ref.) | 38/41 | 136 [131, 194] | 254 | 712 | – | – | – | 1 | 0.37 [0.01, 2.06] |

**Arming baseline (causal, deployable), q = 1.0, 8-fold GroupKFold, 41 tests**

| Scenario | Detected | Median delay (s) [95 % CI] | P90 (s) | Max (s) | Lead vs ISS (s) | Beats ISS | Triggered by plume_t | False alarms | FA/h [95 % Poisson CI] |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (no delay) | 41/41 | 3 [3, 4] | 5 | 6 | 130 | 41/41 | 41/41 | 5 | 2.11 [0.69, 4.93] |
| plume_t removed | 41/41 | 130 [127, 134] | 198 | 836 | 1 | 30/41 | 0/41 | 11 | 4.64 [2.32, 8.31] |
| Shift 15 s | 41/41 | 18 [18, 18] | 20 | 21 | 115 | 41/41 | 41/41 | 7 | 2.95 [1.19, 6.09] |
| Shift 30 s | 41/41 | 33 [33, 33] | 35 | 36 | 101 | 41/41 | 39/41 | 6 | 2.53 [0.93, 5.51] |
| Shift 60 s | 41/41 | 63 [63, 63] | 65 | 66 | 71 | 41/41 | 39/41 | 4 | 1.69 [0.46, 4.32] |
| Slow sensor, τ = 15 s | 41/41 | 7 [5, 60] | 129 | 198 | 123 | 38/41 | 31/41 | 11 | 4.64 [2.32, 8.31] |
| Slow sensor, τ = 30 s | 41/41 | 59 [6, 101] | 193 | 836 | 67 | 35/41 | 25/41 | 12 | 5.07 [2.62, 8.85] |
| Slow sensor, τ = 60 s | 41/41 | 7 [7, 8] | 129 | 836 | 124 | 40/41 | 35/41 | 11 | 4.64 [2.32, 8.31] |
| ISS scatter alarm (ref.) | 41/41 | 133 [130, 141] | 250 | 838 | – | – | – | 1 | 0.42 [0.01, 2.35] |
| Ionisation A alarm (ref.) | 37/41 | 136 [135, 195] | 253 | 712 | – | – | – | 1 | 0.42 [0.01, 2.35] |

**Why: temperature-only diagnostics and score headroom (q = 1.0)**

Median delay, detected, false alarms/h for the fused detector, for plume_t alone with an Isolation Forest (IF), and for plume_t alone with a plain threshold. Headroom: median over tests of the peak normalised plume_t score in 0-120 s vs the mean fused threshold.

| Baseline | Scenario | Fused | IF alone | Threshold alone | plume_t clean-air ref (raw IF) | Peak score | Fused threshold | Tests where peak clears |
|---|---|---|---|---|---|---|---|---|
| heater | Baseline (no delay) | 3 s, 41/41, 0.37 | 3 s, 41/41, 0.37 | 3 s, 41/41, 0.37 | 0.649 | 1.160 | 1.074 | 41/41 |
| heater | Shift 15 s | 18 s, 41/41, 0.37 | 18 s, 41/41, 0.37 | 18 s, 41/41, 0.37 | 0.649 | 1.170 | 1.075 | 41/41 |
| heater | Shift 30 s | 33 s, 41/41, 0.74 | 33 s, 41/41, 0.37 | 33 s, 41/41, 0.37 | 0.653 | 1.164 | 1.074 | 41/41 |
| heater | Shift 60 s | 63 s, 41/41, 0.74 | 63 s, 41/41, 0.74 | 63 s, 41/41, 0.37 | 0.662 | 1.135 | 1.074 | 41/41 |
| heater | Slow sensor, τ = 15 s | 131 s, 41/41, 0.37 | 4 s, 31/41, 0.37 | 3 s, 41/41, 0.37 | 0.729 | 1.045 | 1.074 | 0/41 |
| heater | Slow sensor, τ = 30 s | 131 s, 41/41, 0.37 | 4 s, 5/41, 0.37 | 3 s, 41/41, 0.37 | 0.748 | 1.009 | 1.074 | 0/41 |
| heater | Slow sensor, τ = 60 s | 131 s, 41/41, 0.37 | none, 0/41, 0.74 | 3 s, 41/41, 0.37 | 0.762 | 0.990 | 1.074 | 0/41 |
| arming | Baseline (no delay) | 3 s, 41/41, 2.11 | 3 s, 41/41, 0.42 | 3 s, 41/41, 1.27 | 0.623 | 1.207 | 1.062 | 41/41 |
| arming | Shift 15 s | 18 s, 41/41, 2.95 | 18 s, 41/41, 0.84 | 18 s, 41/41, 0.42 | 0.641 | 1.165 | 1.064 | 41/41 |
| arming | Shift 30 s | 33 s, 41/41, 2.53 | 33 s, 41/41, 0.00 | 33 s, 41/41, 1.69 | 0.633 | 1.204 | 1.061 | 41/41 |
| arming | Shift 60 s | 63 s, 41/41, 1.69 | 63 s, 41/41, 0.00 | 63 s, 41/41, 1.27 | 0.644 | 1.181 | 1.063 | 41/41 |
| arming | Slow sensor, τ = 15 s | 7 s, 41/41, 4.64 | 5 s, 41/41, 1.27 | 4 s, 41/41, 8.44 | 0.717 | 1.072 | 1.056 | 31/41 |
| arming | Slow sensor, τ = 30 s | 59 s, 41/41, 5.07 | 6 s, 30/41, 0.42 | 4 s, 41/41, 3.80 | 0.721 | 1.062 | 1.058 | 25/41 |
| arming | Slow sensor, τ = 60 s | 7 s, 41/41, 4.64 | 6 s, 41/41, 0.42 | 5 s, 41/41, 2.53 | 0.718 | 1.073 | 1.058 | 35/41 |
