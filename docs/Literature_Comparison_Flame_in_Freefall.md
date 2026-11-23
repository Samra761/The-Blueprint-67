# Literature Comparison: Spacecraft Fire Detection vs. Flame in Freefall

**Task 2** · Team The Blueprint 67 · NASA Space Apps Challenge 2026, Islamabad · Revised Nov 24, 2026 (supersedes the Nov 22 version)

> **What changed since Nov 22.** All "this work" metrics now come from the committed causal LOTO artifacts (`results/chained_loto_summary.md`, `reports/sensor_sensitivity/sensor_table_loto.md`) and replace the Nov 15 hackathon figures. The bibliography is checked against Crossref and NASA NTRS, which corrected four citations (see [Sourcing notes](#sourcing-and-verification-notes)). Three IEEE references were added. The LaTeX version is in `paper/related_work.tex` and `paper/references.bib`.

**Terms used below.**

| Term | Meaning |
|---|---|
| `arming` | **Deployable** regime. The clean-air reference is the first 30 s after power-on, so no knowledge of when heating starts is needed. Primary result. |
| `heater_oracle` | **Comparison** regime. The clean-air reference is the last 120 s before heater-on. This needs the event time, so it cannot be deployed. It matches the earlier repository protocol. |
| `psi101` (41 tests) | All 41 flight tests, including tests 60 → Silicone, 103 → Silicone and 105 → Teflon, whose labels were recovered from the NASA PSI-101 table (Task 3 audit). |
| `repo` (38 tests) | The original cohort. The three tests whose files sit in two material folders are excluded. |
| *N* | Number of consecutive anomalous 1 Hz ticks required for an alarm. N = 3 is the operating point. |
| FAR | False alarms per clean-air hour. |

Stage 1 (detection) uses no labels, so its numbers are the same for both cohorts. Every latency is measured from heater-on.

## 1. Comparison table

| # | Work (venue, year) | Setting and data | Approach | Validation and reported findings | Flame in Freefall (this work) |
|---|---|---|---|---|---|
| 1 | Meyer, Urban, Mulholland, Bryg, Yuan, Ruff, Cleary, Yang. *Fire Safety Journal* 98:74–81, 2018 [1] | ISS SAME; oxidative pyrolysis of 5 spacecraft materials; 2 flight smoke detectors | Particle morphology and size; evaluation of existing detectors; no new detector | Particles not significantly different from 1-g. The detectors did not detect every smoke type. Qualitative. | Same SAME family and 5 materials (PSI-101, 41 tests). A trained detector flags **41/41** tests at a median of **3.0 s** (P90 5.0 s) after heater-on (`arming`, N = 3), vs **133 s** for the recalibrated ISS scatter alarm |
| 2 | Urban, Griffin, Ruff, Cleary, Yang, Mulholland, Yuan. SAE 2005-01-2930, 35th ICES, Rome, **2005** [2] | Review plus the Comparative Soot Diagnostics (CSD) flight experiment | Microgravity smoke particle-size measurement | Size distributions of solid particulate smoke increased substantially in µg (in tension with row 1) | No particle sizing. Detector-response ratios (ion-B/scatter, scatter/obscuration) are the top two SHAP features of the legacy classifier. Adding ISS obscuration to plume temperature raises material macro-F1 from **0.59 to 0.78** (greedy ablation) |
| 3 | Fortenberry, Wang, Cleary, Easton, Berger, Meyer. NASEM decadal-survey white paper, NTRS 20210022143, 2021 [3] | Synthesis of ISS and ground results | Position paper arguing for multi-property detection | The ISS detector failed to alarm for **Kapton smoke in half of tests and Teflon in two-thirds**. A single-property detector is unlikely to succeed in all scenarios. | Multi-channel answer: 8 per-channel Isolation Forests with max-score fusion. Detects **11/11 Kapton and 7/7 Teflon** tests (`psi101` labels). *Caveat:* our ISS baseline is recalibrated (also 41/41), not run at the flight threshold |
| 4 | Guzman, Krause, Matar, Prelich, Urban. Poster, Gordon Research Conf., Jun–Jul 2026, NTRS 20260005612 [4] | Partial- and low-gravity habitats; aerosol rig with reference instruments; FDS plume model | Detector signals compared with reference instruments | The ionization detector gave the earliest signal. The system must discriminate combustion particles from cabin and lunar dust. No numeric metrics. | Different earliest channel: the **gas-plume thermocouple** (3 s, triggers all 41 alarms). The best smoke-only channel is **ISS obscuration at 131 s**; ionization A and B take 136 s and 142 s. No dust tests. The rig and heating differ |
| 5 | He, Si, Yang, Xu, Zhang, Wang, Sun. *Sensors* 25(20):6451, 2025 [5] | Terrestrial IoT fire dataset; 4 inputs; 49,888 windows of 20 s | Supervised BiLSTM with layer normalization and self-attention | 10-fold CV: 98.38 % accuracy, FPR 1.50 % | Unsupervised, trained on clean air only, with no fire labels. Per-test LOTO. FAR per clean-air hour: **0.00 /h (N = 2)** to **2.11 /h (N = 3)** (`arming`) |
| 6 | Nakip, Güzeliş, Yıldız. *IEEE Access* 9:84204–84216, 2021 [6] | NIST residential tests: 9 bedroom experiments, 25,796 samples | Recurrent trend-predictive neural network fusing the trend and level of each sensor | 10-fold CV: 96 % accuracy, F1 0.93; alarm within 11 s of fire onset | Event-level latency per held-out flight test. Real-time cost reported: **P99 269 ms per 1 Hz tick** on a single thread |
| 7 | Baek, Alhindi, Jeong, Jeong, Seo, Kang, Heo. *IEEE Sensors J.* 21(24):27982–27992, 2021 [7] | Indoor building fires; multi-sensor prototype | Similarity matching on signal shape (DTW-based) | Reports fewer false alarms. Numeric results not retrieved (paywalled) | Adds **5-class material identification**: steady-state accuracy **0.805** (33/41, `psi101`) / **0.763** (29/38, `repo`) |

Method reference: the Isolation Forest [8] (IEEE ICDM 2008) is the Stage 1 learner. It is cited in the prose, not as a comparison row.

## 2. Our headline metrics (causal chained LOTO-41, PSI-101 SAME, 1 Hz)

Sources: `results/chained_loto_summary.md` and `results/chained_loto_metrics.json` (rows marked †), plus `reports/sensor_sensitivity/sensor_table_loto.md` (rows marked ‡).

| Metric | Cohort | `arming` (causal, deployable) | `heater_oracle` (needs heater-on time) |
|---|---|---|---|
| † Stage 1 detection, N = 3 | all 41 | 41/41; median 3.0 s, P90 5.0 s; FAR 2.11 /h | 41/41; median 3.0 s, P90 5.0 s; FAR 2.22 /h |
| † Stage 1 detection, N = 2 | all 41 | 41/41; median 2.0 s, P90 4.0 s; FAR 0.00 /h | 41/41; median 2.0 s, P90 4.0 s; FAR 1.11 /h |
| † Recalibrated ISS scatter alarm, N = 3 | all 41 | 41/41; median 133 s; FAR 0.42 /h | 41/41; median 134 s; FAR 0.74 /h |
| † Stage 2, heater-aligned (oracle windows) | `psi101` (41) | 0.756 (31/41), macro-F1 0.747 | 0.756 (31/41), macro-F1 0.746 |
|  | `repo` (38) | 0.711 (27/38), macro-F1 0.704 | 0.684 (26/38), macro-F1 0.658 |
| † Chained, steady state, N = 3 | `psi101` (41) | **0.805 (33/41), macro-F1 0.800** | 0.825 (33/40), macro-F1 0.820 |
|  | `repo` (38) | 0.763 (29/38), macro-F1 0.754 | 0.811 (30/37), macro-F1 0.812 |
| † Chained, at first alert, N = 3 | `psi101` (41) | 0.268 (11/41) | 0.268 (11/41) |
| † Per-tick CPU latency, single thread (`chained_loto_latency_ms.csv`) | 4 streamed tests, 6,193 ticks | mean 85.1, **P50 58.7, P99 269.1**, max 547.5 ms (≈ 55 % of the 1 s budget at worst) | — |
| ‡ Plume thermocouple only | all 41 | not evaluated | 41/41; median 3 s, P90 5 s; 0.37 FA/h; beats ISS alarm 41/41; material F1 0.59 |
| ‡ ISS obscuration only (best smoke-only) | all 41 | not evaluated | 41/41; median 131 s, P90 198 s; 0.37 FA/h; beats ISS alarm 31/41; material F1 0.51 |
| ‡ Full 8-channel detector | all 41 | not evaluated | 41/41; median 3 s, P90 5 s; 0.74 FA/h; material F1 0.69 |

The ‡ rows (Task 4 per-sensor ablation) use the per-sensor detector protocol of `src/anomaly.py`, under LOTO-41. That protocol takes its clean-air reference from just before heater-on, so the rows sit in the `heater_oracle` column. Their material F1 is a LOTO-38 Random Forest on `repo` labels.

**Retired Nov 15 figures.** The 0.37 FA/h, RF 28/38 and "median 3 s" values came from the 8-fold, heater-referenced protocol. They are no longer headline numbers, and "71–74 %" is replaced by the cohort-specific accuracies above. The FLEX burn-time model is outside the scope of this comparison.

## 3. Where we add something new

| Dimension | Literature (rows 1–7) | Flame in Freefall |
|---|---|---|
| Lead time vs. flight alarm | Not reported as a distribution | Per test; median 3.0 s vs 133 s (`arming`, N = 3). Smoke-only best channel 131 s |
| False alarms | Spacecraft: not reported. Terrestrial: FPR per sample or window | Per clean-air hour: 0.00–2.11 /h (`arming`, N = 1–3) |
| Validation unit | Qualitative, or k-fold over pooled samples or windows | LOTO per flight test (41 folds); strictly causal trailing-window features; leakage self-checks |
| Material identification | Not addressed | 5 classes: 0.805 (`psi101`) / 0.763 (`repo`) steady-state accuracy (chained, N = 3) |
| Real-time cost | Rarely reported | P99 269 ms per 1 Hz tick, single thread |
| Open flaming / dust / other rigs | Dust overlap flagged (row 4) | Not tested (stated limitation) |

## Key takeaways

- **The spacecraft literature (rows 1–4) establishes the problem.** Flight smoke detectors miss some materials, and single-property detection is not enough. The findings are qualitative, so a numeric head-to-head is not possible. That gap (lead-time distributions, FAR per hour, per-test held-out validation, material identification) is our contribution.
- **The terrestrial ML work (rows 5–7) is supervised.** It reports pooled-sample k-fold metrics on buoyant, Earth-gravity fires and does not identify the burning material.
- **Report our results with their caveats.** All data come from one ISS glovebox rig with pre-ignition oxidative pyrolysis. The 3 s lead is carried by the plume thermocouple directly above the heated sample; the conservative smoke-only figure is 131 s. The ISS baseline is recalibrated, not the fielded alarm.

## Sourcing and verification notes

- **Corrections to the Nov 22 PDF.** Row 1's first author is **M. E. Meyer** per the publisher (Crossref), although the NIST page lists J. Yang first. Row 2 is a **2005** SAE/ICES paper; "2021" was only the date of the NIST web record. Row 3's authors are **C. Fortenberry et al.**, not T'ien/Urban/Wichman, who are co-signers. Row 4 is a poster at a Gordon Research Conference. Row 5's full title includes "A Novel Integrated Model".
- **IEEE references added:** [6] *IEEE Access*, [7] *IEEE Sensors Journal* and [8] IEEE ICDM. All three were checked against Crossref. A fourth candidate (IEEE Sensors J. 2025, multiclass SVDD-DTW) was dropped because its author list could not be confirmed.
- **Still unverified, so do not quote:** numeric results of [7] (paywalled), and the submission month of [3].
- **Findings for rows 1–4** come from abstracts, the NTRS records and the white-paper text, not from full papers.
- **Still excluded:** a non-peer-reviewed web post claiming 96.7 % accuracy for a Bayesian UV-visible detector.

## References (IEEE style)

[1] M. E. Meyer, D. L. Urban, G. W. Mulholland, V. Bryg, Z.-G. Yuan, G. A. Ruff, T. Cleary, and J. Yang, "Evaluation of spacecraft smoke detector performance in the low-gravity environment," *Fire Saf. J.*, vol. 98, pp. 74–81, Jun. 2018, doi: 10.1016/j.firesaf.2018.04.004.

[2] D. Urban, D. Griffin, G. Ruff, T. Cleary, J. Yang, G. Mulholland, and Z. Yuan, "Detection of smoke from microgravity fires," SAE Tech. Paper 2005-01-2930, 35th Int. Conf. Environmental Systems (ICES), Rome, Italy, Jul. 2005, doi: 10.4271/2005-01-2930.

[3] C. Fortenberry, X. Wang, T. Cleary, J. Easton, G. Berger, and M. Meyer, "Research questions and challenges for improved spacecraft fire detection," topical white paper to the NASEM Biological and Physical Sciences Decadal Survey 2022–2032, NASA NTRS 20210022143, 2021.

[4] F. J. Guzman, T. Krause, M. Matar, M. Prelich, and D. Urban, "Early fire detection system for partial and low gravity environments," poster, Gordon Research Conf., Easton, MA, USA, Jun. 28–Jul. 3, 2026, NASA NTRS 20260005612.

[5] Z. He, Y. Si, L. Yang, N. Xu, X. Zhang, M. Wang, and X. Sun, "BiLSTM-LN-SA: A novel integrated model with self-attention for multi-sensor fire detection," *Sensors*, vol. 25, no. 20, Art. no. 6451, Oct. 2025, doi: 10.3390/s25206451.

[6] M. Nakip, C. Güzeliş, and O. Yıldız, "Recurrent trend predictive neural network for multi-sensor fire detection," *IEEE Access*, vol. 9, pp. 84204–84216, 2021, doi: 10.1109/ACCESS.2021.3087736.

[7] J. Baek, T. J. Alhindi, Y.-S. Jeong, M. K. Jeong, S. Seo, J. Kang, and Y. Heo, "Intelligent multi-sensor detection system for monitoring indoor building fires," *IEEE Sensors J.*, vol. 21, no. 24, pp. 27982–27992, Dec. 2021, doi: 10.1109/JSEN.2021.3124266.

[8] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, Dec. 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.
