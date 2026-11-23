**Per-sensor sensitivity, single channel vs. full 8-channel detector (8-fold GroupKFold, t0 = heater on)**

| Rank | Sensor channel | Type | Detected | Det. acc. | Det. F1 | False alarms /h | Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) | Beats ISS alarm | Material F1 (alone) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Gas plume temp. (`plume_t`) | thermocouple | 41/41 | 0.99 | 0.99 | 0.37 | 3 | 5 | 0 | 41/41 | 0.62 |
| 2 | ISS obscuration (`iss_obs`) | optical smoke | 41/41 | 1.00 | 1.00 | 0.00 | 132 | 251 | 129 | 31/41 | 0.42 |
| 3 | ISS scatter (`iss_sct`) | optical smoke | 41/41 | 0.99 | 0.99 | 1.85 | 134 | 253 | 131 | 4/41 | 0.42 |
| 4 | Ionisation A (`ion_a`) | ionisation smoke | 22/41 | 0.76 | 0.69 | 0.37 | 136 | 247 | 132 | 0/41 | 0.38 |
| 5 | DustTrak B (mass) (`dtb`) | aerosol mass | 41/41 | 0.99 | 0.99 | 0.37 | 136 | 259 | 133 | 11/41 | 0.29 |
| 6 | Ionisation B (`ion_b`) | ionisation smoke | 40/41 | 0.98 | 0.98 | 1.11 | 140 | 842 | 136 | 6/41 | 0.24 |
| 7 | P-Trak (ultrafine) (`ptrak`) | particle counter | 41/41 | 0.98 | 0.98 | 1.11 | 147 | 665 | 144 | 3/41 | 0.15 |
| 8 | STS detector (`sts`) | optical smoke | 1/41 | 0.49 | 0.05 | 0.74 | 1014 | 1014 | 1011 | 0/41 | 0.27 |

**Reference rows**

| | Sensor channel | Type | Detected | Det. acc. | Det. F1 | False alarms /h | Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) | | Material F1 (alone) |
|---|---|---|---|---|---|---|---|---|---|
| Full 8-channel detector | baseline | 41/41 | 0.98 | 0.98 | 0.74 | 3 | 5 | 0 | 0.66 |
| ISS scatter alarm (standard) | reference threshold | 41/41 | 0.99 | 0.99 | 0.74 | 134 | 250 | 131 | – |
| Ionisation A alarm (standard) | reference threshold | 38/41 | 0.95 | 0.95 | 0.37 | 136 | 254 | 133 | – |
