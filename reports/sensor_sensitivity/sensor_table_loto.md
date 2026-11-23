**Per-sensor sensitivity, single channel vs. full 8-channel detector (leave-one-test-out, t0 = heater on)**

| Rank | Sensor channel | Type | Detected | Det. acc. | Det. F1 | False alarms /h | Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) | Beats ISS alarm | Material F1 (alone) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Gas plume temp. (`plume_t`) | thermocouple | 41/41 | 0.99 | 0.99 | 0.37 | 3 | 5 | 0 | 41/41 | 0.59 |
| 2 | ISS obscuration (`iss_obs`) | optical smoke | 41/41 | 0.99 | 0.99 | 0.37 | 131 | 198 | 128 | 31/41 | 0.51 |
| 3 | ISS scatter (`iss_sct`) | optical smoke | 40/41 | 0.98 | 0.98 | 1.11 | 134 | 252 | 131 | 1/41 | 0.41 |
| 4 | DustTrak B (mass) (`dtb`) | aerosol mass | 41/41 | 1.00 | 1.00 | 0.00 | 136 | 259 | 133 | 11/41 | 0.24 |
| 5 | Ionisation A (`ion_a`) | ionisation smoke | 32/41 | 0.88 | 0.86 | 0.37 | 136 | 253 | 134 | 0/41 | 0.35 |
| 6 | Ionisation B (`ion_b`) | ionisation smoke | 40/41 | 0.98 | 0.98 | 1.11 | 142 | 738 | 138 | 4/41 | 0.24 |
| 7 | P-Trak (ultrafine) (`ptrak`) | particle counter | 41/41 | 0.99 | 0.99 | 0.37 | 147 | 665 | 144 | 2/41 | 0.15 |
| 8 | STS detector (`sts`) | optical smoke | 1/41 | 0.51 | 0.05 | 0.00 | 1016 | 1016 | 1013 | 0/41 | 0.24 |

**Reference rows**

| Reference | Type | Detected | Det. acc. | Det. F1 | False alarms /h | Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) | Material F1 |
|---|---|---|---|---|---|---|---|---|---|
| Full 8-channel detector | baseline | 41/41 | 0.98 | 0.98 | 0.74 | 3 | 5 | 0 | 0.69 |
| ISS scatter alarm (standard) | reference threshold | 41/41 | 0.99 | 0.99 | 0.74 | 134 | 250 | 131 | – |
| Ionisation A alarm (standard) | reference threshold | 37/41 | 0.94 | 0.94 | 0.37 | 136 | 221 | 133 | – |
