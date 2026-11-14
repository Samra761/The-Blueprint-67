"""Phase 1: per-test baseline correction, trailing rolling features, channel ratios.

Run from the project root (after src/load.py):
    python src/features.py

Input : data/processed/all_tests.parquet
Output: data/processed/features.parquet
        data/processed/feature_columns.json   (feature lists for each model)

All rolling windows are TRAILING (only past data), so every feature at time t
could have been computed live at time t - safe for early warning.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "all_tests.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"
OUT_COLS = ROOT / "data" / "processed" / "feature_columns.json"

# Only these sensor channels are model inputs. Everything else (heater flags,
# SampleResistance, RunMode, EventCode, solenoids/power, TestNumber, Time, date)
# is deliberately never turned into a feature.
CHANNELS = {
    "ISSDetectorObscurationMeasurement(Volts)": "iss_obs",
    "ISSDetectorScatterMeasurement(Volts)": "iss_sct",
    "STSDetectorMeasurement(Volts)": "sts",
    "IonDetectorAMeasurement(Volts)": "ion_a",
    "IonDetectorBMeasurement(Volts)": "ion_b",
    "PTrakMeasurement(PtPerCC)": "ptrak",
    "DustTrakBMeasurement(MgPerM3)": "dtb",
    "GasPlumeTemperature(DegC)": "plume_t",
}
SENTINELS = {"ptrak": [-999]}       # instrument error codes -> NaN
META = ["test_id", "material", "candidate_materials", "label_verified",
        "phase", "t_rel_heat", "Time", "source_file"]
WINDOWS = [5, 10, 30]               # seconds
BASELINE_SECS = 120                 # baseline = last 120 s of clean air
RATIO_WINDOW = 10                   # ratios use the 10 s rolling mean
NOISE_K = 5                         # ratio only if |denominator| > 5 x clean std
# (numerator, denominator) on baseline-corrected magnitudes
RATIOS = [("iss_sct", "iss_obs"), ("ion_a", "iss_sct"),
          ("ion_b", "iss_sct"), ("ptrak", "iss_sct")]


def baseline_correct(test: pd.DataFrame, chans: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract the median of the last BASELINE_SECS of clean air; return clean std too."""
    clean = test[test["phase"] == "clean"]
    recent = clean[clean["t_rel_heat"] >= -BASELINE_SECS]
    ref = recent if len(recent) >= 10 else clean
    test[chans] = test[chans] - ref[chans].median()
    return test, ref[chans].std()


def rolling_features(test: pd.DataFrame, chans: list[str]) -> pd.DataFrame:
    """Trailing time-based rolling mean/std/slope/diff on a 1 Hz grid.

    The test is reindexed onto a complete 1-second grid (gaps -> NaN, not
    filled), so windows never span a gap as if it were contiguous. Rows that
    only exist on the grid are dropped at the end.
    """
    t = test["Time"].astype(int)
    grid = pd.RangeIndex(t.min(), t.max() + 1)
    x = test.set_index(t)[chans].reindex(grid)
    tt = pd.Series(grid - grid[0], index=grid, dtype=float)

    feats = {}
    for w in WINDOWS:
        mp = max(3, int(0.6 * w))               # need >=60% of the window present
        roll = x.rolling(w, min_periods=mp)
        mean, std = roll.mean(), roll.std()
        # least-squares slope over valid points: cov(t,x)/var(t)
        tv = pd.DataFrame({c: tt.where(x[c].notna()) for c in chans})
        mt = tv.rolling(w, min_periods=mp).mean()
        mtx = (tv * x).rolling(w, min_periods=mp).mean()
        mtt = (tv * tv).rolling(w, min_periods=mp).mean()
        slope = (mtx - mt * mean) / (mtt - mt * mt).replace(0, np.nan)
        diff = x - x.shift(w)                   # change over the last w seconds
        for c in chans:
            feats[f"{c}_mean{w}"] = mean[c]
            feats[f"{c}_std{w}"] = std[c]
            feats[f"{c}_slope{w}"] = slope[c]
            feats[f"{c}_diff{w}"] = diff[c]
    out = pd.DataFrame(feats, index=grid).loc[t.values]
    out.index = test.index
    return out


def ratio_features(feats: pd.DataFrame, noise: pd.Series) -> pd.DataFrame:
    out = {}
    for num, den in RATIOS:
        n = feats[f"{num}_mean{RATIO_WINDOW}"].abs()
        d = feats[f"{den}_mean{RATIO_WINDOW}"].abs()
        floor = NOISE_K * noise.get(den, 0)
        out[f"ratio_{num}_over_{den}"] = (n / d).where(d > max(floor, 1e-9))
    return pd.DataFrame(out, index=feats.index)


def build() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(IN)
    df = raw[META + list(CHANNELS)].rename(columns=CHANNELS)
    chans = list(CHANNELS.values())
    for c, bad in SENTINELS.items():
        df[c] = df[c].mask(df[c].isin(bad))
    df[chans] = df.groupby("test_id")[chans].ffill()   # short sentinel gaps only

    parts = []
    for _, test in df.groupby("test_id", sort=True):
        test = test.sort_values("Time").copy()
        test, noise = baseline_correct(test, chans)
        roll = rolling_features(test, chans)
        ratios = ratio_features(roll, noise)
        parts.append(pd.concat([test, roll, ratios], axis=1))
    out = pd.concat(parts, ignore_index=True)

    base = chans                                  # baseline-corrected instantaneous
    rolled = [c for c in out.columns if any(c.startswith(f"{ch}_") for ch in chans)]
    ratio = [c for c in out.columns if c.startswith("ratio_")]
    cols = {
        "anomaly": base + rolled,                 # no ratios: undefined in clean air
        "classifier": base + rolled + ratio,
        "meta": META,
    }
    return out, cols


def main() -> None:
    out, cols = build()
    out.to_parquet(OUT, index=False)
    OUT_COLS.write_text(json.dumps(cols, indent=2))

    print(f"Saved {OUT.relative_to(ROOT)}: {len(out):,} rows x {out.shape[1]} cols")
    print(f"  anomaly features   : {len(cols['anomaly'])}")
    print(f"  classifier features: {len(cols['classifier'])}")
    nan = out[cols["classifier"]].isna().mean().sort_values(ascending=False)
    print("\nMost-missing features (fraction NaN):")
    print(nan.head(8).round(3).to_string())
    # sanity: baseline should be ~0 in clean air, ratios should be mostly defined in smoke
    clean = out[out["phase"] == "clean"]
    print("\nMedian |value| in clean air (should be ~0):")
    print(clean[list(CHANNELS.values())].abs().median().round(4).to_string())
    smoke = out[out["phase"] == "smoke"]
    print("\nRatio coverage in smoke phase (fraction defined):")
    print(smoke[[c for c in out.columns if c.startswith('ratio_')]].notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
