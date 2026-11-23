"""Phase 0: load, clean and phase-label the SAME housekeeping CSVs.

Run from the project root:
    python src/load.py

Output: data/processed/all_tests.parquet  (one row per second per test)

The loaded test set is checked against the audited label manifest
(data/same_label_manifest.csv, see src/labels.py): the run aborts if a test id,
source file, folder material or verification flag disagrees. Two columns are
added: ``label_psi101`` (41 labelled tests) and ``label_status``. ``material``
keeps the legacy ``repo`` policy (38 verified tests + "unverified").
"""
from pathlib import Path

import numpy as np
import pandas as pd

import labels

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # raw SAME CSVs: data/raw/<material>/*.csv
OUT = ROOT / "data" / "processed" / "all_tests.parquet"
MATERIALS = ["Cotton Lamp wick", "Kapton", "Pyrell", "Silicone", "Teflon"]

# Files that are byte-identical copies in two material folders.
# key: file stem -> (folder to keep, candidate materials)
DUPLICATES = {
    "HKPData_gmt211_4": ("Pyrell", ["Pyrell", "Silicone"]),
    "HKPData_gmt252_5": ("Silicone", ["Silicone", "Teflon"]),
}
# Tests to drop entirely: (folder, file stem)
DROP_FILES = {("Pyrell", "HKPData_gmt235_1")}  # heater never fired
DROP_COLS = [
    "DustTrakAMeasurement(MgPerM3)",       # bad scaling / all-zero in some files
    "DustTrakAAnalogMeasurement(Volts)",
]
# Columns kept even if constant, because they identify the row.
ID_COLS = ["TimeString", "Time", "TestNumber", "SampleNumber"]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["TimeString"] != "TimeString"]  # repeated header lines
    num = df.columns.drop("TimeString")
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def label_phases(test: pd.DataFrame) -> pd.Series:
    """clean before first heater sample, heating between first/last, smoke after."""
    heat_t = test.loc[test["HeatingInProgress"] == 1, "Time"]
    if heat_t.empty:
        return pd.Series("no_heating", index=test.index)
    t0, t1 = heat_t.min(), heat_t.max()
    phase = np.where(test["Time"] < t0, "clean",
             np.where(test["Time"] <= t1, "heating", "smoke"))
    return pd.Series(phase, index=test.index)


def load_all() -> tuple[pd.DataFrame, list[str]]:
    frames, log = [], []
    for material in MATERIALS:
        for path in sorted((RAW / material).glob("*.csv")):
            stem = path.stem
            if (material, stem) in DROP_FILES:
                log.append(f"dropped {material}/{path.name}: no heating")
                continue
            dup = DUPLICATES.get(stem)
            if dup and material != dup[0]:
                log.append(f"dropped duplicate {material}/{path.name} (kept {dup[0]} copy)")
                continue

            df = read_csv(path)
            df = df[df["TestNumber"] > 0].copy()
            df["source_file"] = f"{material}/{path.name}"
            if dup:
                df["material"] = "unverified"
                df["candidate_materials"] = "|".join(dup[1])
                df["label_verified"] = False
            else:
                df["material"] = material
                df["candidate_materials"] = material
                df["label_verified"] = True
            frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.drop(columns=DROP_COLS)

    # Test id = TestNumber (this also splits gmt252_5 into tests 103 and 105).
    data["test_id"] = data["TestNumber"].astype(int)
    data = (data.sort_values(["test_id", "Time"])
                .drop_duplicates(["test_id", "Time"])
                .reset_index(drop=True))
    data["phase"] = pd.concat([label_phases(g) for _, g in data.groupby("test_id")])

    # Time since heater on (negative during clean air) - convenient for later phases.
    heat_start = (data[data["HeatingInProgress"] == 1]
                  .groupby("test_id")["Time"].min())
    data["t_rel_heat"] = data["Time"] - data["test_id"].map(heat_start)

    # Drop columns that are constant across the whole dataset.
    meta = {"source_file", "material", "candidate_materials", "label_verified",
            "test_id", "phase", "t_rel_heat"}
    const = [c for c in data.columns
             if c not in meta and c not in ID_COLS and data[c].nunique(dropna=False) <= 1]
    data = data.drop(columns=const)
    log.append(f"dropped {len(const)} constant columns: {const}")

    data = labels.attach_labels(data)      # raises if loader and manifest diverge
    log.append(f"label manifest check passed: {data['test_id'].nunique()} tests match "
               f"{labels.MANIFEST.relative_to(ROOT)}")
    return data, log


def main() -> None:
    data, log = load_all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)

    print("\n".join(log))
    print(f"\nSaved {OUT.relative_to(ROOT)}: {len(data):,} rows x {data.shape[1]} cols, "
          f"{data['test_id'].nunique()} tests")
    summary = (data.groupby(["test_id", "material"])["phase"]
                   .value_counts().unstack(fill_value=0)
                   .reindex(columns=["clean", "heating", "smoke"], fill_value=0))
    print("\nRows per phase per test:")
    print(summary.to_string())
    print("\nTests per material:")
    print(data.groupby("material")["test_id"].nunique().to_string())
    nan = data.isna().sum()
    print("\nRemaining NaNs:", nan[nan > 0].to_dict() or "none")


if __name__ == "__main__":
    main()
