"""Audit the SAME material labels, with focus on tests 60, 103 and 105.

Run from the project root:
    python scripts/audit_same_tests.py

Why 41 tests for anomaly detection but 38 for material classification?
-----------------------------------------------------------------------
``src/load.py`` reads 43 NASA HKP CSVs from ``data/raw/<material>/``. It drops
``Pyrell/HKPData_gmt235_1`` (heater never fired) and keeps one copy of two
files that are byte-identical in two material folders:

* ``HKPData_gmt211_4`` (Pyrell and Silicone folders)  -> TestNumber 60
* ``HKPData_gmt252_5`` (Silicone and Teflon folders)  -> TestNumbers 103, 105

40 files give 41 TestNumbers. Tests from the two duplicated files are marked
``label_verified = False`` and ``src/classifier.py`` trains on the other 38.
Stage 1 (clean-air anomaly detection) uses no labels, so it uses all 41.

What this script checks
-----------------------
Every label claim is tested against the official NASA PSI-101 (SAME-R)
experimental table, which lists each SAME Test # with its sample material,
date, inlet velocity, heater resistance ratio and heating duration.

* C1  input files match the SHA-256 values in the repo's FILE_CHECKSUMS.json.
* C2  the PSI-101 table is a valid oracle: its material agrees with the folder
      label for every verified test, and its date is within one day of the
      telemetry GMT date for every test (six verified runs are logged one day
      early in PSI-101; all are reported).
* C3  for each audited test, PSI-101 has exactly one burn row (not an "a"
      aging row) for that Test #, its date matches the telemetry GMT day
      exactly, and its material is one of the candidate folders of the
      duplicated file.
* C4  the audited test is a real fuel run, not a blank or calibration run:
      the heater fired for as long as in verified runs and the smoke sensors
      responded at least as strongly as the weakest verified run.

A label is "recovered" only if C1-C4 all pass. Signature distances to each
material's centroid and the repo classifier's blind prediction are printed
for context but are never used in the ruling (that would be circular).

Outputs
-------
results/same_label_audit.json    full evidence and rulings
results/same_label_manifest.csv  one row per test: label and label status,
                                 read by src/evaluate_chained.py
Exit status is non-zero if any assertion fails.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENSOR = ROOT / "Sensor_Systems_and_Telemetry_Integration"
PACKETS = SENSOR / "ml_lead_handoff" / "SAME_R_WIDE_PACKETS.jsonl"
INVENTORY = SENSOR / "ml_lead_handoff" / "SAME_R_RAW_FILE_INVENTORY.csv"
CHECKSUMS = SENSOR / "FILE_CHECKSUMS.json"
PSI101 = (SENSOR / "integration_update" / "data" / "raw_psi_experimental_tables"
          / "PSI-101_Experimental table_SAME-R.csv")
LOAD_PY = ROOT / "src" / "load.py"
UNVERIFIED_PRED = ROOT / "reports" / "classifier" / "unverified_predictions.csv"
OUT_DIR = ROOT / "results"
OUT_JSON = OUT_DIR / "same_label_audit.json"
OUT_CSV = OUT_DIR / "same_label_manifest.csv"

AUDIT_TESTS = (60, 103, 105)
CAMPAIGN_YEAR = 2010  # SAME-R flew on the ISS in 2010 (GMT labels 2010/DDD)
# PSI-101 sample names -> folder names used by src/load.py
PSI_TO_REPO = {"Lampwick": "Cotton Lamp wick", "Kapton": "Kapton",
               "Pyrell": "Pyrell", "Silicone": "Silicone", "Teflon": "Teflon"}
CHANNELS = ["iss_obs", "iss_sct", "sts", "ion_a", "ion_b", "ptrak", "dtb",
            "plume_t"]
SMOKE_CHANNELS = ["iss_obs", "iss_sct", "ion_a", "ion_b", "dtb"]
BASELINE_S = 120   # same clean-air reference length as src/features.py
SMOKE_SNR = 5.0    # a channel "responds" if |peak delta| > 5 x clean std


# ----------------------------------------------------------------- helpers
def _rel(path: Path) -> str:
    """Path relative to the repo root when possible (for log lines)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def literal_assignments(path: Path, names: set[str]) -> dict:
    """Read module-level constant assignments from a source file without
    importing it, so the audit uses the loader's real rules."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                out[target.id] = ast.literal_eval(node.value)
    missing = names - out.keys()
    assert not missing, f"{path.name} no longer defines {sorted(missing)}"
    return out


def load_packets() -> pd.DataFrame:
    """Flatten the SAME-R wide packets into one row per (test, second)."""
    rows = []
    with PACKETS.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            row = {k: p[k] for k in ("test_id", "source_epoch_s", "time_string",
                                     "material", "candidate_materials",
                                     "label_verified", "heating_in_progress",
                                     "source_file")}
            row.update(p["channels"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df[CHANNELS] = df[CHANNELS].astype(float)
    return df.sort_values(["test_id", "source_epoch_s"]).reset_index(drop=True)


def load_psi101() -> pd.DataFrame:
    """PSI-101 burn rows (Test # is an integer; 'NNa' aging rows are kept
    separately as context)."""
    t = pd.read_csv(PSI101, dtype=str).rename(columns=lambda c: c.strip())
    t["is_burn_row"] = t["SAME Test #"].str.fullmatch(r"\d+")
    t["material"] = t["Sample Type"].str.strip().map(PSI_TO_REPO)
    assert t["material"].notna().all(), "unknown PSI-101 sample type"

    def day_of_year(s: str) -> int:
        d = dt.datetime.strptime(f"{s.strip()}-{CAMPAIGN_YEAR}", "%d-%b-%Y")
        return d.timetuple().tm_yday

    t["gmt_day"] = t["Date"].map(day_of_year)
    return t


# ---------------------------------------------------------------- profiles
def test_profile(test: pd.DataFrame) -> dict:
    """Heater profile, durations and a sensor signature for one test.

    The signature is the peak baseline-corrected change of each channel
    from heater-on to the end of the run, also expressed in units of the
    clean-air standard deviation (SNR). The whole run is used because some
    PSI-101 runs age the smoke for 720 s before it reaches the detectors.
    This is an offline audit, so the heater flag is allowed here; it is
    never used at inference time.
    """
    t = test["source_epoch_s"].to_numpy()
    heat = test["heating_in_progress"].to_numpy() == 1
    on, off = t[heat].min(), t[heat].max()
    t_rel = t - on
    clean = test[t_rel < 0]
    ref = clean[t_rel[t_rel < 0] >= -BASELINE_S]
    ref = ref if len(ref) >= 10 else clean
    resp = test[t_rel >= 0]
    sig, snr = {}, {}
    for c in CHANNELS:
        base, noise = ref[c].median(), ref[c].std()
        delta = (resp[c] - base).dropna()
        peak = float(delta.loc[delta.abs().idxmax()]) if len(delta) else np.nan
        sig[c] = peak
        snr[c] = (abs(peak) / noise if noise and noise > 0 and np.isfinite(peak)
                  else np.nan)
    gmt = test["time_string"].iloc[0]
    return {
        "test_id": int(test["test_id"].iloc[0]),
        "source_file": test["source_file"].iloc[0],
        "repo_material": test["material"].iloc[0],
        "candidate_materials": test["candidate_materials"].iloc[0],
        "label_verified": bool(test["label_verified"].iloc[0]),
        "gmt_first_sample": gmt,
        "gmt_day": int(gmt.split("/")[1]),
        "samples": int(len(test)),
        "run_span_s": int(t[-1] - t[0] + 1),
        "clean_air_s": int(on - t[0]),
        "heater_on_samples": int(heat.sum()),
        "heater_span_s": int(off - on + 1),
        "post_heater_s": int(t[-1] - off),
        "smoke_channels_responding": int(sum(snr[c] > SMOKE_SNR
                                             for c in SMOKE_CHANNELS
                                             if np.isfinite(snr[c]))),
        "signature_peak_delta": sig,
        "signature_snr": snr,
    }


def signature_distances(prof: pd.DataFrame, tid: int) -> dict:
    """Standardised distance from one test's signature to each material's
    centroid (verified tests only). Context only; not used in the ruling."""
    feats = [f"snr_{c}" for c in CHANNELS]
    X = prof.set_index("test_id")[feats].apply(np.log1p)
    ver = prof.set_index("test_id")["label_verified"]
    mat = prof.set_index("test_id")["repo_material"]
    mu, sd = X[ver].mean(), X[ver].std().replace(0, 1)
    Z = (X - mu) / sd
    out = {}
    for m in sorted(mat[ver].unique()):
        centroid = Z[ver & (mat == m)].mean()
        diff = (Z.loc[tid] - centroid).dropna()
        out[m] = round(float(np.sqrt((diff ** 2).mean())), 3)
    return out


# -------------------------------------------------------------------- main
def main(out_dir: Path | None = None) -> int:
    """Run the audit. ``out_dir`` (default ``results/``) receives the JSON and
    CSV; ``scripts/verify_repo.py`` passes a temporary directory so the
    committed files are compared, never overwritten."""
    global OUT_DIR, OUT_JSON, OUT_CSV
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
        OUT_JSON = OUT_DIR / "same_label_audit.json"
        OUT_CSV = OUT_DIR / "same_label_manifest.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> bool:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)
        return bool(cond)

    # ---- C1 provenance ----------------------------------------------------
    print("C1  Input provenance")
    manifest = json.loads(CHECKSUMS.read_text(encoding="utf-8"))["files"]
    inputs = {}
    for path in (PACKETS, PSI101, INVENTORY):
        rel = path.relative_to(SENSOR).as_posix()
        digest = sha256(path)
        inputs[rel] = digest
        check(manifest.get(rel) == digest, f"sha256 matches FILE_CHECKSUMS.json: {rel}")

    rules = literal_assignments(LOAD_PY, {"MATERIALS", "DUPLICATES", "DROP_FILES"})
    materials = rules["MATERIALS"]
    check(sorted(PSI_TO_REPO.values()) == sorted(materials),
          f"repo material classes are {materials}")

    # ---- why 41 vs 38 -----------------------------------------------------
    inv = pd.read_csv(INVENTORY)
    inv["stem"] = inv["file"].str.removesuffix(".csv")
    dup_groups = inv.groupby("sha256").filter(lambda g: len(g) > 1)
    dup_info = {}
    for stem, (kept, cands) in rules["DUPLICATES"].items():
        rows = inv[inv["stem"] == stem]
        dup_info[stem] = {"folders": sorted(rows["material_folder"]),
                          "identical_sha256": rows["sha256"].nunique() == 1,
                          "kept_folder": kept, "candidates": cands}
        check(rows["sha256"].nunique() == 1 and sorted(rows["material_folder"]) == sorted(cands),
              f"{stem}: byte-identical copies in {sorted(cands)}")
    check(set(dup_groups["stem"]) == set(rules["DUPLICATES"]),
          "no other duplicated files in the raw inventory")

    df = load_packets()
    prof = pd.DataFrame([test_profile(g) for _, g in df.groupby("test_id")])
    for c in CHANNELS:
        prof[f"snr_{c}"] = prof["signature_snr"].map(lambda d, c=c: d[c])
    n_files = len(inv) - len(rules["DROP_FILES"]) - len(rules["DUPLICATES"])
    file_tests = prof.groupby("source_file")["test_id"].apply(sorted).to_dict()
    multi = {f: t for f, t in file_tests.items() if len(t) > 1}
    unverified = sorted(prof.loc[~prof["label_verified"], "test_id"])
    print("\nWhy 41 vs 38")
    print(f"  {len(inv)} raw CSVs - {len(rules['DROP_FILES'])} dropped (no heating) - "
          f"{len(rules['DUPLICATES'])} duplicate copies = {n_files} files")
    print(f"  {n_files} files -> {prof['test_id'].nunique()} TestNumbers "
          f"(multi-test files: {multi})")
    print(f"  unverified (duplicated-file) tests: {unverified} -> "
          f"{int(prof['label_verified'].sum())} labelled tests for the classifier")
    check(prof["test_id"].nunique() == 41 and n_files == 40, "41 tests from 40 files")
    check(unverified == list(AUDIT_TESTS), f"unverified tests are exactly {list(AUDIT_TESTS)}")

    # ---- C2 PSI-101 as an oracle ---------------------------------------
    print("\nC2  PSI-101 table validated against the 38 verified tests")
    psi = load_psi101()
    burns = psi[psi["is_burn_row"]].copy()
    burns["test_no"] = burns["SAME Test #"].astype(int)
    counts = burns["test_no"].value_counts()
    ambiguous = sorted(int(k) for k in counts[counts > 1].index)
    prof = prof.merge(
        burns[~burns["test_no"].isin(ambiguous)].rename(columns={
            "test_no": "test_id", "material": "psi101_material",
            "Description": "psi101_description", "Date": "psi101_date",
            "gmt_day": "psi101_gmt_day",
            "Inlet velocity (cm/s)": "psi101_inlet_velocity_cm_s",
            "Rt/R0": "psi101_rt_r0", "Duration (sec)": "psi101_duration_s",
            "Primary aging time (sec)": "psi101_primary_aging_s",
        })[["test_id", "psi101_material", "psi101_description", "psi101_date",
            "psi101_gmt_day", "psi101_inlet_velocity_cm_s", "psi101_rt_r0",
            "psi101_duration_s", "psi101_primary_aging_s"]],
        on="test_id", how="left")
    ver = prof["label_verified"]
    agree = (prof["psi101_material"] == prof["repo_material"])[ver]
    day_off = prof["gmt_day"] - prof["psi101_gmt_day"]
    date_ok = day_off.abs() <= 1
    offsets = prof.loc[day_off != 0, ["test_id", "label_verified", "psi101_date",
                                      "gmt_first_sample"]]
    print(f"  PSI-101 burn rows: {len(burns)}; Test #s listed twice (excluded): {ambiguous}")
    check(not set(ambiguous) & set(prof["test_id"]), "no repo test maps to an ambiguous Test #")
    check(prof["psi101_material"].notna().all(), "every repo test has a PSI-101 row")
    check(bool(agree.all()),
          f"material agrees on {int(agree.sum())}/{int(ver.sum())} verified tests")
    check(bool(date_ok.all()),
          f"PSI-101 date within 1 day of GMT day on {int(date_ok.sum())}/{len(prof)} tests "
          f"(exact on {int((day_off == 0).sum())})")
    check(bool(offsets["label_verified"].all()),
          f"non-exact dates are only on verified tests: {offsets['test_id'].tolist()}")
    aged = prof.loc[pd.to_numeric(prof["psi101_primary_aging_s"], errors="coerce") > 600,
                    "test_id"].tolist()
    print(f"  note: tests with >600 s primary smoke aging (smoke arrives after the "
          f"classifier's 600 s horizon): {aged}")

    # ---- C3/C4 per audited test --------------------------------------------
    print("\nC3/C4  Audited tests")
    vref = prof[ver]
    heat_lo, heat_hi = vref["heater_on_samples"].min(), vref["heater_on_samples"].max()
    smoke_lo = vref["smoke_channels_responding"].min()
    hints = (pd.read_csv(UNVERIFIED_PRED).set_index("test_id")
             if UNVERIFIED_PRED.exists() else pd.DataFrame())
    rulings = {}
    for tid in AUDIT_TESTS:
        r = prof.set_index("test_id").loc[tid]
        cands = r["candidate_materials"].split("|")
        print(f"  Test {tid} ({r['source_file']}):")
        ok = [
            check(pd.notna(r["psi101_material"]),
                  f"PSI-101 has one burn row: {r['psi101_material']}, "
                  f"'{r['psi101_description']}', {r['psi101_date']}"),
            check(r["psi101_gmt_day"] == r["gmt_day"],
                  f"PSI-101 date {r['psi101_date']} = GMT day {r['psi101_gmt_day']}; "
                  f"telemetry starts {r['gmt_first_sample']}"),
            check(r["psi101_material"] in cands,
                  f"{r['psi101_material']} is one of the duplicate-file folders {cands}"),
            check(heat_lo <= r["heater_on_samples"] <= heat_hi,
                  f"heater on {r['heater_on_samples']} s (verified runs: {heat_lo}-{heat_hi} s)"),
            check(r["smoke_channels_responding"] >= smoke_lo,
                  f"{r['smoke_channels_responding']}/{len(SMOKE_CHANNELS)} smoke channels "
                  f"> {SMOKE_SNR:g} sigma (weakest verified run: {smoke_lo})"),
        ]
        recovered = all(ok) and not failures
        hint = (hints.loc[tid, "predicted"] if tid in hints.index else None)
        rulings[tid] = {
            "ruling": "recovered_label" if recovered else "keep_excluded",
            "label": r["psi101_material"] if recovered else None,
            "evidence": {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in r.drop(["signature_peak_delta", "signature_snr"]).items()
                         if not str(k).startswith("snr_")},
            "signature_peak_delta": r["signature_peak_delta"],
            "signature_snr": r["signature_snr"],
            "context_only": {
                "signature_distance_to_material_centroids": signature_distances(prof, tid),
                "repo_classifier_blind_prediction": hint,
            },
        }

    # ---- verification table ---------------------------------------------
    cols = ["test_id", "source_file", "candidate_materials", "gmt_day", "psi101_date",
            "psi101_material", "psi101_description", "psi101_inlet_velocity_cm_s",
            "psi101_rt_r0", "psi101_primary_aging_s", "clean_air_s", "heater_on_samples",
            "run_span_s", "smoke_channels_responding"]
    table = prof.set_index("test_id").loc[list(AUDIT_TESTS)].reset_index()[cols]
    table["context: nearest signature"] = [
        min(d := rulings[t]["context_only"]["signature_distance_to_material_centroids"],
            key=d.get) for t in AUDIT_TESTS]
    table["context: repo classifier"] = [
        rulings[t]["context_only"]["repo_classifier_blind_prediction"] for t in AUDIT_TESTS]
    table["RULING"] = [f"{rulings[t]['ruling']}: {rulings[t]['label']}" for t in AUDIT_TESTS]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\nVerification table\n" + table.T.to_string(header=False))

    centroids = (vref.groupby("repo_material")[
        ["clean_air_s", "heater_on_samples", "run_span_s", "smoke_channels_responding"]
        + [f"snr_{c}" for c in CHANNELS]].median().round(1))
    with pd.option_context("display.width", 250):
        print("\nVerified-run medians per material (reference profile)\n"
              + centroids.to_string())

    # ---- manifest + split guards -------------------------------------------
    man = prof[["test_id", "source_file", "repo_material", "label_verified",
                "candidate_materials", "psi101_material", "psi101_description",
                "psi101_date", "psi101_inlet_velocity_cm_s", "psi101_rt_r0",
                "psi101_primary_aging_s"]].copy()
    man["label"] = np.where(man["label_verified"], man["repo_material"],
                            man["test_id"].map(lambda t: rulings.get(t, {}).get("label")))
    man["label_status"] = np.where(
        man["label_verified"], "verified_folder_and_psi101",
        np.where(man["label"].notna(), "recovered_psi101", "excluded_unverified"))
    print("\nSplit guards")
    repo_set = set(man.loc[man["label_status"] == "verified_folder_and_psi101", "test_id"])
    psi_set = set(man.loc[man["label"].notna(), "test_id"])
    check(len(repo_set) == 38 and not repo_set & set(AUDIT_TESTS),
          "policy 'repo': 38 supervised tests, none of 60/103/105")
    check(bool((man.loc[man["test_id"].isin(psi_set), "label"]
                == man.loc[man["test_id"].isin(psi_set), "psi101_material"]).all()),
          f"policy 'psi101': {len(psi_set)} supervised tests, every label equals PSI-101")
    check(set(man["label"].dropna()) <= set(materials), "labels are within the 5 repo classes")
    man.to_csv(OUT_CSV, index=False)

    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs_sha256": inputs,
        "material_classes": materials,
        "why_41_vs_38": {
            "raw_csv_files": int(len(inv)),
            "dropped_no_heating": [list(x) for x in rules["DROP_FILES"]],
            "duplicate_files": dup_info,
            "files_used": int(n_files),
            "multi_test_files": {k: [int(x) for x in v] for k, v in multi.items()},
            "tests_total": int(prof["test_id"].nunique()),
            "tests_label_verified": int(ver.sum()),
            "tests_unverified": [int(x) for x in unverified],
        },
        "psi101_oracle_validation": {
            "burn_rows": int(len(burns)),
            "ambiguous_test_numbers": ambiguous,
            "material_agreement_verified": f"{int(agree.sum())}/{int(ver.sum())}",
            "date_within_1_day": f"{int(date_ok.sum())}/{len(prof)}",
            "date_exact": f"{int((day_off == 0).sum())}/{len(prof)}",
            "date_offsets": offsets.to_dict(orient="records"),
            "tests_with_primary_aging_over_600_s": [int(x) for x in aged],
        },
        "rulings": {str(k): v for k, v in rulings.items()},
        "label_policies": {
            "repo": sorted(int(x) for x in repo_set),
            "psi101": sorted(int(x) for x in psi_set),
        },
        "assertion_failures": failures,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {_rel(OUT_JSON)} and {_rel(OUT_CSV)}")
    if failures:
        print(f"\n{len(failures)} assertion(s) FAILED")
        return 1
    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Audit SAME material labels against NASA PSI-101.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write the JSON/CSV here instead of results/ (used by verify_repo.py)")
    sys.exit(main(ap.parse_args().out_dir))
