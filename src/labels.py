"""SAME test label manifest: one place that defines which tests are supervised.

The manifest (``data/same_label_manifest.csv``, 41 rows) is produced by
``scripts/audit_same_tests.py`` from the NASA PSI-101 (SAME-R) experimental
table. It reconciles two label policies:

* ``repo``   - 38 tests whose material folder is unambiguous; tests 60, 103
               and 105 come from files duplicated across two material folders
               and are excluded (this is what ``src/classifier.py`` trains on).
* ``psi101`` - all 41 tests; 60 -> Silicone, 103 -> Silicone, 105 -> Teflon,
               recovered from PSI-101 by the audit (checks C1-C4).

Stage 1 (clean-air anomaly detection) uses no labels and always uses all 41.

``check_loader_frame`` validates any per-sample frame produced by
``src/load.py`` (or the SAME-R packet stream) against the manifest, so the
legacy loader and the audited label set cannot silently diverge.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "same_label_manifest.csv"
RESULTS_MANIFEST = ROOT / "results" / "same_label_manifest.csv"

CLASSES = ["Cotton Lamp wick", "Kapton", "Pyrell", "Silicone", "Teflon"]
POLICIES = ("repo", "psi101")
N_TESTS = 41
N_REPO = 38
UNVERIFIED = (60, 103, 105)
REQUIRED_COLUMNS = [
    "test_id", "source_file", "repo_material", "label_verified",
    "candidate_materials", "psi101_material", "label", "label_status",
]
STATUSES = {"verified_folder_and_psi101", "recovered_psi101", "excluded_unverified"}


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    """Read and schema-check the label manifest."""
    man = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in man.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    man["test_id"] = man["test_id"].astype(int)
    man["label_verified"] = man["label_verified"].astype(str).str.lower().eq("true")
    return man.sort_values("test_id").reset_index(drop=True)


def validate_manifest(man: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = manifest is internally consistent)."""
    p = []
    if man["test_id"].duplicated().any():
        p.append("duplicate test_id rows")
    if len(man) != N_TESTS:
        p.append(f"expected {N_TESTS} tests, found {len(man)}")
    bad = set(man["label_status"]) - STATUSES
    if bad:
        p.append(f"unknown label_status values {sorted(bad)}")
    lab = man["label"].dropna()
    if not set(lab) <= set(CLASSES):
        p.append(f"labels outside the 5 classes: {sorted(set(lab) - set(CLASSES))}")
    ver = man[man["label_verified"]]
    if len(ver) != N_REPO:
        p.append(f"expected {N_REPO} folder-verified tests, found {len(ver)}")
    if not (ver["label"] == ver["repo_material"]).all():
        p.append("a verified test's label differs from its folder material")
    if not (man["label"].isna() | (man["label"] == man["psi101_material"])).all():
        p.append("a label differs from the PSI-101 material")
    unv = set(man.loc[~man["label_verified"], "test_id"])
    if unv != set(UNVERIFIED):
        p.append(f"unverified tests are {sorted(unv)}, expected {list(UNVERIFIED)}")
    if not (man.loc[~man["label_verified"], "repo_material"] == "unverified").all():
        p.append("unverified tests must carry repo_material == 'unverified'")
    for _, r in man[man["label"].notna()].iterrows():
        if r["label"] not in str(r["candidate_materials"]).split("|"):
            p.append(f"test {r['test_id']}: label {r['label']} not among its candidate folders")
    return p


def supervised_tests(policy: str, man: pd.DataFrame | None = None) -> dict[int, str]:
    """``{test_id: material}`` for the tests a classifier may train/evaluate on."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    man = load_manifest() if man is None else man
    if policy == "repo":
        sel = man[man["label_status"] == "verified_folder_and_psi101"]
    else:
        sel = man[man["label"].notna()]
    return dict(zip(sel["test_id"].astype(int), sel["label"]))


def check_loader_frame(df: pd.DataFrame, man: pd.DataFrame | None = None) -> list[str]:
    """Compare a loader/packet frame (one row per sample) with the manifest.

    ``df`` needs ``test_id``, ``source_file``, ``material`` and
    ``label_verified`` columns, as written by ``src/load.py`` and present in
    the SAME-R wide packets. Returns a list of problems.
    """
    man = load_manifest() if man is None else man
    per_test = (df.groupby("test_id")
                  .agg(source_file=("source_file", "first"),
                       material=("material", "first"),
                       label_verified=("label_verified", "first"),
                       n_source=("source_file", "nunique"))
                  .reset_index())
    p = []
    got, want = set(per_test["test_id"].astype(int)), set(man["test_id"])
    if got != want:
        p.append(f"test sets differ: only in loader {sorted(got - want)}, "
                 f"only in manifest {sorted(want - got)}")
    if (per_test["n_source"] > 1).any():
        p.append("a test_id maps to more than one source file")
    m = per_test.merge(man, on="test_id", suffixes=("", "_man"))
    for col_loader, col_man in (("source_file", "source_file_man"),
                                ("material", "repo_material"),
                                ("label_verified", "label_verified_man")):
        diff = m[m[col_loader].astype(str) != m[col_man].astype(str)]
        for _, r in diff.iterrows():
            p.append(f"test {r['test_id']}: {col_loader} loader={r[col_loader]!r} "
                     f"manifest={r[col_man]!r}")
    return p


def attach_labels(df: pd.DataFrame, man: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate ``df`` against the manifest and add ``label_psi101`` and
    ``label_status`` columns. The legacy ``material`` column is left unchanged
    (it is the ``repo`` policy), so existing downstream results are unaffected."""
    man = load_manifest() if man is None else man
    problems = check_loader_frame(df, man)
    if problems:
        raise ValueError("loader output disagrees with data/same_label_manifest.csv:\n  "
                         + "\n  ".join(problems))
    look = man.set_index("test_id")
    out = df.copy()
    out["label_psi101"] = out["test_id"].map(look["label"])
    out["label_status"] = out["test_id"].map(look["label_status"])
    return out
