"""Fast unit tests for the ML core (no raw data, no model training).

Run from the project root:  python -m pytest -q tests
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402
import labels as LB  # noqa: E402
import load as L  # noqa: E402
import repro  # noqa: E402

CHANS = list(F.CHANNELS.values())


# ------------------------------------------------------------------ labels
def test_manifest_is_consistent():
    man = LB.load_manifest()
    assert LB.validate_manifest(man) == []
    assert len(LB.supervised_tests("repo", man)) == 38
    psi = LB.supervised_tests("psi101", man)
    assert len(psi) == 41
    assert (psi[60], psi[103], psi[105]) == ("Silicone", "Silicone", "Teflon")
    assert not set(LB.UNVERIFIED) & set(LB.supervised_tests("repo", man))


def test_manifest_copies_identical():
    def lf(p):
        return p.read_bytes().replace(b"\r\n", b"\n")
    assert lf(LB.MANIFEST) == lf(LB.RESULTS_MANIFEST)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        LB.supervised_tests("all")


def _fake_loader_frame(man):
    rows = []
    for _, r in man.iterrows():
        for t in range(3):
            rows.append({"test_id": r["test_id"], "source_file": r["source_file"],
                         "material": r["repo_material"], "label_verified": r["label_verified"],
                         "Time": t})
    return pd.DataFrame(rows)


def test_loader_check_accepts_matching_frame():
    man = LB.load_manifest()
    out = LB.attach_labels(_fake_loader_frame(man), man)
    assert out["label_psi101"].notna().all()
    assert set(out["label_status"]) == {"verified_folder_and_psi101", "recovered_psi101"}


@pytest.mark.parametrize("mutate", ["drop_test", "wrong_material", "wrong_file"])
def test_loader_check_rejects_divergence(mutate):
    man = LB.load_manifest()
    df = _fake_loader_frame(man)
    if mutate == "drop_test":
        df = df[df["test_id"] != 105]
    elif mutate == "wrong_material":
        df.loc[df["test_id"] == 58, "material"] = "Kapton"
    else:
        df.loc[df["test_id"] == 60, "source_file"] = "Silicone/HKPData_gmt211_4.csv"
    assert LB.check_loader_frame(df, man)
    with pytest.raises(ValueError):
        LB.attach_labels(df, man)


def test_loader_rules_match_manifest():
    """src/load.py duplicate/drop rules are the ones the audit recorded."""
    audit = json.loads((ROOT / "results" / "same_label_audit.json").read_text())
    dup = audit["why_41_vs_38"]["duplicate_files"]
    assert {k: v["kept_folder"] for k, v in dup.items()} == \
        {k: v[0] for k, v in L.DUPLICATES.items()}
    assert [list(x) for x in L.DROP_FILES] == audit["why_41_vs_38"]["dropped_no_heating"]
    assert L.MATERIALS == LB.CLASSES


# ------------------------------------------------------------- causality
def _synthetic_test(n=300, heat_at=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.delete(np.arange(n), [30, 31, 200])
    df = pd.DataFrame(rng.normal(size=(len(t), len(CHANS))), columns=CHANS)
    df["Time"] = t
    df["t_rel_heat"] = t - heat_at
    df["phase"] = np.where(df["t_rel_heat"] < 0, "clean", "heating")
    return df


@pytest.mark.parametrize("cut", [60, 150, 220])
def test_rolling_features_are_causal(cut):
    a = _synthetic_test()
    b = a.copy()
    rng = np.random.default_rng(1)
    after = b["Time"] > cut
    b.loc[after, CHANS] = rng.normal(0, 1e6, (int(after.sum()), len(CHANS)))
    fa, fb = F.rolling_features(a.copy(), CHANS), F.rolling_features(b.copy(), CHANS)
    keep = (a["Time"] <= cut).to_numpy()
    np.testing.assert_array_equal(fa[keep].to_numpy(), fb[keep].to_numpy())


def test_rolling_windows_do_not_bridge_gaps():
    df = _synthetic_test()
    f = F.rolling_features(df.copy(), CHANS)
    # 5 s window needs >= 3 of 5 seconds; second 32 follows the 30-31 gap: only 30-32 -> 28,29,32
    row = f.loc[df.index[df["Time"] == 32][0]]
    assert np.isfinite(row["iss_obs_mean5"])
    # the 1 Hz grid reindexing means the window spans real seconds, not rows
    assert F.WINDOWS == [5, 10, 30]


def test_baseline_uses_only_clean_air():
    df = _synthetic_test()
    df.loc[df["phase"] == "heating", CHANS] += 1e6     # heating must not leak into baseline
    out, _ = F.baseline_correct(df.copy(), CHANS)
    clean = out[out["phase"] == "clean"]
    assert clean[CHANS].abs().median().max() < 5


# ------------------------------------------------------------------ seeds
def test_single_seed_source():
    for f in (ROOT / "src").glob("*.py"):
        if f.name == "repro.py":
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "random_state":
                assert not isinstance(node.value, ast.Constant), f"{f.name}:{node.value.lineno}"


def test_seed_matches_committed_results():
    meta = json.loads((ROOT / "results" / "chained_loto_metrics.json").read_text())["meta"]
    assert repro.CANONICAL_SEED == meta["seed"]


def test_chained_reads_seed_through_ast():
    import evaluate_chained as E
    assert E.AN["IF_PARAMS"]["random_state"] == repro.SEED
    assert E.CL["RF_PARAMS"]["random_state"] == repro.SEED
