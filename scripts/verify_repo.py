#!/usr/bin/env python3
"""Fast end-to-end integrity check for the Flame in Freefall repository.

Run from anywhere:
    python scripts/verify_repo.py            # fast checks (~10 s, no model training)
    python scripts/verify_repo.py --full     # also re-run every pipeline stage (hours)
    python scripts/verify_repo.py --full --dry-run   # print the --full commands only

The fast mode never trains a model and never writes inside the repository.
It checks, in order:

  env        Python >= 3.11 and the core packages; versions vs. the recorded
             reference environment of the committed chained-LOTO run.
  compile    every .py file in src/, scripts/ and dashboard/ byte-compiles;
             the core src/ modules import.
  seed       one seed source (src/repro.py); no hard-coded seeds left in src/;
             active seed == seed recorded in results/chained_loto_metrics.json.
  data       SAME-R wide-packet schema (41 tests, 8 channels, 58,386 packets),
             SHA-256 against FILE_CHECKSUMS.json and the chained-run inputs;
             feature_columns.json and config/sensor_map.yaml agree with
             src/features.py.
  labels     data/ and results/ manifests are identical and internally
             consistent; the packet stream matches the manifest test-by-test.
  audit      scripts/audit_same_tests.py re-runs into a temp dir and its output
             equals the committed results/ files (timestamp excepted).
  chained    results/chained_loto_*: schema, 41/41 folds, label policies,
             Stage 1 / chained metrics recomputed from the per-test CSV,
             latency percentiles recomputed from the latency CSV, leakage
             self-checks passed, summary.md re-rendered byte-identically.
  sensors    reports/sensor_sensitivity/: all files and 300 DPI figures exist;
             all 255 subset rows recomputed from per_test_*.csv.gz; the
             Markdown table agrees with the CSV.
  causality  trailing-window features at time t are bit-identical when every
             sample after t is replaced by noise (no look-ahead leakage).
  tests      root unit tests (tests/) and the sensor-workspace suite.

Exit status: 0 if no check FAILs (WARNs are allowed), 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib
import json
import os
import platform
import py_compile
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SENSOR = ROOT / "Sensor_Systems_and_Telemetry_Integration"
PACKETS = SENSOR / "ml_lead_handoff" / "SAME_R_WIDE_PACKETS.jsonl"
CHECKSUMS = SENSOR / "FILE_CHECKSUMS.json"
RESULTS = ROOT / "results"
SENS = ROOT / "reports" / "sensor_sensitivity"
CHANNELS = ["iss_obs", "iss_sct", "sts", "ion_a", "ion_b", "ptrak", "dtb", "plume_t"]
N_TESTS, N_PACKETS = 41, 58386

# Reference environment recorded in results/chained_loto_metrics.json (meta).
REFERENCE_ENV = {"python": "3.11", "numpy": "2.4.6", "pandas": "3.0.3", "scikit-learn": "1.9.0"}
CORE = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
        "scipy": "scipy", "matplotlib": "matplotlib", "joblib": "joblib"}
OPTIONAL = {"lightgbm": "lightgbm (src/classifier.py)", "shap": "shap (src/classifier.py)",
            "pyarrow": "pyarrow (parquet I/O in src/load.py, features.py)",
            "streamlit": "streamlit (dashboard/app.py)", "boto3": "boto3 (src/fetch_psi.py)",
            "yaml": "pyyaml (sensor_map.yaml check)"}

sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------- harness
class Check:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def ok(self, name, msg):
        self.rows.append(("PASS", name, msg))

    def warn(self, name, msg):
        self.rows.append(("WARN", name, msg))

    def fail(self, name, msg):
        self.rows.append(("FAIL", name, msg))

    def expect(self, cond, name, msg, warn_only=False):
        (self.ok if cond else (self.warn if warn_only else self.fail))(name, msg)
        return cond


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------- checks
def check_env(c: Check):
    v = sys.version_info
    c.expect(v >= (3, 11), "env.python", f"Python {platform.python_version()} (need >= 3.11; "
             f"reference run used {REFERENCE_ENV['python']}.x)")
    from importlib.metadata import version, PackageNotFoundError
    for dist, mod in CORE.items():
        try:
            importlib.import_module(mod)
            ver = version(dist)
        except (ImportError, PackageNotFoundError) as e:
            c.fail(f"env.{dist}", f"not importable: {e}")
            continue
        ref = REFERENCE_ENV.get(dist)
        if ref and ver != ref:
            c.warn(f"env.{dist}", f"{ver} (reference run used {ref}; tree-model outputs may "
                   f"differ slightly on a full re-run)")
        else:
            c.ok(f"env.{dist}", ver + (" (matches reference run)" if ref else ""))
    missing = []
    for mod, what in OPTIONAL.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(what)
    if missing:
        c.warn("env.optional", "not installed (only needed for --full / dashboard): "
               + "; ".join(missing))
    else:
        c.ok("env.optional", "all optional packages importable")


def check_compile(c: Check):
    files = sorted(p for d in ("src", "scripts", "dashboard") for p in (ROOT / d).rglob("*.py"))
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            try:
                py_compile.compile(str(f), cfile=str(Path(tmp) / "x.pyc"), doraise=True)
            except py_compile.PyCompileError as e:
                bad.append(f"{rel(f)}: {e.msg.strip().splitlines()[-1]}")
    c.expect(not bad, "compile.all", f"{len(files)} files byte-compile"
             if not bad else "; ".join(bad))
    c.expect((SRC / "__init__.py").exists(), "compile.package", "src/__init__.py present")
    # core modules that do not need optional deps; classifier.py needs lightgbm+shap
    mods = ["repro", "labels", "load", "features", "anomaly", "evaluate_chained",
            "sensor_sensitivity"]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{m}: {type(e).__name__}: {e}")
    c.expect(not failed, "compile.imports",
             f"imported {', '.join(mods)}" if not failed else "; ".join(failed))
    try:
        importlib.import_module("classifier")
        c.ok("compile.classifier", "src/classifier.py imports")
    except ImportError as e:
        c.warn("compile.classifier", f"src/classifier.py not importable here ({e}); "
               "byte-compile passed")


def check_seed(c: Check):
    import repro
    hard = []
    for f in sorted(SRC.glob("*.py")):
        if f.name == "repro.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in ("random_state", "seed") \
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                hard.append(f"{f.name}:{node.value.lineno} {node.arg}={node.value.value}")
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in (
                    "default_rng", "RandomState", "seed") and node.args \
                    and isinstance(node.args[0], ast.Constant):
                hard.append(f"{f.name}:{node.lineno} {node.func.attr}({node.args[0].value})")
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SEED"
                                                    for t in node.targets):
                hard.append(f"{f.name}:{node.lineno} defines its own SEED")
    c.expect(not hard, "seed.single_source",
             f"all seeds come from src/repro.py (SEED={repro.SEED})" if not hard
             else "hard-coded: " + "; ".join(hard))
    meta = json.loads((RESULTS / "chained_loto_metrics.json").read_text())["meta"]
    c.expect(meta["seed"] == repro.SEED, "seed.matches_results",
             f"active SEED={repro.SEED}, committed results used seed {meta['seed']}"
             + ("" if meta["seed"] == repro.SEED else
                " -> committed numbers will NOT be reproduced; re-run with --full"),
             warn_only=True)


def load_packets_light():
    """Parse the packet stream once (ids, labels and channel values)."""
    import pandas as pd
    rows, keys_bad, ch_bad = [], 0, 0
    need = {"test_id", "source_file", "material", "label_verified", "channels",
            "heating_in_progress", "elapsed_test_time_s", "timestamp_utc"}
    with PACKETS.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if not need <= d.keys():
                keys_bad += 1
                continue
            ch = d["channels"]
            if set(ch) != set(CHANNELS):
                ch_bad += 1
            rows.append((d["test_id"], d["source_file"], d["material"], d["label_verified"],
                         d["heating_in_progress"]))
    df = pd.DataFrame(rows, columns=["test_id", "source_file", "material", "label_verified",
                                     "heat"])
    return df, keys_bad, ch_bad


def check_data(c: Check, cache: dict):
    if not c.expect(PACKETS.exists(), "data.packets", rel(PACKETS)):
        return
    df, keys_bad, ch_bad = load_packets_light()
    cache["packets"] = df
    c.expect(len(df) == N_PACKETS and keys_bad == 0, "data.packets.schema",
             f"{len(df):,} packets, {keys_bad} with missing fields (expect {N_PACKETS:,}, 0)")
    c.expect(ch_bad == 0, "data.packets.channels", f"8 channels {CHANNELS} in every packet"
             if ch_bad == 0 else f"{ch_bad} packets with a different channel set")
    n = df["test_id"].nunique()
    heated = df.groupby("test_id")["heat"].max().eq(1).sum()
    c.expect(n == N_TESTS and heated == N_TESTS, "data.packets.tests",
             f"{n} tests, {heated} with heater-on samples (expect {N_TESTS})")
    digest = sha256(PACKETS)
    sums = json.loads(CHECKSUMS.read_text())
    flat = sums.get("files", sums) if isinstance(sums, dict) else {}
    want = next((v for k, v in flat.items() if str(k).endswith("SAME_R_WIDE_PACKETS.jsonl")), None)
    if isinstance(want, dict):
        want = want.get("sha256")
    if want:
        c.expect(digest == want, "data.packets.sha256", "matches FILE_CHECKSUMS.json")
    else:
        c.warn("data.packets.sha256", "no entry in FILE_CHECKSUMS.json")
    meta = json.loads((RESULTS / "chained_loto_metrics.json").read_text())["meta"]
    c.expect(meta["inputs_sha256"]["SAME_R_WIDE_PACKETS.jsonl"] == digest,
             "data.packets.provenance", "identical to the input of the committed chained run")

    import features as F
    cols = json.loads((ROOT / "data" / "processed" / "feature_columns.json").read_text())
    chans = list(F.CHANNELS.values())
    c.expect(chans == CHANNELS, "data.features.channels", "src/features.py channel order")
    n_roll = len(chans) * len(F.WINDOWS) * 4
    exp_anom = len(chans) + n_roll
    c.expect(len(cols["anomaly"]) == exp_anom and len(cols["classifier"]) == exp_anom
             + len(F.RATIOS), "data.features.columns",
             f"feature_columns.json: {len(cols['anomaly'])} anomaly / "
             f"{len(cols['classifier'])} classifier (expect {exp_anom} / {exp_anom + len(F.RATIOS)})")
    try:
        import yaml
        sm = yaml.safe_load((ROOT / "config" / "sensor_map.yaml").read_text(encoding="utf-8"))
        raw = {v["raw_column"]: k for k, v in sm["channels"].items() if "raw_column" in v}
        c.expect(all(raw.get(k) == v for k, v in F.CHANNELS.items()), "data.sensor_map",
                 "config/sensor_map.yaml raw columns match src/features.py")
    except ImportError:
        c.warn("data.sensor_map", "pyyaml not installed; skipped")


def check_labels(c: Check, cache: dict):
    import labels as LB
    a, b = ROOT / "data" / "same_label_manifest.csv", RESULTS / "same_label_manifest.csv"
    c.expect(a.read_bytes() == b.read_bytes(), "labels.copies",
             "data/ and results/ manifests are byte-identical")
    man = LB.load_manifest()
    probs = LB.validate_manifest(man)
    c.expect(not probs, "labels.manifest", "41 tests, 38 folder-verified, 60/103/105 recovered "
             "from PSI-101, all labels in 5 classes" if not probs else "; ".join(probs))
    meta = json.loads((RESULTS / "chained_loto_metrics.json").read_text())["meta"]
    for pol in LB.POLICIES:
        got = sorted(LB.supervised_tests(pol, man))
        c.expect(got == meta["label_policies"][pol], f"labels.policy.{pol}",
                 f"{len(got)} supervised tests == chained-run policy")
    want = meta["inputs_sha256"]["same_label_manifest.csv"]
    raw = b.read_bytes()
    lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    crlf = hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
    if want == sha256(b):
        c.ok("labels.provenance", "manifest byte-identical to the input of the committed chained run")
    else:
        c.expect(want in (lf, crlf), "labels.provenance",
                 "manifest content identical to the chained-run input (hash recorded on a "
                 "CRLF checkout; .gitattributes stores LF)" if want in (lf, crlf)
                 else "manifest differs from the input of the committed chained run")
    if "packets" in cache:
        probs = LB.check_loader_frame(cache["packets"], man)
        c.expect(not probs, "labels.loader", "packet stream matches the manifest for every test "
                 "(id, source file, folder material, verified flag)" if not probs
                 else "; ".join(probs[:5]))


def check_audit(c: Check):
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_same_tests.py"),
                            "--out-dir", tmp], capture_output=True, text=True, cwd=ROOT)
        if not c.expect(p.returncode == 0, "audit.run", "scripts/audit_same_tests.py: all "
                        "assertions passed" if p.returncode == 0 else
                        (p.stdout + p.stderr)[-600:]):
            return
        new_csv = (Path(tmp) / "same_label_manifest.csv").read_bytes()
        c.expect(new_csv == (RESULTS / "same_label_manifest.csv").read_bytes(),
                 "audit.manifest", "regenerated manifest == committed")
        j_new = json.loads((Path(tmp) / "same_label_audit.json").read_text())
        j_old = json.loads((RESULTS / "same_label_audit.json").read_text())
        j_new.pop("generated_utc", None), j_old.pop("generated_utc", None)
        c.expect(j_new == j_old, "audit.report", "regenerated audit JSON == committed "
                 "(timestamp excluded)")
        c.expect(not j_old.get("assertion_failures"), "audit.assertions",
                 "committed audit has no assertion failures")


def check_chained(c: Check):
    import numpy as np
    import pandas as pd
    files = ["chained_loto_metrics.json", "chained_loto_summary.md",
             "chained_loto_per_test.csv", "chained_loto_latency_ms.csv"]
    miss = [f for f in files if not (RESULTS / f).exists()]
    if not c.expect(not miss, "chained.files", "4 result files present" if not miss
                    else f"missing {miss}"):
        return
    m = json.loads((RESULTS / "chained_loto_metrics.json").read_text())
    keys = {"meta", "leakage_checks", "stage1", "oracle_stage2", "chained", "latency_ms",
            "repo_reference"}
    c.expect(keys <= m.keys(), "chained.schema", "top-level keys present")
    meta = m["meta"]
    c.expect(meta["loto_complete"] and meta["folds_evaluated"] == N_TESTS, "chained.folds",
             f"{meta['folds_evaluated']} LOTO folds, complete={meta['loto_complete']}")
    exp_s1 = {f"{b}/n{n}" for b in ("arming", "heater_oracle") for n in (1, 2, 3)}
    exp_ch = {f"{b}/{p}/n{n}" for b in ("arming", "heater_oracle")
              for p in ("repo", "psi101") for n in (1, 2, 3)}
    c.expect(set(m["stage1"]) == exp_s1 and set(m["chained"]) == exp_ch,
             "chained.configs", "6 Stage 1 and 12 chained configurations")

    pt = pd.read_csv(RESULTS / "chained_loto_per_test.csv")
    c.expect(len(pt) == 2 * 3 * 2 * N_TESTS, "chained.per_test.rows", f"{len(pt)} rows "
             f"(2 baselines x 3 persistence x 2 policies x {N_TESTS})")
    # Stage 1 uses no labels, so its per-test columns must be identical under both policies
    s1 = ["alert_s", "fa_events", "clean_armed_s", "trigger_sensor", "iss_alert_s",
          "iss_fa_events"]
    k = ["baseline", "n_persist", "test_id"]
    a = pt[pt.policy == "repo"].set_index(k)[s1].sort_index()
    b = pt[pt.policy == "psi101"].set_index(k)[s1].sort_index()
    c.expect(a.index.equals(b.index) and a.equals(b), "chained.stage1_label_free",
             "Stage 1 per-test results identical under 'repo' and 'psi101' label policies")
    bad = []
    for key, v in m["stage1"].items():
        b, n = key.split("/n")
        g = pt[(pt.baseline == b) & (pt.n_persist == int(n)) & (pt.policy == "psi101")]
        s, d = v["per_sensor_iforest"], g["alert_s"].dropna()
        far = g["fa_events"].sum() / (g["clean_armed_s"].sum() / 3600)
        if not (s["detected"] == len(d) and np.isclose(s["latency_median_s"], d.median())
                and np.isclose(s["latency_p90_s"], d.quantile(0.9))
                and np.isclose(s["far_per_clean_hour"], far)
                and np.isclose(v["iss_scatter_recalibrated"]["latency_median_s"],
                               g["iss_alert_s"].median())):
            bad.append(key)
    for key, v in m["chained"].items():
        b, pol, n = key.split("/")
        g = pt[(pt.baseline == b) & (pt.n_persist == int(n[1:])) & (pt.policy == pol)
               & pt.labelled]
        for col, k in (("pred_steady_state", "test_level_steady_state"),
                       ("pred_first_alert", "test_level_first_alert"),
                       ("pred_until_600s", "test_level_until_600s")):
            if int((g[col] == g["label"]).sum()) != v[k]["correct"]:
                bad.append(f"{key}:{k}")
    c.expect(not bad, "chained.recompute", "Stage 1 detection/latency/FAR and chained "
             "accuracies recomputed from per-test CSV match the JSON" if not bad
             else f"mismatch in {bad[:6]}")

    lat = pd.read_csv(RESULTS / "chained_loto_latency_ms.csv")
    sel = {"all_armed_ticks": lat, "stage1_only": lat[~lat.stage2_ran],
           "stage1_plus_stage2": lat[lat.stage2_ran]}
    lbad = [k for k, g in sel.items()
            if not (m["latency_ms"][k]["n_ticks"] == len(g)
                    and np.isclose(m["latency_ms"][k]["p50"], g.latency_ms.quantile(0.5))
                    and np.isclose(m["latency_ms"][k]["max"], g.latency_ms.max()))]
    c.expect(not lbad, "chained.latency", "tick counts, P50 and max recomputed from "
             "latency CSV" if not lbad else f"mismatch in {lbad}")
    p99 = m["latency_ms"]["all_armed_ticks"]["p99"]
    c.expect(p99 < m["latency_ms"]["budget_ms_at_1hz"], "chained.realtime",
             f"P99 per-tick latency {p99:.0f} ms < 1000 ms budget at 1 Hz")

    lk = m["leakage_checks"]
    sv = lk["streaming_vs_batch"]
    ok = (len(lk["future_perturbation_tests_bit_identical"]) >= 1
          and all(r["alarm_agreement"] == 1.0 and r["max_abs_fused_score_diff"] == 0.0
                  and r["stage2_argmax_agreement"] == 1.0 for r in sv))
    c.expect(ok, "chained.leakage", f"future-perturbation bit-identical on tests "
             f"{lk['future_perturbation_tests_bit_identical']}; streaming == batch on "
             f"{[r['test_id'] for r in sv]}")
    import evaluate_chained as E
    md = E.render_markdown(m)
    c.expect(md == (RESULTS / "chained_loto_summary.md").read_text(encoding="utf-8"),
             "chained.summary_md", "summary.md re-rendered from JSON is byte-identical")


def check_sensors(c: Check):
    import numpy as np
    import pandas as pd
    need = [f"{s}_{sp}.{e}" for sp in ("loto", "gkf8") for s, e in (
        ("all_subsets", "csv"), ("best_subset_per_size", "csv"), ("classifier", "csv"),
        ("minimal_subset", "json"), ("per_test", "csv.gz"), ("reference_rows", "csv"),
        ("sensor_table", "csv"), ("sensor_table", "md"))] + [
        "greedy_selection_loto.csv", "README.md"]
    figs = [f"figures/{n}.{e}" for n in ("latency_per_sensor_loto", "latency_per_sensor_gkf8",
                                         "greedy_selection_loto") for e in ("png", "pdf")]
    miss = [f for f in need + figs if not (SENS / f).exists()]
    c.expect(not miss, "sensors.files", f"{len(need) + len(figs)} files present"
             if not miss else f"missing {miss}")
    try:
        from PIL import Image
        dpis = {f: Image.open(SENS / f).info.get("dpi", (0, 0))[0] for f in figs
                if f.endswith(".png") and (SENS / f).exists()}
        c.expect(all(round(d) >= 300 for d in dpis.values()), "sensors.dpi",
                 "PNG figures at " + ", ".join(f"{round(d)}" for d in dpis.values()) + " DPI")
    except ImportError:
        c.warn("sensors.dpi", "Pillow not installed; DPI not checked")
    with gzip.open(SENS / "per_test_loto.csv.gz", "rb") as fh:
        head = fh.readline()
    c.expect(b"alert_s" in head, "sensors.per_test.schema", "per_test_loto.csv.gz readable")

    import sensor_sensitivity as S
    bad = []
    for split in ("loto", "gkf8"):
        res = pd.read_csv(SENS / f"per_test_{split}.csv.gz")
        summ = (res.groupby(["config", "channels", "n_channels"])
                   .apply(S.det_metrics, include_groups=False).reset_index()
                   .set_index("config").sort_index())
        ref = pd.read_csv(SENS / f"all_subsets_{split}.csv").set_index("config").sort_index()
        if len(ref) != 255 or list(summ.index) != list(ref.index):
            bad.append(f"{split}: {len(ref)} subsets")
            continue
        for col in ref.columns:
            if col not in summ.columns:
                continue
            x, y = summ[col], ref[col]
            same = (np.allclose(x.astype(float), y.astype(float), equal_nan=True)
                    if pd.api.types.is_numeric_dtype(y) else (x.astype(str) == y.astype(str)).all())
            if not same:
                bad.append(f"{split}:{col}")
    c.expect(not bad, "sensors.recompute", "all 255 channel subsets (LOTO + 8-fold) "
             "recomputed from per_test_*.csv.gz match all_subsets_*.csv" if not bad
             else f"mismatch {bad}")

    tab = pd.read_csv(SENS / "sensor_table_loto.csv")
    md = (SENS / "sensor_table_loto.md").read_text(encoding="utf-8")
    md_rows = {}
    for line in md.splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        mm = re.search(r"\(`(\w+)`\)", line)
        if mm and len(cells) >= 11:
            md_rows[mm.group(1)] = cells
    mism = []
    for _, r in tab.iterrows():
        cells = md_rows.get(r["channel"])
        # the Markdown rounds seconds to integers (round-half-even) and rates to 2 dp
        if cells is None or cells[3] != r["detected"] \
                or abs(float(cells[7]) - r["latency_median_s"]) > 0.5 \
                or abs(float(cells[8]) - r["latency_p90_s"]) > 0.5 \
                or abs(float(cells[6]) - r["fa_per_hr"]) > 0.006 \
                or abs(float(cells[11]) - r["cls_macro_f1"]) > 0.006:
            mism.append(r["channel"])
    c.expect(len(tab) == 8 and not mism, "sensors.table_md",
             "sensor_table_loto.md agrees with sensor_table_loto.csv (8 channels)"
             if not mism else f"disagree for {mism}")


def check_causality(c: Check):
    """Replace every sample after a cut with noise; features up to the cut must not move."""
    import numpy as np
    import pandas as pd
    import features as F
    rng = np.random.default_rng(0)
    n, cut = 400, 250
    chans = list(F.CHANNELS.values())
    t = np.arange(n)
    t = np.delete(t, [40, 41, 300])                      # include gaps
    base = pd.DataFrame(rng.normal(size=(len(t), len(chans))), columns=chans)
    base["Time"] = t
    base["t_rel_heat"] = t - 200
    base["phase"] = np.where(base["t_rel_heat"] < 0, "clean", "heating")
    pert = base.copy()
    after = pert["Time"] > cut
    pert.loc[after, chans] = rng.normal(0, 1e6, (int(after.sum()), len(chans)))
    fa = F.rolling_features(base.copy(), chans)
    fb = F.rolling_features(pert.copy(), chans)
    keep = (base["Time"] <= cut).to_numpy()
    same = np.array_equal(fa[keep].to_numpy(), fb[keep].to_numpy(), equal_nan=True)
    moved = not np.array_equal(fa[~keep].to_numpy(), fb[~keep].to_numpy(), equal_nan=True)
    c.expect(same and moved, "causality.rolling",
             f"{fa.shape[1]} trailing-window features bit-identical up to t={cut} s after "
             "future perturbation" if same else "features at t <= cut changed: LOOK-AHEAD LEAK")
    ra = F.ratio_features(fa, pd.Series(0.1, index=chans))
    rb = F.ratio_features(fb, pd.Series(0.1, index=chans))
    c.expect(np.array_equal(ra[keep].to_numpy(), rb[keep].to_numpy(), equal_nan=True),
             "causality.ratios", "ratio features causal")


def check_tests(c: Check):
    try:
        import pytest  # noqa: F401
    except ImportError:
        c.warn("tests", "pytest not installed; skipped")
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests"], cwd=ROOT, capture_output=True, text=True, env=env)
    tail = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
    c.expect(p.returncode == 0, "tests.root", tail if p.returncode == 0
             else (p.stdout + p.stderr)[-800:])
    env["PYTHONPATH"] = "integration_update/src"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests", "frontend_adapter/test_adapter.py", "integration_update/tests"],
                       cwd=SENSOR, capture_output=True, text=True, env=env)
    tail = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
    c.expect(p.returncode == 0, "tests.sensor_workspace", tail if p.returncode == 0
             else (p.stdout + p.stderr)[-800:])


# ------------------------------------------------------------------------ full
FULL_STEPS = [
    ("Label audit (writes results/same_label_*)", ["scripts/audit_same_tests.py"], "~2 s", False),
    ("Load raw SAME CSVs", ["src/load.py"], "~1 min", True),
    ("Trailing-window features", ["src/features.py"], "~1 min", True),
    ("Stage 1 anomaly detector, 8-fold", ["src/anomaly.py"], "~10-15 min", True),
    ("Stage 2 material classifier, LOTO-38", ["src/classifier.py"], "~5 min", True),
    ("Per-sensor sensitivity, LOTO + 8-fold", ["src/sensor_sensitivity.py"], "~1.5 h", False),
    ("Chained causal LOTO-41", ["src/evaluate_chained.py"], "~1.3 h", False),
]


def run_full(dry: bool) -> int:
    raw = ROOT / "data" / "raw"
    have_raw = any(raw.glob("*/*.csv"))
    print("\nFull reproduction (results/ and reports/ are overwritten):")
    rc = 0
    for title, cmd, eta, needs_raw in FULL_STEPS:
        line = f"  python {' '.join(cmd):32s} {eta:>10s}  {title}"
        if needs_raw and not have_raw:
            print(line + "  [SKIP: data/raw/<material>/*.csv missing - run src/fetch_psi.py]")
            continue
        print(line)
        if dry:
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        print(f"    -> exit {p.returncode} in {time.time() - t0:.0f} s", flush=True)
        if p.returncode:
            rc = 1
            break
    return rc


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="re-run every pipeline stage after the fast checks (hours)")
    ap.add_argument("--dry-run", action="store_true", help="with --full: print commands only")
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest suites")
    ap.add_argument("--json", type=Path, help="also write the check table as JSON")
    args = ap.parse_args()

    os.chdir(ROOT)
    t0 = time.time()
    c, cache = Check(), {}
    steps = [("env", check_env), ("compile", check_compile), ("seed", check_seed),
             ("data", lambda x: check_data(x, cache)), ("labels", lambda x: check_labels(x, cache)),
             ("audit", check_audit), ("chained", check_chained), ("sensors", check_sensors),
             ("causality", check_causality)]
    if not args.skip_tests:
        steps.append(("tests", check_tests))
    timing = {}
    for name, fn in steps:
        t = time.time()
        try:
            fn(c)
        except Exception as e:  # noqa: BLE001 - a crashing check is a failed check
            c.fail(f"{name}.crash", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        timing[name] = time.time() - t

    width = max(len(r[1]) for r in c.rows)
    print(f"Flame in Freefall - repository verification ({ROOT.name})\n")
    for status, name, msg in c.rows:
        print(f"[{status}] {name:<{width}}  {msg}")
    n = {s: sum(r[0] == s for r in c.rows) for s in ("PASS", "WARN", "FAIL")}
    elapsed = time.time() - t0
    print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed in {elapsed:.1f} s "
          f"(" + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()) + ")")
    if args.json:
        args.json.write_text(json.dumps({"rows": c.rows, "counts": n, "seconds": elapsed,
                                         "timing_s": timing}, indent=2))
    rc = 1 if n["FAIL"] else 0
    if args.full:
        if rc and not args.dry_run:
            print("\nFast checks failed; not starting the full run.")
            return rc
        rc = max(rc, run_full(args.dry_run))
    return rc


if __name__ == "__main__":
    sys.exit(main())
