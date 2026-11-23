"""Task 4: per-sensor sensitivity breakdown of the SAME early-warning pipeline.

Run from the project root:
    python src/sensor_sensitivity.py                 # LOTO + 8-fold, detector + classifier
    python src/sensor_sensitivity.py --split loto    # one split only
    python src/sensor_sensitivity.py --no-classifier # detector only (fast)

What it does
------------
For each of the 8 SAME sensor channels it evaluates
  * single-channel   : the per-sensor Isolation Forest detector (src/anomaly.py)
                       fed with that ONE channel only,
  * leave-one-out    : the same detector with that channel REMOVED (7 channels),
  * full baseline    : all 8 channels (the published detector),
and every one of the 255 non-empty channel subsets (cheap, see below), so the
smallest subset that keeps full-model detection speed can be read off directly.

Detector protocol: identical to src/anomaly.py (trained on clean air only,
inner GroupKFold calibration, q = 1.0 threshold, 3 s persistence, t0 = heater on;
SAME heats samples to pyrolysis, there is no open-flame ignition time).
Standard alarms (ISS photoelectric scatter detector, ionisation detector A) are
calibrated per fold exactly as in anomaly.py and reported as the reference.

Efficiency trick (results identical to anomaly.fit_detector): in the per-sensor
detector each channel's Isolation Forest, its out-of-fold clean-air scores and its
reference level depend only on that channel and the training tests. So per outer
fold we fit each channel ONCE and build any channel subset by taking the max of the
normalised channel scores, then re-calibrating the subset threshold on the inner
out-of-fold scores. `--check` verifies this against anomaly.fit_detector.

Material classifier (what is burning?): Random Forest with the parameters of
src/classifier.py (LightGBM is not required), 30 s windows, test-level decision from
windows 120-600 s, evaluated with LOTO and 8-fold GroupKFold, on the same channel
configurations. A feature is allowed only if every channel it uses is in the subset
(single-channel configs therefore have no ratio features).

Event-level detection metrics (per configuration)
  * positive event = the heater-on period of a test   -> TP if alerted at t >= 0
  * negative event = the clean-air period of a test   -> FP if >= 1 false alarm
  * accuracy = (TP + TN) / (2 x tests),  F1 = 2TP / (2TP + FP + FN)
  * false alarms per hour of clean air (alarm events / clean hours)
  * latency = t_detect - t0 (s); median and 90th percentile over detected tests

Input : Sensor_Systems_and_Telemetry_Integration/ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl
        (the validated SAME-R packets; same 41 tests as data/processed/all_tests.parquet)
Output: reports/sensor_sensitivity/
"""
import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anomaly as A  # noqa: E402  (detector code reused as-is)
import features as F  # noqa: E402
import load as L  # noqa: E402
from repro import SEED, seed_everything  # noqa: E402

PACKETS = (ROOT / "Sensor_Systems_and_Telemetry_Integration" / "ml_lead_handoff"
           / "SAME_R_WIDE_PACKETS.jsonl")
OUT = ROOT / "reports" / "sensor_sensitivity"

CHANNELS = A.CHANNELS
LABELS = {
    "iss_obs": "ISS obscuration", "iss_sct": "ISS scatter", "sts": "STS detector",
    "ion_a": "Ionisation A", "ion_b": "Ionisation B", "ptrak": "P-Trak (ultrafine)",
    "dtb": "DustTrak B (mass)", "plume_t": "Gas plume temp.",
}
KIND = {"iss_obs": "optical smoke", "iss_sct": "optical smoke", "sts": "optical smoke",
        "ion_a": "ionisation smoke", "ion_b": "ionisation smoke",
        "ptrak": "particle counter", "dtb": "aerosol mass", "plume_t": "thermocouple"}
Q = A.Q_MAIN
BYTES_PER_SAMPLE = 4            # float32 per channel per sample at 1 Hz (telemetry budget)

# classifier settings, mirrored from src/classifier.py
C_WIN, C_STRIDE, C_TMAX, C_DECIDE, C_MINROWS = 30, 10, 600, 120, 20
CLASSES = ["Cotton Lamp wick", "Kapton", "Pyrell", "Silicone", "Teflon"]
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                 class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)

# palette: same light-mode reference instance as reports/anomaly figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C_FULL, C_CH, C_ISS, C_ION = "#2a78d6", "#52514e", "#eb6834", "#1baf7a"


# ======================================================================= data
def load_packets(path: Path) -> pd.DataFrame:
    """Rebuild the src/load.py output (all_tests) from the validated packets."""
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            r = {"test_id": int(p["test_id"]), "Time": float(p["source_epoch_s"]),
                 "HeatingInProgress": p["heating_in_progress"], "material": p["material"],
                 "candidate_materials": p["candidate_materials"],
                 "label_verified": bool(p["label_verified"]), "source_file": p["source_file"]}
            r.update(p["channels"])
            recs.append(r)
    df = (pd.DataFrame(recs).sort_values(["test_id", "Time"])
          .drop_duplicates(["test_id", "Time"]).reset_index(drop=True))
    df["phase"] = pd.concat([L.label_phases(g) for _, g in df.groupby("test_id")])
    heat0 = df[df["HeatingInProgress"] == 1].groupby("test_id")["Time"].min()
    df["t_rel_heat"] = df["Time"] - df["test_id"].map(heat0)
    return df


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Same steps as features.build(), from an in-memory frame (ptrak -999 already null)."""
    chans = list(F.CHANNELS.values())
    df = raw[F.META + chans].copy()
    for c, bad in F.SENTINELS.items():
        df[c] = df[c].mask(df[c].isin(bad))
    df[chans] = df.groupby("test_id")[chans].ffill()
    parts = []
    for _, test in df.groupby("test_id", sort=True):
        test = test.sort_values("Time").copy()
        test, noise = F.baseline_correct(test, chans)
        roll = F.rolling_features(test, chans)
        parts.append(pd.concat([test, roll, F.ratio_features(roll, noise)], axis=1))
    out = pd.concat(parts, ignore_index=True)
    rolled = [c for c in out.columns if any(c.startswith(f"{ch}_") for ch in chans)]
    ratio = [c for c in out.columns if c.startswith("ratio_")]
    cols = {"anomaly": chans + rolled, "classifier": chans + rolled + ratio}
    return out.sort_values(["test_id", "Time"]).reset_index(drop=True), cols


def splitter(name: str, n_tests: int):
    return LeaveOneGroupOut() if name == "loto" else GroupKFold(8)


def all_subsets():
    return [tuple(s) for k in range(1, 9) for s in itertools.combinations(CHANNELS, k)]


def cfg_name(sub) -> str:
    if len(sub) == 8:
        return "full_8ch"
    if len(sub) == 1:
        return f"only_{sub[0]}"
    if len(sub) == 7:
        return f"without_{(set(CHANNELS) - set(sub)).pop()}"
    return "+".join(sub)


# ================================================================== detector
def run_detector(df: pd.DataFrame, split: str):
    """Outer CV; returns per-channel normalised test scores, per-fold subset thresholds,
    standard-alarm thresholds, and the fold membership."""
    gid = df["test_id"].to_numpy()
    groups = {c: [f"{c}{s}" for s in A.SENSOR_FEATS] for c in CHANNELS}
    subsets = all_subsets()
    te_score = {c: np.full(len(df), np.nan) for c in CHANNELS}
    folds = []
    outer = list(splitter(split, df["test_id"].nunique()).split(df, groups=gid))
    t0 = time.time()
    for k, (tr, te) in enumerate(outer):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te]
        g_tr = dtr["test_id"].to_numpy()
        tm, cm = A.train_mask(dtr), A.clean_mask(dtr)
        inner = list(GroupKFold(A.INNER_FOLDS).split(dtr, groups=g_tr))
        oofn = {}
        for c, cols in groups.items():
            X = A.X_of(dtr, cols)
            oof = np.full(len(dtr), np.nan)
            for itr, iva in inner:
                oof[iva] = A.if_score(A.fit_if(X[itr][tm[itr]]), X[iva])
            ref = float(np.nanquantile(A.sustained(dtr, oof)[cm], A.REF_Q))
            oofn[c] = oof / ref
            te_score[c][te] = A.if_score(A.fit_if(X[tm]), A.X_of(dte, cols)) / ref
        thr = {}
        for sub in subsets:
            ens = np.max([oofn[c] for c in sub], axis=0)
            sus = A.sustained(dtr, ens)[cm]
            thr[sub] = float(np.quantile(sus[~np.isnan(sus)], Q))
        folds.append({"tests": sorted(set(gid[te])), "thr": thr,
                      "std": {n: v[Q] for n, v in A.std_thresholds(dtr).items()}})
        print(f"  [{split}] detector fold {k + 1}/{len(outer)}  ({time.time() - t0:.0f}s)",
              flush=True)
    return te_score, folds


def evaluate_detector(df, te_score, folds, split):
    """Per-test alert times for every subset and for the standard alarms."""
    subsets = all_subsets()
    std_sus = {n: A.sustained(df, s) for n, s in A.standard_signals(df).items()}
    sub_sus = {sub: A.sustained(df, np.max([te_score[c] for c in sub], axis=0))
               for sub in subsets}
    rows = []
    for fold in folds:
        for tid in fold["tests"]:
            test = df[df["test_id"] == tid]
            idx, t_rel = test.index.to_numpy(), test["t_rel_heat"].to_numpy()
            base = {"split": split, "test_id": tid, "material": test["material"].iloc[0],
                    "clean_secs": int((test["phase"] == "clean").sum())}
            for n in std_sus:
                _, base[f"{n}_alert_s"], base[f"{n}_fa"] = A.alert_times(
                    t_rel, std_sus[n][idx], fold["std"][n])
            for sub in subsets:
                _, t_alert, fa = A.alert_times(t_rel, sub_sus[sub][idx], fold["thr"][sub])
                rows.append({**base, "config": cfg_name(sub), "channels": "|".join(sub),
                             "n_channels": len(sub), "alert_s": t_alert, "fa_events": fa})
    return pd.DataFrame(rows)


def det_metrics(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    tp = int(g["alert_s"].notna().sum())
    fp = int((g["fa_events"] > 0).sum())
    fn, tn = n - tp, n - fp
    lead = g["iss_scatter_alert_s"] - g["alert_s"]
    det = g["alert_s"].dropna()
    return pd.Series({
        "tests": n, "detected": tp,
        "det_accuracy": (tp + tn) / (2 * n),
        "det_precision": tp / (tp + fp) if tp + fp else np.nan,
        "det_recall": tp / n,
        "det_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "fa_per_hr": g["fa_events"].sum() / (g["clean_secs"].sum() / 3600),
        "tests_with_fa": fp,
        "latency_median_s": det.median() if len(det) else np.nan,
        "latency_p90_s": det.quantile(0.9) if len(det) else np.nan,
        "lead_vs_iss_median_s": lead.median(),
        "beats_iss_alarm": int((lead > 0).sum()),
    })


def check_equivalence(df):
    """Our cached subset construction must equal anomaly.fit_detector for the full set."""
    gid = df["test_id"].to_numpy()
    tr, te = next(GroupKFold(8).split(df, groups=gid))
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te]
    groups = {c: [f"{c}{s}" for s in A.SENSOR_FEATS] for c in CHANNELS}
    det = A.fit_detector(dtr, groups)
    ref_score, _ = A.score_detector(det, dte)
    # rebuild with this module's fold routine on a 1-fold "split"
    sub = tuple(CHANNELS)
    tm, cm = A.train_mask(dtr), A.clean_mask(dtr)
    oofn, tes = {}, {}
    for c, cols in groups.items():
        X = A.X_of(dtr, cols)
        oof = np.full(len(dtr), np.nan)
        for itr, iva in GroupKFold(A.INNER_FOLDS).split(dtr, groups=dtr["test_id"]):
            oof[iva] = A.if_score(A.fit_if(X[itr][tm[itr]]), X[iva])
        ref = float(np.nanquantile(A.sustained(dtr, oof)[cm], A.REF_Q))
        oofn[c], tes[c] = oof / ref, A.if_score(A.fit_if(X[tm]), A.X_of(dte, cols)) / ref
    sus = A.sustained(dtr, np.max([oofn[c] for c in sub], axis=0))[cm]
    thr = float(np.quantile(sus[~np.isnan(sus)], Q))
    mine = np.max([tes[c] for c in sub], axis=0)
    ok = np.allclose(mine, ref_score) and np.isclose(thr, det["thresholds"][Q])
    print(f"equivalence check vs anomaly.fit_detector: {'PASS' if ok else 'FAIL'} "
          f"(max |diff| {np.max(np.abs(mine - ref_score)):.2e}, thr {thr:.4f} vs "
          f"{det['thresholds'][Q]:.4f})")
    return ok


# ================================================================ classifier
def chans_of(feat: str) -> set:
    base = feat.split("__")[0]
    if base.startswith("ratio_"):
        num, den = base[len("ratio_"):].split("_over_")
        return {num, den}
    return {c for c in CHANNELS if base == c or base.startswith(c + "_")}


def build_windows(df: pd.DataFrame, base_feats: list) -> tuple[pd.DataFrame, list]:
    """Mirror of classifier.build_windows on an in-memory feature frame."""
    ratios = [c for c in base_feats if c.startswith("ratio_")]
    df = df[(df["t_rel_heat"] >= 0) & (df["t_rel_heat"] < C_TMAX)]
    rows = []
    for tid, test in df.groupby("test_id"):
        t = test["t_rel_heat"].to_numpy()
        Fm = test[base_feats].to_numpy(dtype=float)
        Cm = test[CHANNELS].to_numpy(dtype=float)
        Rm = test[ratios].to_numpy(dtype=float)
        meta = test.iloc[0]
        for start in range(0, C_TMAX - C_WIN + 1, C_STRIDE):
            m = (t >= start) & (t < start + C_WIN)
            if m.sum() < C_MINROWS:
                continue
            rec = {"test_id": tid, "material": meta["material"],
                   "label_verified": bool(meta["label_verified"]),
                   "win_start": start, "win_end": start + C_WIN}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN windows
                rec.update(zip([f"{c}__wmean" for c in base_feats], np.nanmean(Fm[m], 0)))
                rec.update(zip([f"{c}__wmax" for c in CHANNELS], np.nanmax(Cm[m], 0)))
                rec.update(zip([f"{c}__wmin" for c in CHANNELS], np.nanmin(Cm[m], 0)))
            rec.update(zip([f"{c}__wdefined" for c in ratios], np.mean(~np.isnan(Rm[m]), 0)))
            rows.append(rec)
    win = pd.DataFrame(rows)
    return win, [c for c in win.columns if "__w" in c]


def run_classifier(win, feats, configs, split):
    win = win[win["label_verified"]].reset_index(drop=True)
    y = win["material"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    g = win["test_id"].to_numpy()
    outer = list(splitter(split, 0).split(win, y, g))
    zone = (win["win_start"] >= C_DECIDE).to_numpy()
    rows, t0 = [], time.time()
    for name, sub in configs.items():
        cols = [f for f in feats if chans_of(f) <= set(sub)]
        X = win[cols].to_numpy(dtype=float)          # NaN passed through, as in classifier.py
        proba = np.zeros((len(win), len(CLASSES)))
        for tr, te in outer:
            proba[te] = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr]).predict_proba(X[te])
        p = pd.DataFrame(proba[zone], columns=CLASSES)
        p["test_id"] = g[zone]
        p = p.groupby("test_id").mean()
        truth = win.groupby("test_id")["material"].first().loc[p.index]
        pred = p.idxmax(axis=1)
        wt, wp = win["material"].to_numpy()[zone], np.array(CLASSES)[proba[zone].argmax(1)]
        rows.append({"split": split, "config": name, "n_features": len(cols),
                     "cls_test_accuracy": accuracy_score(truth, pred),
                     "cls_test_macro_f1": f1_score(truth, pred, average="macro"),
                     "cls_window_accuracy": accuracy_score(wt, wp),
                     "cls_tests_correct": f"{int((truth == pred).sum())}/{len(truth)}"})
        print(f"  [{split}] classifier {name:<18} acc {rows[-1]['cls_test_accuracy']:.3f} "
              f"F1 {rows[-1]['cls_test_macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ==================================================================== plots
def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_latency(res: pd.DataFrame, path: Path, split_label: str):
    """Horizontal box plot: per-test detection latency for each single channel,
    the full 8-channel detector, and the two standard ISS-style alarms."""
    full = res[res["config"] == "full_8ch"]
    rows = [("8-channel detector", full["alert_s"], C_FULL)]
    single = {c: res[res["config"] == f"only_{c}"]["alert_s"] for c in CHANNELS}
    order = sorted(CHANNELS, key=lambda c: (single[c].median(), -single[c].notna().sum()))
    rows += [(LABELS[c], single[c], C_CH) for c in order]
    rows += [("ISS scatter alarm (ref.)", full["iss_scatter_alert_s"], C_ISS),
             ("Ionisation A alarm (ref.)", full["ion_a_alert_s"], C_ION)]
    n_tests = len(full)

    fig, ax = plt.subplots(figsize=(7.16, 4.4), facecolor=SURF)   # IEEE double column
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, which="major")
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    rng = np.random.default_rng(SEED)
    for i, (lab, s, col) in enumerate(rows):
        y = len(rows) - 1 - i
        d = s.dropna().to_numpy()
        if len(d):
            ax.boxplot(d, positions=[y], orientation="horizontal", widths=0.55, whis=(10, 90),
                       showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=col, alpha=0.18, edgecolor=col, lw=1),
                       medianprops=dict(color=col, lw=2.2),
                       whiskerprops=dict(color=col, lw=1), capprops=dict(color=col, lw=1))
            ax.scatter(d,
                       y + rng.uniform(-0.17, 0.17, len(d)), s=7, color=col,
                       alpha=0.55, lw=0, zorder=3)
            med = np.median(d)
            note = f"{med:.0f} s" + ("" if len(d) == n_tests else f"  ({len(d)}/{n_tests})")
            ax.text(1.005, y, note, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=INK2)
        else:
            ax.text(1.005, y, f"none  (0/{n_tests})", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8, color=INK)
    ax.axhline(len(rows) - 1.5, color=GRID, lw=0.8)
    ax.axhline(1.5, color=GRID, lw=0.8)
    ax.set_xlim(0, 2000)
    ax.set_xticks([0, 5, 10, 30, 100, 300, 1000])
    ax.set_xticklabels(["0", "5", "10", "30", "100", "300", "1000"])
    ax.set_xlabel("Detection latency after heater on, t_detect − t0 (s, log scale above 10 s)",
                  fontsize=8)
    ax.text(1.005, len(rows) - 0.35, "median", transform=ax.get_yaxis_transform(),
            fontsize=7, color=INK2, fontweight="bold")
    ax.set_title(f"Per-sensor detection latency, SAME ({n_tests} tests, {split_label})",
                 loc="left", color=INK, fontsize=10)
    fig.text(0.01, 0.01, "Box: 25–75 %, whiskers: 10–90 %, dots: individual tests. "
             "All detectors calibrated to the same clean-air false-alarm quantile (q = 1.0).",
             fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.93, 1))
    fig.savefig(path, dpi=300, facecolor=SURF)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURF)
    plt.close(fig)


def plot_greedy(g: pd.DataFrame, path: Path, split_label: str):
    """Two stacked panels sharing the channel-count axis (no dual y-axis):
    detection latency (top) and material-identification F1 (bottom)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True, facecolor=SURF,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
    x = g["n_channels"].to_numpy()
    for ax in (a1, a2):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    a1.plot(x, g["latency_median_s"], color=C_FULL, lw=2, marker="o", ms=4.5, zorder=3,
            label="median")
    a1.plot(x, g["latency_p90_s"], color=C_FULL, lw=1.2, ls="--", marker="o", ms=3,
            mfc=SURF, zorder=3, label="90th pct.")
    a1.set_ylabel("Detection delay (s)", fontsize=7.5)
    a1.set_ylim(0, max(10, g["latency_p90_s"].max() * 1.3))
    a1.legend(fontsize=6.5, frameon=False, labelcolor=INK2, loc="upper right")
    a2.plot(x, g["cls_test_macro_f1"], color=C_ISS, lw=2, marker="o", ms=4.5, zorder=3)
    a2.set_ylabel("Material macro-F1", fontsize=7.5)
    a2.set_ylim(0, 1)
    for xi, yi, lab in zip(x, g["cls_test_macro_f1"], g["added_short"]):
        a2.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=6, color=INK2)
    a2.set_xticks(range(1, int(x.max()) + 1))
    a2.set_xlabel("Number of sensor channels (greedy forward selection)", fontsize=7.5)
    a1.set_title(f"Adding sensors one at a time ({split_label})", loc="left", color=INK,
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURF)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURF)
    plt.close(fig)


class ClassifierCache:
    """Runs the material classifier per channel subset once and keeps the results."""

    def __init__(self, path, win, feats, split, reuse=False):
        self.path, self.win, self.feats, self.split = path, win, feats, split
        self.table = (pd.read_csv(path) if reuse and path.exists()
                      else pd.DataFrame(columns=["split", "config", "channels"]))

    def get(self, configs: dict) -> pd.DataFrame:
        have = set(self.table["config"])
        todo = {k: v for k, v in configs.items() if k not in have}
        if todo:
            new = run_classifier(self.win, self.feats, todo, self.split)
            new["channels"] = new["config"].map(lambda k: "|".join(todo[k]))
            self.table = pd.concat([self.table, new], ignore_index=True)
            self.table.to_csv(self.path, index=False)
        return self.table[self.table["config"].isin(configs)]

    def f1(self, sub) -> float:
        key = cfg_name(tuple(c for c in CHANNELS if c in sub))
        t = self.get({key: tuple(c for c in CHANNELS if c in sub)})
        return float(t["cls_test_macro_f1"].iloc[0])


def greedy_forward(summ, cache, mini, k_max):
    """Start from the detection-minimal subset; at each step add the channel that most
    improves material macro-F1 (ties -> lower detection latency)."""
    sd = summ.set_index("channels")
    key = lambda sub: "|".join(c for c in CHANNELS if c in sub)  # noqa: E731
    cur = list(mini["channels"].split("|"))
    steps = [(list(cur), "+".join(SHORT[c] for c in cur))]
    while len(cur) < min(k_max, 8):
        cands = []
        for c in CHANNELS:
            if c in cur:
                continue
            sub = cur + [c]
            d = sd.loc[key(sub)]
            cands.append((cache.f1(sub), -d["latency_median_s"], -d["fa_per_hr"], c))
        cands.sort(reverse=True)
        cur.append(cands[0][3])
        steps.append((list(cur), "+" + SHORT[cands[0][3]]))
    rows = []
    for sub, added in steps:
        d = sd.loc[key(sub)]
        t = cache.table.set_index("config").loc[cfg_name(tuple(c for c in CHANNELS if c in sub))]
        rows.append({"n_channels": len(sub), "added_short": added, "channels": key(sub),
                     "detected": d["detected"], "det_f1": d["det_f1"],
                     "fa_per_hr": d["fa_per_hr"], "latency_median_s": d["latency_median_s"],
                     "latency_p90_s": d["latency_p90_s"],
                     "cls_test_accuracy": t["cls_test_accuracy"],
                     "cls_test_macro_f1": t["cls_test_macro_f1"],
                     "cls_tests_correct": t["cls_tests_correct"],
                     "telemetry_bytes_per_s": len(sub) * BYTES_PER_SAMPLE})
    return pd.DataFrame(rows)


# ===================================================================== main
SHORT = {"iss_obs": "OBS", "iss_sct": "SCT", "sts": "STS", "ion_a": "IonA", "ion_b": "IonB",
         "ptrak": "PTrak", "dtb": "DTB", "plume_t": "T"}


def select_best(summ: pd.DataFrame) -> pd.DataFrame:
    """Per subset size: fewest false alarms, most detections, lowest latency."""
    s = summ.copy()
    s = s.sort_values(["n_channels", "detected", "fa_per_hr", "latency_median_s",
                       "latency_p90_s"], ascending=[True, False, True, True, True])
    best = s.groupby("n_channels").head(1).copy()
    best["short"] = best["channels"].map(lambda x: "+".join(SHORT[c] for c in x.split("|")))
    return best


def minimal_subset(summ: pd.DataFrame, full: pd.Series) -> pd.Series:
    """Smallest subset that matches the full model on detections, false alarms and
    median latency (within 1 s) and is no worse than +2 s on P90."""
    ok = summ[(summ["detected"] >= full["detected"])
              & (summ["fa_per_hr"] <= full["fa_per_hr"] + 1e-9)
              & (summ["latency_median_s"] <= full["latency_median_s"] + 1)
              & (summ["latency_p90_s"] <= full["latency_p90_s"] + 2)]
    return ok.sort_values(["n_channels", "latency_median_s", "latency_p90_s"]).iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["loto", "gkf8", "both"], default="both")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify vs anomaly.fit_detector")
    ap.add_argument("--packets", type=Path, default=PACKETS)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse saved per_test_*.csv.gz / classifier_*.csv from a previous run")
    ap.add_argument("--greedy-max", type=int, default=8,
                    help="greedy forward selection up to this many channels (0 = off)")
    args = ap.parse_args()
    seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    raw = load_packets(args.packets)
    df, cols = build_features(raw)
    print(f"{df['test_id'].nunique()} tests, {len(df):,} rows, "
          f"{len(cols['anomaly'])} anomaly features", flush=True)
    if args.check and not check_equivalence(df):
        sys.exit("equivalence check failed")

    splits = ["loto", "gkf8"] if args.split == "both" else [args.split]
    labels = {"loto": "leave-one-test-out", "gkf8": "8-fold GroupKFold"}
    win = feats = None
    if not args.no_classifier:
        win, feats = build_windows(df, cols["classifier"])

    for split in splits:
        det_file = OUT / f"per_test_{split}.csv.gz"
        if args.reuse and det_file.exists():
            res = pd.read_csv(det_file)
            print(f"  [{split}] reusing {det_file.name}", flush=True)
        else:
            te_score, folds = run_detector(df, split)
            res = evaluate_detector(df, te_score, folds, split)
            res.to_csv(det_file, index=False)
        summ = (res.groupby(["config", "channels", "n_channels"])
                .apply(det_metrics, include_groups=False).reset_index())
        full = summ.set_index("config").loc["full_8ch"]
        ref = res[res["config"] == "full_8ch"]
        std = {}
        for n in ("iss_scatter", "ion_a"):
            g = ref.copy()
            g["alert_s"], g["fa_events"] = ref[f"{n}_alert_s"], ref[f"{n}_fa"]
            std[n] = det_metrics(g)
        summ.to_csv(OUT / f"all_subsets_{split}.csv", index=False)

        best = select_best(summ)
        best.to_csv(OUT / f"best_subset_per_size_{split}.csv", index=False)
        mini = minimal_subset(summ, full)

        configs = {"full_8ch": tuple(CHANNELS)}
        configs.update({f"only_{c}": (c,) for c in CHANNELS})
        configs.update({f"without_{c}": tuple(x for x in CHANNELS if x != c) for c in CHANNELS})
        configs[f"minimal::{mini['config']}"] = tuple(mini["channels"].split("|"))
        cls, greedy = pd.DataFrame(columns=["config"]), None
        if win is not None:
            cache = ClassifierCache(OUT / f"classifier_{split}.csv", win, feats, split,
                                    reuse=args.reuse)
            cls = cache.get(configs)
            if args.greedy_max:
                greedy = greedy_forward(summ, cache, mini, args.greedy_max)
                greedy.to_csv(OUT / f"greedy_selection_{split}.csv", index=False)
                cls = cache.table

        # ---------------- 8-row single-channel table (+ baseline rows)
        sd = summ.set_index("config")
        cd = cls.set_index("config") if len(cls) else pd.DataFrame()
        rows = []
        for c in CHANNELS:
            s, lo = sd.loc[f"only_{c}"], sd.loc[f"without_{c}"]
            r = {"channel": c, "sensor": LABELS[c], "type": KIND[c],
                 "detected": f"{int(s['detected'])}/{int(s['tests'])}",
                 "det_accuracy": s["det_accuracy"], "det_f1": s["det_f1"],
                 "fa_per_hr": s["fa_per_hr"],
                 "latency_median_s": s["latency_median_s"], "latency_p90_s": s["latency_p90_s"],
                 "delta_median_vs_full_s": s["latency_median_s"] - full["latency_median_s"],
                 "beats_iss_alarm": f"{int(s['beats_iss_alarm'])}/{int(s['tests'])}",
                 "loo_latency_median_s": lo["latency_median_s"],
                 "loo_delta_median_s": lo["latency_median_s"] - full["latency_median_s"],
                 "loo_det_f1": lo["det_f1"]}
            if len(cd):
                r.update({"cls_macro_f1": cd.loc[f"only_{c}", "cls_test_macro_f1"],
                          "cls_accuracy": cd.loc[f"only_{c}", "cls_test_accuracy"],
                          "loo_cls_macro_f1": cd.loc[f"without_{c}", "cls_test_macro_f1"]})
            rows.append(r)
        table = pd.DataFrame(rows).sort_values("latency_median_s", na_position="last")
        table.insert(0, "rank", range(1, 9))
        extra = [{"channel": "ALL", "sensor": "Full 8-channel detector", "type": "baseline",
                  "detected": f"{int(full['detected'])}/{int(full['tests'])}",
                  "det_accuracy": full["det_accuracy"], "det_f1": full["det_f1"],
                  "fa_per_hr": full["fa_per_hr"], "latency_median_s": full["latency_median_s"],
                  "latency_p90_s": full["latency_p90_s"], "delta_median_vs_full_s": 0.0,
                  "beats_iss_alarm": f"{int(full['beats_iss_alarm'])}/{int(full['tests'])}",
                  **({"cls_macro_f1": cd.loc["full_8ch", "cls_test_macro_f1"],
                      "cls_accuracy": cd.loc["full_8ch", "cls_test_accuracy"]} if len(cd) else {})}]
        for n, lab in (("iss_scatter", "ISS scatter alarm (standard)"),
                       ("ion_a", "Ionisation A alarm (standard)")):
            s = std[n]
            extra.append({"channel": n, "sensor": lab, "type": "reference threshold",
                          "detected": f"{int(s['detected'])}/{int(s['tests'])}",
                          "det_accuracy": s["det_accuracy"], "det_f1": s["det_f1"],
                          "fa_per_hr": s["fa_per_hr"], "latency_median_s": s["latency_median_s"],
                          "latency_p90_s": s["latency_p90_s"],
                          "delta_median_vs_full_s": s["latency_median_s"] - full["latency_median_s"]})
        table.round(3).to_csv(OUT / f"sensor_table_{split}.csv", index=False)
        pd.DataFrame(extra).round(3).to_csv(OUT / f"reference_rows_{split}.csv", index=False)

        md = table_markdown(table, pd.DataFrame(extra), labels[split])
        (OUT / f"sensor_table_{split}.md").write_text(md, encoding="utf-8")

        # telemetry / compute budget of the minimal subset vs full
        k = int(mini["n_channels"])
        budget = {
            "minimal_subset": mini["channels"].split("|"),
            "minimal_subset_metrics": {x: (float(mini[x]) if isinstance(mini[x], (float, np.floating, int, np.integer)) else mini[x])
                                       for x in ["detected", "det_f1", "fa_per_hr",
                                                 "latency_median_s", "latency_p90_s"]},
            "full_metrics": {x: float(full[x]) for x in ["detected", "det_f1", "fa_per_hr",
                                                          "latency_median_s", "latency_p90_s"]},
            "channels": {"full": 8, "minimal": k},
            "telemetry_bytes_per_s_at_1Hz_float32": {"full": 8 * BYTES_PER_SAMPLE,
                                                     "minimal": k * BYTES_PER_SAMPLE},
            "telemetry_MB_per_day": {"full": round(8 * BYTES_PER_SAMPLE * 86400 / 1e6, 3),
                                     "minimal": round(k * BYTES_PER_SAMPLE * 86400 / 1e6, 3)},
            "isolation_forests": {"full": 8, "minimal": k},
            "detector_features": {"full": 8 * len(A.SENSOR_FEATS),
                                  "minimal": k * len(A.SENSOR_FEATS)},
            "reduction_pct": round(100 * (1 - k / 8), 1),
        }
        if len(cd):
            mk = f"minimal::{mini['config']}"
            budget["minimal_subset_classifier_macro_f1"] = float(cd.loc[mk, "cls_test_macro_f1"])
            budget["full_classifier_macro_f1"] = float(cd.loc["full_8ch", "cls_test_macro_f1"])
        (OUT / f"minimal_subset_{split}.json").write_text(json.dumps(budget, indent=2))

        plot_latency(res, OUT / "figures" / f"latency_per_sensor_{split}.png", labels[split])
        if greedy is not None:
            plot_greedy(greedy, OUT / "figures" / f"greedy_selection_{split}.png", labels[split])
            print("\nGreedy forward selection:\n" + greedy.round(3).to_string(index=False))
        print("\n" + md)
        print(json.dumps(budget, indent=2))


def fmt(x, nd=2, suffix=""):
    return "–" if pd.isna(x) else f"{x:.{nd}f}{suffix}"


def table_markdown(t: pd.DataFrame, extra: pd.DataFrame, split_label: str) -> str:
    has_cls = "cls_macro_f1" in t.columns
    head = ("| Rank | Sensor channel | Type | Detected | Det. acc. | Det. F1 | False alarms /h "
            "| Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) | Beats ISS alarm "
            + ("| Material F1 (alone) " if has_cls else "") + "|")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [f"**Per-sensor sensitivity, single channel vs. full 8-channel detector "
             f"({split_label}, t0 = heater on)**", "", head, sep]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['rank']} | {r['sensor']} (`{r['channel']}`) | {r['type']} | {r['detected']} "
            f"| {fmt(r['det_accuracy'])} | {fmt(r['det_f1'])} | {fmt(r['fa_per_hr'])} "
            f"| {fmt(r['latency_median_s'], 0)} | {fmt(r['latency_p90_s'], 0)} "
            f"| {fmt(r['delta_median_vs_full_s'], 0, '')} | {r['beats_iss_alarm']} "
            + (f"| {fmt(r['cls_macro_f1'])} " if has_cls else "") + "|")
    lines.append("")
    rh = ("| Reference | Type | Detected | Det. acc. | Det. F1 | False alarms /h "
          "| Median delay (s) | P90 delay (s) | Δ median vs 8-ch (s) "
          + ("| Material F1 " if has_cls else "") + "|")
    lines += ["**Reference rows**", "", rh, "|" + "---|" * (rh.count("|") - 1)]
    for _, r in extra.iterrows():
        lines.append(
            f"| {r['sensor']} | {r['type']} | {r['detected']} | {fmt(r['det_accuracy'])} "
            f"| {fmt(r['det_f1'])} | {fmt(r['fa_per_hr'])} | {fmt(r['latency_median_s'], 0)} "
            f"| {fmt(r['latency_p90_s'], 0)} | {fmt(r['delta_median_vs_full_s'], 0)} "
            + (f"| {fmt(r.get('cls_macro_f1'))} " if has_cls else "") + "|")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
