"""Phase 2: Isolation Forest early-warning detector + lead-time evaluation.

Run from the project root (after src/features.py):
    python src/anomaly.py          (~10-15 min: nested cross-validation)

Detectors
---------
* per_sensor           (MAIN) one Isolation Forest per sensor channel, each on
                       that channel's compact features (value, 10 s mean/slope/std).
                       Each sensor's score is divided by its own clean-air
                       reference level; the detector score is the MAX over sensors,
                       so one strongly anomalous sensor is not diluted by the
                       others. We also record which sensor triggered the alert.
* per_sensor_no_plume  same, without plume temperature -> smoke sensors only.
* single_if            one Isolation Forest on all 104 anomaly features (reference;
                       the plume-temperature signal gets diluted here).

Protocol
--------
* Trained ONLY on clean air (last 120 s before the heater turns on).
* Outer GroupKFold by test: every test is scored by models that never saw it.
* Inner GroupKFold on the training tests produces out-of-fold clean-air scores,
  which set (a) each sensor's reference level and (b) the alert threshold at
  quantile q of the sustained score. q = 1.0 -> no false alarm on any
  calibration clean air.
* Alert: score above threshold for N_PERSIST consecutive seconds; alert time is
  the N-th second (when it would fire live).
* Standard alarms, calibrated the same way at the same q (equal sensitivity):
    iss_scatter (the ISS smoke detector), ion_a, and the instrument's own
    STSDetectorAlarm flag (fires in only 5 tests).

Outputs
-------
reports/anomaly/per_test.csv, sweep.csv, summary.txt, figures/*.png
models/anomaly_iforest.joblib
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from repro import SEED, seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports" / "anomaly"
FIG = REP / "figures"
MODEL_OUT = ROOT / "models" / "anomaly_iforest.joblib"

CHANNELS = ["iss_obs", "iss_sct", "sts", "ion_a", "ion_b", "ptrak", "dtb", "plume_t"]
SENSOR_FEATS = ["", "_mean10", "_slope10", "_std10"]
TRAIN_CLEAN_SECS = 120
N_PERSIST = 3
OUTER_FOLDS, INNER_FOLDS = 8, 4
REF_Q = 0.999                      # per-sensor normalisation level
Q_MAIN = 1.0
Q_SWEEP = [0.95, 0.99, 0.995, 0.999, 1.0]
IF_PARAMS = dict(n_estimators=200, max_samples=256, random_state=SEED, n_jobs=-1)
MAIN = "per_sensor"

# Palette (dataviz reference instance, light mode)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_MAIN, C_NP, C_SINGLE = "#2a78d6", "#6250d6", "#008300"
C_ISS, C_ION, C_STS, C_HEAT = "#eb6834", "#1baf7a", "#e87ba4", "#f0efec"


# --------------------------------------------------------------------------- data
def load():
    feats = pd.read_parquet(PROC / "features.parquet")
    cols = json.loads((PROC / "feature_columns.json").read_text())
    sts = pd.read_parquet(PROC / "all_tests.parquet",
                          columns=["test_id", "Time", "STSDetectorAlarm"])
    feats = feats.merge(sts, on=["test_id", "Time"], how="left")
    feats = feats.sort_values(["test_id", "Time"]).reset_index(drop=True)

    def sensor_groups(chans):
        return {c: [f"{c}{s}" for s in SENSOR_FEATS] for c in chans}

    detectors = {
        "per_sensor": sensor_groups(CHANNELS),
        "per_sensor_no_plume": sensor_groups([c for c in CHANNELS if c != "plume_t"]),
        "single_if": {"all": cols["anomaly"]},
    }
    return feats, detectors


def sustained(df: pd.DataFrame, score) -> np.ndarray:
    """Min of `score` over the last N_PERSIST *consecutive* seconds (NaN across gaps)."""
    # df is sorted by (test_id, Time); split into runs of consecutive seconds
    t, tid = df["Time"].to_numpy(), df["test_id"].to_numpy()
    seg = np.cumsum(np.r_[True, (np.diff(t) != 1) | (np.diff(tid) != 0)])
    return (pd.Series(np.asarray(score, dtype=float)).groupby(seg)
              .rolling(N_PERSIST, min_periods=N_PERSIST).min()
              .reset_index(level=0, drop=True).sort_index().to_numpy())


def train_mask(df):
    return ((df["phase"] == "clean") & (df["t_rel_heat"] >= -TRAIN_CLEAN_SECS)).to_numpy()


def clean_mask(df):
    return (df["phase"] == "clean").to_numpy()


def X_of(df, cols):
    return df[cols].fillna(0).to_numpy()


def fit_if(X):
    return IsolationForest(**IF_PARAMS).fit(X)


def if_score(model, X):
    return -model.score_samples(X)          # higher = more anomalous


# ------------------------------------------------------------------ detector
def fit_detector(df_train: pd.DataFrame, groups: dict) -> dict:
    """Fit one IF per feature group, with per-group reference levels and
    thresholds from inner out-of-fold clean-air scores."""
    gid = df_train["test_id"].to_numpy()
    tm, cm = train_mask(df_train), clean_mask(df_train)
    oof = {g: np.full(len(df_train), np.nan) for g in groups}
    for tr, va in GroupKFold(INNER_FOLDS).split(df_train, groups=gid):
        for g, cols in groups.items():
            X = X_of(df_train, cols)
            oof[g][va] = if_score(fit_if(X[tr][tm[tr]]), X[va])
    ref = {g: float(np.nanquantile(sustained(df_train, oof[g])[cm], REF_Q)) for g in groups}
    ens = np.max([oof[g] / ref[g] for g in groups], axis=0)
    sus = sustained(df_train, ens)[cm]
    sus = sus[~np.isnan(sus)]
    thresholds = {q: float(np.quantile(sus, q)) for q in Q_SWEEP}
    models = {g: fit_if(X_of(df_train, cols)[tm]) for g, cols in groups.items()}
    return {"groups": groups, "models": models, "ref": ref, "thresholds": thresholds}


def score_detector(det: dict, df: pd.DataFrame):
    """Return (ensemble score, per-group normalised scores as DataFrame)."""
    per = pd.DataFrame({g: if_score(det["models"][g], X_of(df, cols)) / det["ref"][g]
                        for g, cols in det["groups"].items()}, index=df.index)
    return per.max(axis=1).to_numpy(), per


# --------------------------------------------------------- standard alarms
def standard_signals(df):
    """Single-channel 'standard detector' signals, oriented so smoke = up."""
    return {"iss_scatter": -df["iss_sct"].to_numpy(),   # scatter drops with smoke
            "ion_a": df["ion_a"].to_numpy()}


def std_thresholds(df_train):
    cm = clean_mask(df_train)
    out = {}
    for name, sig in standard_signals(df_train).items():
        sus = sustained(df_train, sig)[cm]
        out[name] = {q: float(np.nanquantile(sus, q)) for q in Q_SWEEP}
    return out


# --------------------------------------------------------------- evaluation
def alert_times(t_rel: np.ndarray, sus: np.ndarray, thr: float):
    """First alert at/after heater-on (index, time) and false-alarm events before it."""
    on = sus > thr
    after = np.flatnonzero(on & (t_rel >= 0))
    first_i = int(after[0]) if len(after) else None
    pre = on & (t_rel < 0)
    fa_events = int(np.sum(pre & ~np.r_[False, pre[:-1]]))
    return first_i, (float(t_rel[first_i]) if first_i is not None else np.nan), fa_events


def run_cv(df, detectors):
    gid = df["test_id"].to_numpy()
    scores = {d: np.full(len(df), np.nan) for d in detectors}
    per_sensor = {d: pd.DataFrame(index=df.index, columns=list(g), dtype=float)
                  for d, g in detectors.items()}
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(OUTER_FOLDS).split(df, groups=gid)):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te]
        fold = {"tests": sorted(set(gid[te])), "std": std_thresholds(dtr), "thr": {}}
        for d, groups in detectors.items():
            det = fit_detector(dtr, groups)
            s, per = score_detector(det, dte)
            scores[d][te] = s
            per_sensor[d].loc[dte.index] = per.values
            fold["thr"][d] = det["thresholds"]
        folds.append(fold)
        print(f"  fold {k + 1}/{OUTER_FOLDS} done", flush=True)
    sus = {d: sustained(df, scores[d]) for d in detectors}
    std_sus = {k: sustained(df, s) for k, s in standard_signals(df).items()}
    return scores, per_sensor, sus, std_sus, folds


def evaluate(df, detectors, per_sensor, sus, std_sus, folds, q):
    rows = []
    for fold in folds:
        for tid in fold["tests"]:
            test = df[df["test_id"] == tid]
            idx = test.index.to_numpy()
            t_rel = test["t_rel_heat"].to_numpy()
            base = {"q": q, "test_id": tid, "material": test["material"].iloc[0],
                    "clean_secs": int((test["phase"] == "clean").sum())}
            for name in std_sus:
                _, base[f"{name}_alert_s"], base[f"{name}_fa_events"] = alert_times(
                    t_rel, std_sus[name][idx], fold["std"][name][q])
            sts_after = (test["STSDetectorAlarm"].to_numpy() == 1) & (t_rel >= 0)
            base["sts_alert_s"] = float(t_rel[sts_after][0]) if sts_after.any() else np.nan
            for d in detectors:
                i, t_alert, fa = alert_times(t_rel, sus[d][idx], fold["thr"][d][q])
                trig = (per_sensor[d].loc[idx[i]].astype(float).idxmax()
                        if i is not None and len(detectors[d]) > 1 else None)
                rows.append({**base, "detector": d, "alert_s": t_alert, "fa_events": fa,
                             "trigger_sensor": trig,
                             "lead_vs_iss_s": base["iss_scatter_alert_s"] - t_alert,
                             "lead_vs_ion_s": base["ion_a_alert_s"] - t_alert,
                             "lead_vs_sts_s": base["sts_alert_s"] - t_alert})
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        hrs = g["clean_secs"].sum() / 3600
        return pd.Series({
            "tests": len(g),
            "detected": int(g["alert_s"].notna().sum()),
            "delay_median_s": g["alert_s"].median(),
            "delay_p90_s": g["alert_s"].quantile(0.9),
            "fa_per_hr": g["fa_events"].sum() / hrs,
            "tests_with_fa": int((g["fa_events"] > 0).sum()),
            "iss_delay_median_s": g["iss_scatter_alert_s"].median(),
            "iss_fa_per_hr": g["iss_scatter_fa_events"].sum() / hrs,
            "lead_vs_iss_median_s": g["lead_vs_iss_s"].median(),
            "earlier_than_iss": int((g["lead_vs_iss_s"] > 0).sum()),
            "lead_vs_ion_median_s": g["lead_vs_ion_s"].median(),
            "lead_vs_sts_median_s": g["lead_vs_sts_s"].median(),
            "sts_fired": int(g["sts_alert_s"].notna().sum()),
        })
    return res.groupby(["q", "detector"]).apply(agg, include_groups=False).reset_index()


# ------------------------------------------------------------------- plots
def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.yaxis.label.set_color(INK2)


def plot_test(df, tid, scores, res_t, thr):
    test = df[df["test_id"] == tid]
    win = test[(test["t_rel_heat"] >= -180) & (test["t_rel_heat"] <= 600)]
    t = win["t_rel_heat"]
    r = res_t.set_index("detector")
    heat_end = win.loc[win["phase"] == "heating", "t_rel_heat"].max()
    panels = [("Detector score\n(÷ threshold)", None), ("ISS scatter (V, Δ)", "iss_sct"),
              ("Ion A (V, Δ)", "ion_a"), ("P-Trak (pt/cc, Δ)", "ptrak"),
              ("Plume temp (°C, Δ)", "plume_t")]
    events = [(r.loc[MAIN, "alert_s"], C_MAIN), (r.loc["per_sensor_no_plume", "alert_s"], C_NP),
              (r.loc[MAIN, "iss_scatter_alert_s"], C_ISS), (r.loc[MAIN, "ion_a_alert_s"], C_ION),
              (r.loc[MAIN, "sts_alert_s"], C_STS)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 9), sharex=True, facecolor=SURF)
    for ax, (label, col) in zip(axes, panels):
        style(ax)
        if pd.notna(heat_end):
            ax.axvspan(0, heat_end, color=C_HEAT, zorder=0)
        if col is None:
            for d, c, l in ((MAIN, C_MAIN, "Per-sensor IF"),
                            ("per_sensor_no_plume", C_NP, "Per-sensor IF, no plume temp")):
                ax.plot(t, scores[d][win.index] / thr[d], color=c, lw=1.5, label=l)
            ax.axhline(1, color=INK2, lw=1, ls=":")
            ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK2)
        else:
            ax.plot(t, win[col], color=INK2, lw=1.2)
        ax.set_ylabel(label, fontsize=8)
        for x, c in events:
            if pd.notna(x) and x <= 600:
                ax.axvline(x, color=c, lw=1.5)
    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in
               ((C_MAIN, "Per-sensor IF alert"), (C_NP, "…no plume temp"),
                (C_ISS, "ISS scatter alarm"), (C_ION, "Ion A alarm"), (C_STS, "STS flag"))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=C_HEAT, label="Heater on"))
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7, frameon=False,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.965))
    trig = r.loc[MAIN, "trigger_sensor"]
    fig.suptitle(f"Test {tid} — {test['material'].iloc[0]}   (alert triggered by: {trig})",
                 color=INK, fontsize=11, x=0.06, ha="left", y=0.99)
    axes[-1].set_xlabel("Seconds relative to heater on", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / f"test_{tid:03d}.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def plot_lead(res_main):
    d = (res_main[res_main["detector"] == MAIN].dropna(subset=["lead_vs_iss_s"])
         .sort_values("lead_vs_iss_s"))
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = np.arange(len(d))
    ax.barh(y, d["lead_vs_iss_s"], color=C_MAIN, height=0.7)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [f"{t} · {m}" for t, m in zip(d["test_id"], d["material"])], fontsize=7)
    ax.set_xlabel("Seconds the per-sensor Isolation Forest alerted before the ISS scatter "
                  "alarm (negative = later)", fontsize=8, color=INK2)
    ax.set_title(f"Lead time per test (median {d['lead_vs_iss_s'].median():.0f} s)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "lead_time_vs_iss.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def plot_sweep(summ):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURF)
    style(ax)
    for d, c, l in ((MAIN, C_MAIN, "Per-sensor IF"),
                    ("per_sensor_no_plume", C_NP, "Per-sensor IF, no plume temp"),
                    ("single_if", C_SINGLE, "Single IF, all features")):
        s = summ[summ["detector"] == d].sort_values("q")
        ax.plot(s["fa_per_hr"], s["delay_median_s"], color=c, lw=2, marker="o", ms=8, label=l)
    s = summ[summ["detector"] == MAIN].sort_values("q")
    ax.plot(s["iss_fa_per_hr"], s["iss_delay_median_s"], color=C_ISS, lw=2, marker="o", ms=8,
            label="ISS scatter threshold")
    ax.set_xlabel("False alarms per hour of clean air (points: q = 0.95 → 1.0)", fontsize=8)
    ax.set_ylabel("Median seconds from heater on to alert", fontsize=8)
    ax.set_title("Detection delay vs false alarms", loc="left", color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=110, facecolor=SURF)
    plt.close(fig)


# -------------------------------------------------------------- final model
def fit_final(df, detectors):
    bundle = {"n_persist": N_PERSIST, "q": Q_MAIN, "train_clean_secs": TRAIN_CLEAN_SECS,
              "ref_q": REF_Q, "main": MAIN, "detectors": {}}
    for d, groups in detectors.items():
        det = fit_detector(df, groups)
        det["threshold"] = det["thresholds"][Q_MAIN]
        bundle["detectors"][d] = det
    bundle["std_thresholds"] = {k: v[Q_MAIN] for k, v in std_thresholds(df).items()}
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)


# --------------------------------------------------------------------- main
def main():
    seed_everything()
    FIG.mkdir(parents=True, exist_ok=True)
    for old in FIG.glob("*.png"):
        old.unlink()
    df, detectors = load()
    print(f"{df['test_id'].nunique()} tests, training rows (clean, last "
          f"{TRAIN_CLEAN_SECS}s): {int(train_mask(df).sum()):,}", flush=True)

    scores, per_sensor, sus, std_sus, folds = run_cv(df, detectors)
    res = pd.concat([evaluate(df, detectors, per_sensor, sus, std_sus, folds, q)
                     for q in Q_SWEEP])
    summ = summarise(res)
    res.to_csv(REP / "per_test.csv", index=False)
    summ.to_csv(REP / "sweep.csv", index=False)

    main_res = res[res["q"] == Q_MAIN]
    m = main_res[main_res["detector"] == MAIN]
    text = (
        f"Operating point q={Q_MAIN}, persistence {N_PERSIST}s, "
        f"{OUTER_FOLDS}-fold GroupKFold by test, main detector = {MAIN}\n\n"
        + summ[summ["q"] == Q_MAIN].set_index("detector").T.round(2).to_string()
        + "\n\nSensor that triggered the main detector's alert:\n"
        + m["trigger_sensor"].value_counts().to_string()
        + "\n\nPer material (main detector), medians in seconds after heater on:\n"
        + m.groupby("material")[["alert_s", "iss_scatter_alert_s", "lead_vs_iss_s"]]
           .median().round(1).to_string()
        + "\n\nTests with a false alarm (main detector):\n"
        + (m.loc[m["fa_events"] > 0, ["test_id", "material", "fa_events", "clean_secs"]]
            .to_string(index=False) or "none")
    )
    (REP / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)

    for fold in folds:
        thr = {d: fold["thr"][d][Q_MAIN] for d in detectors}
        for tid in fold["tests"]:
            plot_test(df, tid, scores, main_res[main_res["test_id"] == tid], thr)
    plot_lead(main_res)
    plot_sweep(summ)
    print("Fitting final models on all tests...", flush=True)
    fit_final(df, detectors)
    print(f"Saved {REP.relative_to(ROOT)}/ and {MODEL_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
