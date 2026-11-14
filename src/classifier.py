"""Phase 3: material classifier (what is burning?).

Run from the project root (after src/features.py):
    python src/classifier.py

Design
------
* Labelled data: the 38 tests with a verified material. The 3 unverified tests
  (60, 103, 105) are never trained on; the final model scores them as a hint
  for which candidate material is right.
* Windows: 30 s long, every 10 s, from heater-on to 600 s after it. Window
  features = mean of every Phase 1 classifier feature over the window, plus
  max/min of each baseline-corrected sensor and the fraction of seconds where
  each ratio is defined.
* Models: LightGBM (main) and Random Forest, both with balanced class weights,
  fixed hyper-parameters (no tuning on 38 tests).
* Validation: leave-one-test-out, so every prediction is for an unseen burn.
* Test-level decision: average the class probabilities of the windows from
  120-600 s (when smoke is at the detectors) and take the most likely class.
* Ablations (LightGBM): no plume temperature; detector ratios only.

Outputs
-------
reports/classifier/summary.txt, windows_oof.csv, tests_oof.csv,
                   unverified_predictions.csv, figures/*.png
models/material_classifier.joblib
"""
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)       # all-NaN means
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports" / "classifier"
FIG = REP / "figures"
MODEL_OUT = ROOT / "models" / "material_classifier.joblib"

CHANNELS = ["iss_obs", "iss_sct", "sts", "ion_a", "ion_b", "ptrak", "dtb", "plume_t"]
WIN, STRIDE, T_MAX = 30, 10, 600
DECIDE_FROM = 120                     # test-level decision uses windows starting >= this
MIN_ROWS = 20                         # window needs >= 20 of its 30 seconds
CLASSES = ["Cotton Lamp wick", "Kapton", "Pyrell", "Silicone", "Teflon"]

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.5, class_weight="balanced",
                  random_state=0, verbose=-1, n_jobs=4)
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                 class_weight="balanced_subsample", random_state=0, n_jobs=-1)

# Palette (dataviz reference instance, light mode)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_LGB, C_RF, C_ABL1, C_ABL2 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


# ------------------------------------------------------------------ windows
def build_windows() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROC / "features.parquet")
    base_feats = json.loads((PROC / "feature_columns.json").read_text())["classifier"]
    ratios = [c for c in base_feats if c.startswith("ratio_")]
    df = df[(df["t_rel_heat"] >= 0) & (df["t_rel_heat"] < T_MAX)]

    rows = []
    for tid, test in df.groupby("test_id"):
        t = test["t_rel_heat"].to_numpy()
        F = test[base_feats].to_numpy(dtype=float)
        C = test[CHANNELS].to_numpy(dtype=float)
        R = test[ratios].to_numpy(dtype=float)
        meta = test.iloc[0]
        for start in range(0, T_MAX - WIN + 1, STRIDE):
            m = (t >= start) & (t < start + WIN)
            if m.sum() < MIN_ROWS:
                continue
            rec = {"test_id": tid, "material": meta["material"],
                   "candidate_materials": meta["candidate_materials"],
                   "label_verified": bool(meta["label_verified"]),
                   "win_start": start, "win_end": start + WIN}
            rec.update(dict(zip([f"{c}__wmean" for c in base_feats], np.nanmean(F[m], axis=0))))
            rec.update(dict(zip([f"{c}__wmax" for c in CHANNELS], np.nanmax(C[m], axis=0))))
            rec.update(dict(zip([f"{c}__wmin" for c in CHANNELS], np.nanmin(C[m], axis=0))))
            rec.update(dict(zip([f"{c}__wdefined" for c in ratios], np.mean(~np.isnan(R[m]), axis=0))))
            rows.append(rec)
    win = pd.DataFrame(rows)
    feats = [c for c in win.columns if "__w" in c]
    return win, feats


def feature_sets(feats):
    return {
        "lgbm": feats,
        "rf": feats,
        "lgbm_no_plume": [f for f in feats if not f.startswith("plume_t")],
        "lgbm_ratios_only": [f for f in feats if f.startswith("ratio_")],
    }


def make_model(name):
    return RandomForestClassifier(**RF_PARAMS) if name == "rf" else lgb.LGBMClassifier(**LGB_PARAMS)


# ---------------------------------------------------------------- evaluate
def loto(win, feats_by_model):
    """Leave-one-test-out probabilities for every window and model."""
    y = win["material"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    groups = win["test_id"].to_numpy()
    proba = {m: np.zeros((len(win), len(CLASSES))) for m in feats_by_model}
    splits = list(LeaveOneGroupOut().split(win, y, groups))
    for m, cols in feats_by_model.items():
        X = win[cols].to_numpy(dtype=float)
        for tr, te in splits:
            proba[m][te] = make_model(m).fit(X[tr], y[tr]).predict_proba(X[te])
        print(f"  {m}: {len(splits)} folds done", flush=True)
    return y, proba


def test_level(win, proba, t_end_max=T_MAX):
    """Average window probabilities from DECIDE_FROM up to t_end_max -> one label per test."""
    sel = (win["win_start"] >= DECIDE_FROM) & (win["win_end"] <= t_end_max)
    p = pd.DataFrame(proba[sel.to_numpy()], columns=CLASSES)
    p["test_id"] = win.loc[sel, "test_id"].to_numpy()
    p = p.groupby("test_id").mean()
    truth = win.groupby("test_id")["material"].first().loc[p.index]
    return p, truth, p.idxmax(axis=1)


def time_to_id(win, proba):
    """Test-level accuracy using only windows that ENDED by time T (cumulative mean)."""
    out = []
    truth = win.groupby("test_id")["material"].first()
    for T in range(WIN, T_MAX + 1, STRIDE):
        sel = (win["win_end"] <= T).to_numpy()
        p = pd.DataFrame(proba[sel], columns=CLASSES)
        p["test_id"] = win.loc[sel, "test_id"].to_numpy()
        pred = p.groupby("test_id").mean().idxmax(axis=1)
        out.append((T, accuracy_score(truth.loc[pred.index], pred)))
    return pd.DataFrame(out, columns=["t_s", "accuracy"])


# ------------------------------------------------------------------- plots
def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot_confusion(truth, pred, title, fname):
    cm = confusion_matrix(truth, pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURF)
    style(ax)
    cmap = matplotlib.colors.ListedColormap(SEQ_BLUE)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    short = ["Cotton", "Kapton", "Pyrell", "Silicone", "Teflon"]
    ax.set_xticks(range(5), short)
    ax.set_yticks(range(5), short)
    for i in range(5):
        for j in range(5):
            if cm[i, j]:
                dark = cm[i, j] > cm.max() * 0.55
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="#fcfcfb" if dark else INK)
    ax.set_xlabel("Predicted material", fontsize=9)
    ax.set_ylabel("True material", fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=110, facecolor=SURF)
    plt.close(fig)


def plot_time_to_id(curves):
    fig, ax = plt.subplots(figsize=(7.5, 4.3), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    labels = {"lgbm": ("LightGBM", C_LGB), "rf": ("Random Forest", C_RF),
              "lgbm_no_plume": ("LightGBM, no plume temp", C_ABL1),
              "lgbm_ratios_only": ("LightGBM, ratios only", C_ABL2)}
    for m, c in curves.items():
        name, col = labels[m]
        ax.plot(c["t_s"], c["accuracy"] * 100, color=col, lw=2, label=name)
    ax.axhline(20, color=INK2, lw=1, ls=":")
    ax.text(140, 16, "chance (5 classes)", color=INK2, fontsize=7, ha="left")
    ax.axvspan(0, 130, color="#f0efec", zorder=0)
    ax.text(5, 96, "before smoke reaches\nthe detectors", color=INK2, fontsize=7, va="top")
    ax.set_ylim(0, 100)
    ax.set_xlim(WIN, T_MAX)
    ax.set_xlabel("Seconds after heater on (only data up to this time is used)", fontsize=8)
    ax.set_ylabel("Tests correctly identified (%)", fontsize=8)
    ax.set_title("How soon is the material identified?", loc="left", color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(0.2, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "time_to_identification.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def plot_importance(imp):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURF)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.barh(range(len(top)), top.values, color=C_LGB, height=0.7)
    ax.set_yticks(range(len(top)), top.index, fontsize=7)
    ax.set_xlabel("Mean |SHAP value| (all classes)", fontsize=8)
    ax.set_title("Top 15 features, final LightGBM model", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=110, facecolor=SURF)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    win_all, feats = build_windows()
    win = win_all[win_all["label_verified"]].reset_index(drop=True)
    unv = win_all[~win_all["label_verified"]].reset_index(drop=True)
    print(f"{len(win):,} labelled windows from {win['test_id'].nunique()} tests, "
          f"{len(feats)} features; {unv['test_id'].nunique()} unverified tests held aside")
    print(win.groupby("material")["test_id"].nunique().to_string(), flush=True)

    fsets = feature_sets(feats)
    y, proba = loto(win, fsets)

    lines, curves, test_tables = [], {}, {}
    zone = (win["win_start"] >= DECIDE_FROM).to_numpy()
    for m in fsets:
        pred_w = np.array(CLASSES)[proba[m].argmax(1)]
        truth_w = win["material"].to_numpy()
        p, truth, pred = test_level(win, proba[m])
        test_tables[m] = pd.DataFrame({"material": truth, "predicted": pred,
                                       "confidence": p.max(axis=1).round(3)}).join(p.round(3))
        curves[m] = time_to_id(win, proba[m])
        lines.append({
            "model": m,
            "window_acc_120_600": accuracy_score(truth_w[zone], pred_w[zone]),
            "window_macroF1_120_600": f1_score(truth_w[zone], pred_w[zone], average="macro"),
            "window_acc_0_120": accuracy_score(truth_w[~zone], pred_w[~zone]),
            "test_acc": accuracy_score(truth, pred),
            "test_macroF1": f1_score(truth, pred, average="macro"),
            "tests_correct": f"{int((truth == pred).sum())}/{len(truth)}",
        })
    summ = pd.DataFrame(lines).set_index("model")

    # per-class F1 (test level) for the main model
    t = test_tables["lgbm"]
    per_class = pd.Series(f1_score(t["material"], t["predicted"], labels=CLASSES, average=None),
                          index=CLASSES).round(2)
    acc_at = {m: c.set_index("t_s")["accuracy"] for m, c in curves.items()}
    milestones = pd.DataFrame({m: s.reindex([60, 120, 150, 180, 240, 300, 450, 600]).round(2)
                               for m, s in acc_at.items()})

    # ---- final model on all labelled windows, SHAP, unverified tests
    X = win[feats].to_numpy(dtype=float)
    final = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)
    sv = shap.TreeExplainer(final).shap_values(win[feats])
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv          # (n, f, classes)
    imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=feats).sort_values(ascending=False)
    by_sensor = imp.groupby(lambda f: "ratios" if f.startswith("ratio_") else f.split("_")[0]
                            if not f.startswith(("ion_", "iss_")) else "_".join(f.split("_")[:2])).sum()
    by_sensor = (by_sensor / by_sensor.sum()).sort_values(ascending=False).round(3)

    unv_p = pd.DataFrame(final.predict_proba(unv[feats].to_numpy(dtype=float)), columns=CLASSES)
    unv_p["test_id"] = unv["test_id"].to_numpy()
    unv_p = unv_p[(unv["win_start"] >= DECIDE_FROM).to_numpy()].groupby("test_id").mean()
    unv_tab = unv_p.round(3)
    unv_tab.insert(0, "candidates", unv.groupby("test_id")["candidate_materials"].first())
    unv_tab.insert(1, "predicted", unv_p.idxmax(axis=1))

    joblib.dump({"model": final, "features": feats, "classes": CLASSES, "window_s": WIN,
                 "stride_s": STRIDE, "decide_from_s": DECIDE_FROM, "lgb_params": LGB_PARAMS},
                MODEL_OUT)

    # ---- save
    pd.concat([win[["test_id", "material", "win_start", "win_end"]],
               pd.DataFrame(proba["lgbm"], columns=[f"p_{c}" for c in CLASSES])], axis=1
              ).to_csv(REP / "windows_oof.csv", index=False)
    pd.concat(test_tables, names=["model"]).to_csv(REP / "tests_oof.csv")
    unv_tab.to_csv(REP / "unverified_predictions.csv")
    pd.concat(curves, names=["model"]).to_csv(REP / "time_to_identification.csv")
    imp.to_csv(REP / "feature_importance.csv", header=["mean_abs_shap"])

    wrong = t[t["material"] != t["predicted"]][["material", "predicted", "confidence"]]
    text = (
        f"Leave-one-test-out, {win['test_id'].nunique()} labelled tests, "
        f"{len(win):,} windows ({WIN}s, stride {STRIDE}s, 0-{T_MAX}s after heater on)\n\n"
        f"{summ.round(3).to_string()}\n\n"
        f"Per-class F1, test level (LightGBM):\n{per_class.to_string()}\n\n"
        f"Test-level accuracy using data up to T seconds after heater on:\n"
        f"{milestones.to_string()}\n\n"
        f"Misclassified tests (LightGBM):\n{wrong.to_string() if len(wrong) else 'none'}\n\n"
        f"Share of SHAP importance by sensor (final LightGBM):\n{by_sensor.to_string()}\n\n"
        f"Top 15 features:\n{imp.head(15).round(4).to_string()}\n\n"
        f"Unverified tests (final LightGBM, windows 120-600 s):\n{unv_tab.to_string()}\n"
    )
    (REP / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    plot_confusion(t["material"], t["predicted"],
                   f"LightGBM, per test ({summ.loc['lgbm', 'tests_correct']} correct)",
                   "confusion_tests_lgbm.png")
    zone_truth = win.loc[zone, "material"]
    zone_pred = np.array(CLASSES)[proba["lgbm"][zone].argmax(1)]
    plot_confusion(zone_truth, zone_pred, "LightGBM, per 30 s window (120-600 s)",
                   "confusion_windows_lgbm.png")
    tr = test_tables["rf"]
    plot_confusion(tr["material"], tr["predicted"],
                   f"Random Forest, per test ({summ.loc['rf', 'tests_correct']} correct)",
                   "confusion_tests_rf.png")
    plot_time_to_id(curves)
    plot_importance(imp)
    print(f"Saved {REP.relative_to(ROOT)}/ and {MODEL_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
