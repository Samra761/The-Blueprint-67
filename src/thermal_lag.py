"""Sensor ablation and thermal-lag simulation, with 8-fold false-alarm-rate curves.

Run from the project root:
    python src/thermal_lag.py            # full run
    python src/thermal_lag.py --check    # self-checks only
    python src/thermal_lag.py --plots    # rebuild tables and figures from saved results

Question
--------
The early-warning detector (src/anomaly.py) alerts ~3 s after heater-on, but the alert
always comes from plume_t, a thermocouple right above the heated sample. How much of the
detector survives if that sensor is absent, or responds late as a distant or slow one would?

Scenarios (only plume_t changes; the 7 smoke channels are untouched)
--------------------------------------------------------------------
  baseline     plume_t as recorded (the published detector)
  ablation     plume_t removed, 7 smoke channels left
  shift_D      pure transport delay:  y(t) = x(t - D),              D   = 15, 30, 60 s
  slow_D       first-order response:  y += (x - y)(1 - exp(-dt/tau)), tau = 15, 30, 60 s
The delay is applied to the RAW temperature, before baseline correction and rolling
features, and the detector is retrained and recalibrated for every scenario.

A delayed sensor needs readings from before the recording starts and from inside recording
gaps (up to 641 s in clean air). Those are filled by bouncing back and forth through real
readings ('reflect'): the test's first 30 s for the start (always clean air, never uses the
heater time), the last 30 s before the gap for gaps. Scenarios ending in '--hold' repeat
every delay with the nearest earlier reading frozen instead. Frozen readings have zero
variance, which the detector itself finds abnormal, so 'hold' is only a sensitivity check.

Diagnostics (same delays, separate "heat arrives late" from "the fused detector cannot
use it"):
  only_<v>     plume_t alone, per-sensor Isolation Forest
  thr_<v>      plume_t alone, plain threshold on the baseline-corrected temperature

Baselines (reference level subtracted from each channel)
--------------------------------------------------------
  heater   last 120 s of clean air before heater-on, as in src/features.py. Matches the
           published numbers, but needs the event time, so it is not deployable.
  arming   first 30 s after power-on, as in src/evaluate_chained.py (Task 5). Causal.
           Nothing is trained, calibrated or scored while arming.

Protocol: identical to src/anomaly.py (clean-air training, 8-fold GroupKFold by test,
inner 4-fold out-of-fold calibration, 3 s persistence). Thresholds are swept over
calibration quantiles q = 0.95 ... 1.0 and then raised above the q = 1.0 level by
2-20 % ('x1.02' ... 'x1.20'). Smoke-channel models are fitted once per fold and shared
by all scenarios.

Input : Sensor_Systems_and_Telemetry_Integration/ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl
Output: reports/thermal_lag/
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anomaly as A  # noqa: E402  (detector code reused as-is)
import features as F  # noqa: E402
import sensor_sensitivity as S  # noqa: E402  (packet loader)
from repro import SEED  # noqa: E402  (single seed source, src/repro.py)

OUT = ROOT / "reports" / "thermal_lag"
FIG = OUT / "figures"
MANIFEST = ROOT / "results" / "same_label_manifest.csv"

HOT = "plume_t"
SMOKE = [c for c in A.CHANNELS if c != HOT]
DELAYS = [15, 30, 60]
ARM_S = 30                                   # as in src/evaluate_chained.py
BASELINES = ["heater", "arming"]
Q_GRID = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.9965, 0.998,
          0.999, 0.9995, 0.9998, 1.0]
MARGINS = [1.02, 1.05, 1.10, 1.15, 1.20]     # multiples of the q = 1.0 threshold
SETTINGS = [f"q{q:g}" for q in Q_GRID] + [f"x{m:.2f}" for m in MARGINS]
MAIN = "q1"
N_BOOT = 2000

# scenario -> plume_t variant (None = channel removed)
VARIANTS = {"raw": ("none", 0, "reflect")}
for _m in ("shift", "slow"):
    for _d in DELAYS:
        VARIANTS[f"{_m}_{_d}"] = (_m, _d, "reflect")
        VARIANTS[f"{_m}_{_d}--hold"] = (_m, _d, "hold")
SCENARIOS = {"baseline": "raw", "ablation": None,
             **{v: v for v in VARIANTS if v != "raw"}}
MAIN_SCEN = [s for s in SCENARIOS if "--" not in s]
DIAG = {f"{k}_{v}": v for k in ("only", "thr") for v in VARIANTS if "--" not in v}
LABEL = {"baseline": "Baseline (no delay)", "ablation": "plume_t removed",
         **{f"shift_{d}": f"Shift {d} s" for d in DELAYS},
         **{f"slow_{d}": f"Slow sensor, τ = {d} s" for d in DELAYS}}


# ================================================================ delay models
def bounce(seg: np.ndarray, n: int) -> np.ndarray:
    """n values walking away from seg[0] and back: seg[1], seg[2], ..., seg[2], seg[1], ..."""
    k = len(seg)
    if k == 1:
        return np.full(n, seg[0])
    j = np.arange(1, n + 1) % (2 * (k - 1))
    return seg[np.where(j < k, j, 2 * (k - 1) - j)]


def delay_signal(t: np.ndarray, x: np.ndarray, mode: str, d: int,
                 pad: str = "reflect") -> np.ndarray:
    """Delayed copy of x (sampled at integer seconds t), returned at the same times.

    Works in real time on a complete 1 s grid, so a delay of d means d seconds even across
    missing samples. Gaps and the pre-recording history are filled as described in the
    module doc. Uses no data after t, except that history for the first ARM_S seconds.
    """
    if mode == "none" or d == 0:
        return x.copy()
    t = t.astype(int)
    grid = np.arange(t[0], t[-1] + 1)
    g = pd.Series(x, index=t).reindex(grid).ffill().to_numpy()
    if pad == "reflect":
        for i in np.flatnonzero(np.diff(t) > 2):           # at least 2 missing seconds
            a, b = t[i] - grid[0] + 1, t[i + 1] - grid[0]
            g[a:b] = bounce(g[max(0, a - ARM_S):a][::-1], b - a)
    n_pad = 6 * d
    hist = (np.full(n_pad, g[0]) if pad == "hold"
            else bounce(g[:min(ARM_S, len(g))], n_pad)[::-1])
    z = np.r_[hist, g]
    if mode == "shift":
        y = z[n_pad - d: len(z) - d]
    elif mode == "slow":
        a = 1 - np.exp(-1.0 / d)
        out = np.empty_like(z)
        acc = hist.mean()
        for i, v in enumerate(z):
            acc += a * (v - acc)
            out[i] = acc
        y = out[n_pad:]
    else:
        raise ValueError(mode)
    return y[t - grid[0]]


# ==================================================================== features
def channel_features(raw: pd.DataFrame, values: pd.DataFrame, chans: list,
                     baseline: str) -> pd.DataFrame:
    """[value, mean10, slope10, std10] for `chans`, baseline-corrected, per test."""
    parts = []
    for _, idx in raw.groupby("test_id", sort=True).groups.items():
        test = raw.loc[idx, ["Time", "phase", "t_rel_heat"]].copy()
        test[chans] = values.loc[idx, chans].astype(float).ffill()
        if baseline == "heater":
            test, _ = F.baseline_correct(test, chans)
        else:
            ref = test["Time"] < test["Time"].iloc[0] + ARM_S
            test[chans] = test[chans] - test.loc[ref, chans].median()
        parts.append(pd.concat([test[chans], F.rolling_features(test, chans)], axis=1))
    cols = [f"{c}{s}" for c in chans for s in A.SENSOR_FEATS]
    return pd.concat(parts).loc[raw.index, cols]


def prepare(raw: pd.DataFrame, baseline: str):
    """Frame of keys/masks, dict variant -> feature matrix, and baseline-corrected signals
    of the standard alarms."""
    df = raw[["test_id", "Time", "phase", "t_rel_heat"]].copy()
    t0 = raw.groupby("test_id")["Time"].transform("min")
    df["armed"] = True if baseline == "heater" else (raw["Time"] >= t0 + ARM_S)
    sm = channel_features(raw, raw, SMOKE, baseline)
    X = {c: A.X_of(sm, [f"{c}{s}" for s in A.SENSOR_FEATS]) for c in SMOKE}
    for name, (mode, d, pad) in VARIANTS.items():
        v = raw[[HOT]].copy()
        for _, idx in raw.groupby("test_id").groups.items():
            v.loc[idx, HOT] = delay_signal(raw.loc[idx, "Time"].to_numpy(),
                                           raw.loc[idx, HOT].to_numpy(float), mode, d, pad)
        X[name] = A.X_of(channel_features(raw, v, [HOT], baseline),
                         [f"{HOT}{s}" for s in A.SENSOR_FEATS])
    std = A.standard_signals(sm)
    return df, X, std


# ==================================================================== detector
def fit_if(X):
    return IsolationForest(**{**A.IF_PARAMS, "n_jobs": 1}).fit(X)


def one_fit(Xtr, Xte, keys, tm, cm, armed_tr, inner):
    """One channel, one outer fold: normalised out-of-fold train scores and test scores."""
    oof = np.full(len(Xtr), np.nan)
    for itr, iva in inner:
        oof[iva] = A.if_score(fit_if(Xtr[itr][tm[itr]]), Xtr[iva])
    oof[~armed_tr] = np.nan
    ref = float(np.nanquantile(A.sustained(keys, oof)[cm], A.REF_Q))
    return oof / ref, A.if_score(fit_if(Xtr[tm]), Xte) / ref, ref


def thresholds(sus_clean: np.ndarray, margins: bool = True) -> np.ndarray:
    s = sus_clean[~np.isnan(sus_clean)]
    th = np.quantile(s, Q_GRID)
    return np.r_[th, th[-1] * np.array(MARGINS)] if margins else np.r_[th, [th[-1]] * len(MARGINS)]


def run_baseline(raw, baseline, n_jobs):
    """Per-test results for every scenario and setting under one baseline."""
    t_start = time.time()
    df, X, std = prepare(raw, baseline)
    gid = df["test_id"].to_numpy()
    armed = df["armed"].to_numpy(bool)
    outer = list(GroupKFold(A.OUTER_FOLDS).split(df, groups=gid))

    jobs, meta = [], []
    for k, (tr, te) in enumerate(outer):
        keys = df.iloc[tr][["test_id", "Time"]].reset_index(drop=True)
        dtr = df.iloc[tr].reset_index(drop=True)
        arm_tr = armed[tr]
        tm, cm = A.train_mask(dtr) & arm_tr, A.clean_mask(dtr) & arm_tr
        inner = list(GroupKFold(A.INNER_FOLDS).split(dtr, groups=gid[tr]))
        for name, M in X.items():
            jobs.append(delayed(one_fit)(M[tr], M[te], keys, tm, cm, arm_tr, inner))
            meta.append((k, name))
    print(f"[{baseline}] {len(jobs)} channel fits ...", flush=True)
    fits = dict(zip(meta, Parallel(n_jobs=n_jobs)(jobs)))
    print(f"[{baseline}] fitted ({time.time() - t_start:.0f}s)", flush=True)

    for v in VARIANTS:                               # plain threshold: value is the score
        if "--" not in v:
            for k, (tr, te) in enumerate(outer):
                val = X[v][:, 0]
                fits[k, f"thr_{v}"] = (np.where(armed, val, np.nan)[tr], val[te], np.nan)
    te_score = {n: np.full(len(df), np.nan) for n in {m[1] for m in fits}}
    for k, (tr, te) in enumerate(outer):
        for n in te_score:
            te_score[n][te] = fits[k, n][1]
    for n in te_score:
        te_score[n][~armed] = np.nan
    std = {n: np.where(armed, s, np.nan) for n, s in std.items()}
    std_sus = {n: A.sustained(df, s) for n, s in std.items()}

    rows = []
    scen_cols = {s: SMOKE + ([v] if v else []) for s, v in SCENARIOS.items()}
    scen_cols.update({s: [v if s.startswith("only") else s] for s, v in DIAG.items()})
    sat = []
    scen_sus = {s: A.sustained(df, np.max([te_score[c] for c in cols], axis=0))
                for s, cols in scen_cols.items()}
    for k, (tr, te) in enumerate(outer):
        dtr = df.iloc[tr].reset_index(drop=True)
        cm = A.clean_mask(dtr) & armed[tr]
        thr = {s: thresholds(A.sustained(dtr, np.max([fits[k, c][0] for c in cols], axis=0))[cm])
               for s, cols in scen_cols.items()}
        std_thr = {n: thresholds(A.sustained(dtr, s[tr])[cm], margins=False)
                   for n, s in std.items()}
        for tid in sorted(set(gid[te])):
            idx = np.flatnonzero(gid == tid)
            t_rel = df["t_rel_heat"].to_numpy()[idx]
            hot = (t_rel >= 0) & (t_rel <= 120)      # before smoke usually arrives
            for s, v in SCENARIOS.items():
                if v:
                    sat.append({"baseline": baseline, "scenario": s, "fold": k + 1,
                                "test_id": int(tid), "plume_ref_raw_score": fits[k, v][2],
                                "plume_peak_0_120s": np.nanmax(te_score[v][idx][hot]),
                                "fused_threshold_q1": thr[s][Q_GRID.index(1.0)]})
            base = {"baseline": baseline, "fold": k + 1, "test_id": int(tid),
                    "clean_secs": int(np.sum((t_rel < 0) & armed[idx]))}
            for j, setting in enumerate(SETTINGS):
                ref = {}
                for n in std:
                    _, ref[f"{n}_alert_s"], ref[f"{n}_fa"] = A.alert_times(
                        t_rel, std_sus[n][idx], std_thr[n][j])
                for s, cols in scen_cols.items():
                    i, alert, fa = A.alert_times(t_rel, scen_sus[s][idx], thr[s][j])
                    trig = None
                    if i is not None:
                        trig = cols[int(np.argmax([te_score[c][idx[i]] for c in cols]))]
                        trig = trig if trig in SMOKE else HOT
                    rows.append({**base, "scenario": s, "setting": setting, "alert_s": alert,
                                 "fa_events": fa, "trigger": trig, **ref})
    print(f"[{baseline}] done ({time.time() - t_start:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sat)


# ==================================================================== metrics
def poisson_ci(n, hours, level=0.95):
    a = 1 - level
    lo = st.chi2.ppf(a / 2, 2 * n) / 2 if n else 0.0
    return lo / hours, st.chi2.ppf(1 - a / 2, 2 * n + 2) / 2 / hours


def curve_point(g: pd.DataFrame, boot: np.ndarray) -> dict:
    g = g.sort_values("test_id")
    fa, secs = g["fa_events"].to_numpy(), g["clean_secs"].to_numpy()
    al = g["alert_s"].to_numpy()
    hrs, n = secs.sum() / 3600, int(fa.sum())
    lo, hi = poisson_ci(n, hrs)
    bfa = fa[boot].sum(1) / (secs[boot].sum(1) / 3600)
    det = al[~np.isnan(al)]
    # A test that never alerts counts as an infinite delay, in the CI as in the P90 below.
    bmed = np.median(np.where(np.isnan(al), np.inf, al)[boot], axis=1)
    lead = g["iss_scatter_alert_s"].to_numpy() - al
    rate = n / hrs
    return {"tests": len(g), "detected": len(det), "fa_events": n, "clean_hours": hrs,
            "fa_per_hr": rate, "fa_poisson_lo": lo, "fa_poisson_hi": hi,
            "fa_boot_lo": np.quantile(bfa, 0.025), "fa_boot_hi": np.quantile(bfa, 0.975),
            "tests_with_fa": int((fa > 0).sum()),
            "delay_median_s": np.median(det) if len(det) else np.nan,
            "delay_boot_lo": np.quantile(bmed, 0.025, method="lower"),
            "delay_boot_hi": np.quantile(bmed, 0.975, method="higher"),
            "delay_p90_s": np.quantile(det, 0.9) if len(det) else np.nan,
            "delay_max_s": det.max() if len(det) else np.nan,
            "lead_vs_iss_median_s": np.nanmedian(lead) if np.isfinite(lead).any() else np.nan,
            "beats_iss": int(np.nansum(lead > 0)),
            "trigger_plume": int((g["trigger"] == HOT).sum()),
            # share of tests in which a false alarm would be expected anyway in [0, alert]
            "chance_alert_share": float(np.mean(1 - np.exp(-rate * det / 3600))) if len(det) else np.nan}


def make_tables(res: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    n = res["test_id"].nunique()
    boot = rng.integers(0, n, (N_BOOT, n))
    ref = []
    for nm, lab in (("iss_scatter", "iss_scatter_alarm"), ("ion_a", "ion_a_alarm")):
        r = res[(res["scenario"] == "baseline") & res["setting"].str.startswith("q")].copy()
        r["scenario"], r["alert_s"], r["fa_events"] = lab, r[f"{nm}_alert_s"], r[f"{nm}_fa"]
        r["trigger"] = None
        ref.append(r)
    full = pd.concat([res, *ref], ignore_index=True)
    rows = []
    for (b, s, q), g in full.groupby(["baseline", "scenario", "setting"], sort=False):
        assert len(g) == n and g["test_id"].is_unique
        rows.append({"baseline": b, "scenario": s, "setting": q, **curve_point(g, boot)})
    per_fold = (full.groupby(["baseline", "scenario", "setting", "fold"], sort=False)
                .agg(tests=("test_id", "size"), detected=("alert_s", "count"),
                     fa_events=("fa_events", "sum"), clean_secs=("clean_secs", "sum"),
                     delay_median_s=("alert_s", "median")).reset_index())
    per_fold["fa_per_hr"] = per_fold["fa_events"] / (per_fold["clean_secs"] / 3600)
    return pd.DataFrame(rows), per_fold


def paired_fa(res: pd.DataFrame) -> pd.DataFrame:
    """False-alarm difference vs baseline, test-paired bootstrap, at selected settings."""
    rng = np.random.default_rng(SEED + 1)
    n = res["test_id"].nunique()
    boot = rng.integers(0, n, (N_BOOT, n))
    rows = []
    for (b, q), g in res[res["setting"].isin(["q0.95", "q0.99", "q0.999", MAIN])].groupby(
            ["baseline", "setting"], sort=False):
        fa = g.pivot(index="test_id", columns="scenario", values="fa_events")
        secs = g[g["scenario"] == "baseline"].set_index("test_id")["clean_secs"].loc[fa.index].to_numpy()
        hb = secs[boot].sum(1) / 3600
        for s in fa.columns:
            if s == "baseline" or s in DIAG:
                continue
            diff = (fa[s] - fa["baseline"]).to_numpy()
            bd = diff[boot].sum(1) / hb
            rows.append({"baseline": b, "setting": q, "scenario": s,
                         "fa_per_hr_diff": diff.sum() / (secs.sum() / 3600),
                         "ci_lo": np.quantile(bd, 0.025), "ci_hi": np.quantile(bd, 0.975)})
    return pd.DataFrame(rows)


def summary_md(cur: pd.DataFrame) -> str:
    out = []
    for b, lab in (("heater", "Heater-aligned baseline (published protocol)"),
                   ("arming", "Arming baseline (causal, deployable)")):
        out += [f"**{lab}, q = 1.0, 8-fold GroupKFold, 41 tests**", "",
                "| Scenario | Detected | Median delay (s) [95 % CI] | P90 (s) | Max (s) | Lead vs ISS (s) "
                "| Beats ISS | Triggered by plume_t | False alarms | FA/h [95 % Poisson CI] |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        c = cur[(cur["baseline"] == b) & (cur["setting"] == MAIN)].set_index("scenario")
        names = {**LABEL, "iss_scatter_alarm": "ISS scatter alarm (ref.)",
                 "ion_a_alarm": "Ionisation A alarm (ref.)"}
        for s, nm in names.items():
            r = c.loc[s]
            isref = s.endswith("_alarm")
            hi = r["delay_boot_hi"]
            out.append(
                f"| {nm} | {int(r['detected'])}/{int(r['tests'])} | {r['delay_median_s']:.0f} "
                f"[{r['delay_boot_lo']:.0f}, {hi:.0f}] | {r['delay_p90_s']:.0f} | {r['delay_max_s']:.0f} "
                + ("| – | – | – " if isref else
                   f"| {r['lead_vs_iss_median_s']:.0f} | {int(r['beats_iss'])}/41 "
                   f"| {int(r['trigger_plume'])}/{int(r['detected'])} ")
                + f"| {int(r['fa_events'])} | {r['fa_per_hr']:.2f} "
                  f"[{r['fa_poisson_lo']:.2f}, {r['fa_poisson_hi']:.2f}] |")
        out.append("")
    return "\n".join(out)


def diag_md(cur: pd.DataFrame) -> str:
    """Temperature-only diagnostics and score headroom, next to the fused detector."""
    hd = pd.read_csv(OUT / "score_headroom_per_test.csv")
    hd["clears"] = hd["plume_peak_0_120s"] > hd["fused_threshold_q1"]
    hd = hd.groupby(["baseline", "scenario"]).agg(
        ref=("plume_ref_raw_score", "mean"), peak=("plume_peak_0_120s", "median"),
        thr=("fused_threshold_q1", "mean"), clears=("clears", "sum"))
    out = ["**Why: temperature-only diagnostics and score headroom (q = 1.0)**", "",
           "Median delay, detected, false alarms/h for the fused detector, for plume_t alone with "
           "an Isolation Forest (IF), and for plume_t alone with a plain threshold. Headroom: median "
           "over tests of the peak normalised plume_t score in 0-120 s vs the mean fused threshold.",
           "",
           "| Baseline | Scenario | Fused | IF alone | Threshold alone | plume_t clean-air ref (raw IF) "
           "| Peak score | Fused threshold | Tests where peak clears |",
           "|---|---|---|---|---|---|---|---|---|"]
    for b in BASELINES:
        c = cur[(cur["baseline"] == b) & (cur["setting"] == MAIN)].set_index("scenario")

        def cell(s):
            r = c.loc[s]
            med = f"{r['delay_median_s']:.0f} s" if r["detected"] else "none"
            return f"{med}, {int(r['detected'])}/41, {r['fa_per_hr']:.2f}"
        for s in MAIN_SCEN:
            v = SCENARIOS[s]
            if v:
                h = hd.loc[(b, s)]
                out.append(f"| {b} | {LABEL[s]} | {cell(s)} | {cell('only_' + v)} "
                           f"| {cell('thr_' + v)} | {h['ref']:.3f} | {h['peak']:.3f} "
                           f"| {h['thr']:.3f} | {int(h['clears'])}/41 |")
    return "\n".join(out) + "\n"


# ====================================================================== plots
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# delay length is ordinal -> one blue ramp, light to dark; the two anchors get own hues
C = {"baseline": INK, "ablation": "#eb6834", 15: "#86b6ef", 30: "#2a78d6", 60: "#104281",
     "iss": "#1baf7a"}
FAMILY = {"shift": "Pure shift (transport delay)", "slow": "Slow sensor (first-order lag)"}


def style(ax, grid="both"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, facecolor=SURF)
    fig.savefig(FIG / f"{name}.pdf", facecolor=SURF)
    plt.close(fig)


def scen_color(s):
    return C.get(s) or C[int(s.split("_")[1])]


def fam_scens(f):
    return ["baseline", "ablation"] + [f"{f}_{d}" for d in DELAYS]


Q_SET = [s for s in SETTINGS if s.startswith("q")]
QX = np.arange(len(Q_SET))


def q_axis(ax):
    show = ["q0.95", "q0.97", "q0.98", "q0.99", "q0.995", "q0.998", "q0.9995", "q1"]
    ax.set_xticks([Q_SET.index(s) for s in show], [s[1:] for s in show], fontsize=7)
    ax.set_xlabel("Calibration quantile q (more sensitive ← → stricter)", fontsize=8, color=INK2)


def fig_fa_curves(cur):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True, sharey=True, facecolor=SURF)
    for r, b in enumerate(BASELINES):
        for c, f in enumerate(FAMILY):
            ax = axes[r, c]
            style(ax)
            for s in fam_scens(f) + ["iss_scatter_alarm"]:
                d = cur[(cur["baseline"] == b) & (cur["scenario"] == s)].set_index("setting").loc[Q_SET]
                iss = s == "iss_scatter_alarm"
                col = C["iss"] if iss else scen_color(s)
                if s == "baseline":
                    ax.fill_between(QX, d["fa_boot_lo"], d["fa_boot_hi"], color=INK, alpha=0.07, lw=0)
                ax.plot(QX, d["fa_per_hr"], color=col, lw=2, ls="--" if iss else "-", marker="o",
                        ms=3.5, label="ISS scatter alarm" if iss else LABEL[s])
            ax.set_yscale("symlog", linthresh=1, linscale=0.5)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
            q_axis(ax)
            if r == 0:
                ax.set_xlabel("")
            ax.set_title(f"{FAMILY[f]}\n{b}-aligned baseline" if b == "heater"
                         else f"{FAMILY[f]}\narming baseline (causal)",
                         loc="left", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel("False alarms per hour of clean air", fontsize=8, color=INK2)
            ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc="lower left")
    fig.text(0.01, 0.008, "8-fold GroupKFold, 41 tests. Grey band: 95 % test-bootstrap interval of the "
             "baseline. One false alarm ≈ 0.4 per hour; log scale above 1.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "fa_rate_curves")


def fig_per_fold(pf, cur):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True, sharey=True, facecolor=SURF)
    for ax, s in zip(axes.flat, MAIN_SCEN):
        style(ax)
        for k in range(1, A.OUTER_FOLDS + 1):
            d = pf[(pf["baseline"] == "heater") & (pf["scenario"] == s) & (pf["fold"] == k)]
            ax.plot(QX, d.set_index("setting").loc[Q_SET, "fa_per_hr"], color="#b9b8b3", lw=0.9)
        d = cur[(cur["baseline"] == "heater") & (cur["scenario"] == s)].set_index("setting").loc[Q_SET]
        ax.plot(QX, d["fa_per_hr"], color="#2a78d6", lw=2.2)
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        ax.set_title(LABEL[s], loc="left", fontsize=9, color=INK)
        ax.set_xticks([Q_SET.index(v) for v in ("q0.95", "q0.99", "q0.999", "q1")],
                      ["0.95", "0.99", "0.999", "1"], fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Calibration quantile q", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("False alarms / hour", fontsize=8, color=INK2)
    fig.text(0.01, 0.008, "Grey: each of the 8 folds. Blue: all 41 tests pooled. Heater-aligned baseline.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fa_rate_per_fold")


def fig_tradeoff(cur):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), sharey=True, facecolor=SURF)
    for ax, f in zip(axes, FAMILY):
        style(ax)
        for s in fam_scens(f) + ["iss_scatter_alarm"]:
            d = cur[(cur["baseline"] == "heater") & (cur["scenario"] == s)].set_index("setting")
            d = d.loc[[q for q in SETTINGS if q in d.index]]
            d = d[d["detected"] == 41]               # only settings that miss nothing
            d = d.groupby("fa_per_hr", as_index=False)["delay_median_s"].min()
            iss = s == "iss_scatter_alarm"
            col = C["iss"] if iss else scen_color(s)
            ax.plot(d["fa_per_hr"], d["delay_median_s"], color=col, lw=2, ls="--" if iss else "-",
                    marker="o", ms=4, label="ISS scatter alarm" if iss else LABEL[s])
        ax.set_xscale("symlog", linthresh=1, linscale=0.5)
        ax.set_xlim(-0.05, 100)
        ax.set_xticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
        ax.set_ylim(0, 150)
        ax.set_xlabel("False alarms per hour of clean air", fontsize=8, color=INK2)
        ax.set_title(FAMILY[f], loc="left", fontsize=9.5, color=INK)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, -0.16))
    axes[0].set_ylabel("Median delay after heater on (s)", fontsize=8, color=INK2)
    fig.text(0.01, 0.01, "Lowest median delay at each achieved false-alarm rate, over all threshold "
             "settings that detect all 41 tests. Heater-aligned baseline.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "delay_vs_false_alarms")


def fig_delay(cur):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0), sharey=True, facecolor=SURF)
    x = np.array([0] + DELAYS, float)
    for ax, b in zip(axes, BASELINES):
        style(ax)
        c = cur[(cur["baseline"] == b) & (cur["setting"] == MAIN)].set_index("scenario")
        for f, col, dx in (("shift", "#2a78d6", -0.6), ("slow", INK, 0.6)):
            names = ["baseline"] + [f"{f}_{d}" for d in DELAYS]
            y, lo, hi = (c.loc[names, k].to_numpy(float) for k in
                         ("delay_median_s", "delay_boot_lo", "delay_boot_hi"))
            ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], color=col, lw=2, marker="o", ms=6,
                        elinewidth=1, capsize=2, label=f"{FAMILY[f]}, fused detector")
            for xi, yi in list(zip(x, y))[1:]:
                off = (-13, 5) if f == "shift" else (15, -12 if yi > 100 else 3)
                ax.annotate(f"{yi:.0f} s", (xi + dx, yi), textcoords="offset points",
                            xytext=off, ha="center", fontsize=7.5, color=INK2)
        y = c.loc[["thr_raw"] + [f"thr_slow_{d}" for d in DELAYS], "delay_median_s"]
        ax.plot(x, y, color=INK2, lw=1.4, ls=":", marker="s", ms=4, mfc=SURF,
                label="Slow sensor, plain temperature threshold")
        for s, col, lab in (("ablation", C["ablation"], "plume_t removed"),
                            ("iss_scatter_alarm", C["iss"], "ISS scatter alarm")):
            ax.axhline(c.loc[s, "delay_median_s"], color=col, lw=1.6, ls="--",
                       label=lab)
        ax.set_xticks([0] + DELAYS)
        ax.set_xlim(-4, 68)
        ax.set_ylim(-5, 150)
        ax.set_xlabel("Simulated delay D or time constant τ (s)", fontsize=8, color=INK2)
        ax.set_title("Heater-aligned baseline" if b == "heater" else "Arming baseline (causal)",
                     loc="left", fontsize=9.5, color=INK)
    axes[0].set_ylabel("Median detection delay after heater on (s)", fontsize=8, color=INK2)
    fig.text(0.01, 0.01, "q = 1.0, 41 tests, 8-fold GroupKFold. Bars: 95 % test-bootstrap interval of the "
             "median.", fontsize=7, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.04))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "delay_vs_simulated_lag")


# ===================================================================== checks
def check_delay_models(raw):
    """Delay 0 is the identity; delays are exact; nothing after t affects y(t)."""
    rng = np.random.default_rng(SEED)
    for tid in (58, 95, 107):                        # shortest clean air, and large gaps
        g = raw[raw["test_id"] == tid]
        t, x = g["Time"].to_numpy(), g[HOT].to_numpy(float)
        on = int(np.flatnonzero(g["t_rel_heat"].to_numpy() >= 0)[0])
        for mode in ("shift", "slow"):
            assert np.array_equal(delay_signal(t, x, mode, 0), x)
            for d in DELAYS:
                for pad in ("reflect", "hold"):
                    y = delay_signal(t, x, mode, d, pad)
                    for cut in (ARM_S, on, on + 40):
                        x2 = x.copy()
                        x2[cut + 1:] = rng.normal(60, 30, len(x2) - cut - 1)
                        ok = t <= t[cut]
                        assert np.array_equal(delay_signal(t, x2, mode, d, pad)[ok], y[ok]), \
                            f"future data leaks: {tid} {mode} {d} {pad} cut {cut}"
    # a delayed sensor must not look frozen after a long gap (test 107: 511 s)
    g = raw[raw["test_id"] == 107]
    t, x = g["Time"].to_numpy(), g[HOT].to_numpy(float)
    i = int(np.argmax(np.diff(t))) + 1
    assert np.std(delay_signal(t, x, "shift", 30)[i:i + 25]) > 0.05
    # step response: shift moves the step exactly D s; slow reaches 63.2 % at tau
    t = np.arange(0, 400.0)
    x = np.where(t >= 100, 1.0, 0.0)
    for d in DELAYS:
        s = delay_signal(t, x, "shift", d)
        assert s[99 + d] == 0 and s[100 + d] == 1
        y = delay_signal(t, x, "slow", d)
        assert abs(y[99 + d] - (1 - np.exp(-1))) < 1e-9
    return "identity at 0 s, exact step responses, no future leakage"


def check_against_published(res):
    """Baseline and ablation alert times vs reports/anomaly/per_test.csv (q = 1.0)."""
    pub = pd.read_csv(ROOT / "reports" / "anomaly" / "per_test.csv")
    pub = pub[pub["q"] == 1.0]
    out = {}
    for s, d in (("baseline", "per_sensor"), ("ablation", "per_sensor_no_plume")):
        p = pub[pub["detector"] == d].set_index("test_id")
        m = res[(res["baseline"] == "heater") & (res["scenario"] == s)
                & (res["setting"] == MAIN)].set_index("test_id").loc[p.index]
        same = np.isclose(m["alert_s"], p["alert_s"], equal_nan=True)
        out[s] = {"alert_time_identical": f"{int(same.sum())}/{len(p)}",
                  "max_abs_alert_diff_s": float(np.nanmax(np.abs(m["alert_s"] - p["alert_s"]))),
                  "fa_events": int(m["fa_events"].sum()),
                  "fa_events_published": int(p["fa_events"].sum()),
                  "fa_tests": sorted(int(i) for i in m.index[m["fa_events"] > 0]),
                  "fa_tests_published": sorted(int(i) for i in p.index[p["fa_events"] > 0]),
                  "iss_alert_identical": f"{int(np.isclose(m['iss_scatter_alert_s'], p['iss_scatter_alert_s'], equal_nan=True).sum())}/{len(p)}"}
    return out


def check_features(raw):
    """Zero-delay features equal the repo's, bit for bit, under both baselines."""
    df, cols = S.build_features(raw)
    _, X, _ = prepare(raw, "heater")
    for c in SMOKE + ["raw"]:
        name = HOT if c == "raw" else c
        want = A.X_of(df, [f"{name}{s}" for s in A.SENSOR_FEATS])
        assert np.array_equal(X[c], want), f"heater features differ: {name}"
    import evaluate_chained as E
    ep = E.load_packets()
    _, Xa, _ = prepare(raw, "arming")
    for tid in (58, 95, 107):
        g = ep[ep["test_id"] == tid]
        ticks, _ = E.causal_features(g[["Time"] + E.CHANNELS], "arming")
        m = (raw["test_id"] == tid).to_numpy()
        for c in SMOKE + ["raw"]:
            name = HOT if c == "raw" else c
            assert np.allclose(Xa[c][m], A.X_of(ticks, [f"{name}{s}" for s in A.SENSOR_FEATS]),
                               atol=1e-9), f"arming features differ: {name}"
    return "heater = src/features.py (bit-identical), arming = src/evaluate_chained.py"


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the self-checks and stop")
    ap.add_argument("--plots", action="store_true", help="reuse saved per-test results")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.plots:
        raw = S.load_packets(S.PACKETS)
        assert raw["test_id"].nunique() == 41
        checks = {"delay_models": check_delay_models(raw), "features": check_features(raw)}
        for k, v in checks.items():
            print(f"check {k}: PASS ({v})", flush=True)
        if args.check:
            return
        runs = [run_baseline(raw, b, args.jobs) for b in BASELINES]
        res = pd.concat([r[0] for r in runs], ignore_index=True)
        pd.concat([r[1] for r in runs]).round(5).to_csv(OUT / "score_headroom_per_test.csv",
                                                        index=False)
        man = pd.read_csv(MANIFEST).set_index("test_id")["label"]     # Task 3 audited labels
        res.insert(3, "material", res["test_id"].map(man))
        res.to_csv(OUT / "per_test.csv.gz", index=False)
        checks["vs_published_q1"] = check_against_published(res)
        meta = {"checks": checks, "runtime_s": round(time.time() - t0),
                "python": platform.python_version(), "scikit_learn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "outer_folds": A.OUTER_FOLDS, "inner_folds": A.INNER_FOLDS,
                "persistence_s": A.N_PERSIST, "settings": SETTINGS}
        (OUT / "run_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(checks["vs_published_q1"], indent=2))
    res = pd.read_csv(OUT / "per_test.csv.gz")

    cur, pf = make_tables(res)
    cur.round(4).to_csv(OUT / "curves.csv", index=False)
    pf.round(4).to_csv(OUT / "per_fold.csv", index=False)
    paired_fa(res).round(4).to_csv(OUT / "fa_difference_vs_baseline.csv", index=False)
    mat = (res[(res["setting"] == MAIN) & res["scenario"].isin(MAIN_SCEN)]
           .groupby(["baseline", "scenario", "material"], sort=False)["alert_s"]
           .agg(tests="size", median_s="median", max_s="max").reset_index())
    mat.to_csv(OUT / "delay_by_material.csv", index=False)
    md = summary_md(cur) + "\n" + diag_md(cur)
    (OUT / "summary_table.md").write_text(md, encoding="utf-8")
    fig_fa_curves(cur)
    fig_per_fold(pf, cur)
    fig_tradeoff(cur)
    fig_delay(cur)
    try:
        print("\n" + md)
    except UnicodeEncodeError:                       # Windows cp1252 console
        print("\n" + md.encode("ascii", "replace").decode())
    print(f"Saved {OUT.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
