"""Leakage-free, causal, end-to-end evaluation of the two-stage SAME pipeline.

Run from the project root (after ``python scripts/audit_same_tests.py``):
    python src/evaluate_chained.py                 # full run, ~40 min on 2 CPUs
    python src/evaluate_chained.py --max-folds 4   # quick smoke test

Pipeline under test (one 1 Hz tick at a time)
--------------------------------------------
Stage 1  per-sensor Isolation Forest, max-score fusion (``src/anomaly.py``,
         detector ``per_sensor``): one forest per channel on
         [value, mean10, slope10, std10], each score divided by its clean-air
         reference level, fused by the maximum. Alarm when the fused score is
         above threshold for N consecutive seconds (N = 1, 2, 3).
Stage 2  Random Forest material classifier (``src/classifier.py`` RF_PARAMS)
         on 30 s window features. It runs ONLY at ticks where the Stage 1
         alarm is active, on the trailing window [t - 29, t].

Causality and leakage controls
------------------------------
* Features at tick t use only samples at times <= t. Rolling, slope, diff and
  ratio features are the repo's own trailing functions (``src/features.py``).
* Baseline correction never uses the heater flag at inference. ``src/features.py``
  subtracts the median of the last 120 s before heater-on, which presumes
  knowing when the event starts. Here each sequence is baselined on its first
  ARM_S seconds after power-on ("arming"); no alarm can be raised while
  arming. ARM_S is shorter than the shortest clean-air period (42 s), which is
  asserted, so the arming window is always clean air. The repo's
  heater-aligned baseline is evaluated as an ablation (``heater_oracle``).
* Leave-one-test-out over all 41 tests. In each fold, every fitted quantity
  (Isolation Forests, per-sensor reference levels, alarm thresholds, ISS
  scatter threshold, imputer medians, Random Forest) is fit on the 40
  training tests only. Stage 1 thresholds come from inner GroupKFold
  out-of-fold clean-air scores (as in ``src/anomaly.py``) at q = 1.0.
* Stage 2 trains only on tests whose label is allowed by the label policy in
  ``results/same_label_manifest.csv`` (written by the audit script):
  ``repo`` = 38 folder-verified tests, ``psi101`` = 41 tests with labels
  verified against NASA PSI-101. Assertions stop the run if any other test
  enters a supervised split.
* Self-checks: (1) future-perturbation test: replacing all data after a cut
  leaves every feature at or before the cut bit-identical; (2) a separate
  tick-by-tick streaming implementation reproduces the batch scores, alarms
  and Stage 2 probabilities; it is also what the CPU latency is measured on.

Outputs
-------
results/chained_loto_metrics.json   all metrics (machine-readable, flat keys)
results/chained_loto_summary.md     tables for the paper
results/chained_loto_per_test.csv   one row per (baseline, policy, N, test)
results/chained_loto_latency_ms.csv per-tick wall-clock latency (streamed tests)
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as repo_features  # noqa: E402  (repo trailing-window feature code)
from repro import SEED, seed_everything  # noqa: E402  (single seed source)

SENSOR = ROOT / "Sensor_Systems_and_Telemetry_Integration"
PACKETS = SENSOR / "ml_lead_handoff" / "SAME_R_WIDE_PACKETS.jsonl"
FEATURE_COLS = ROOT / "data" / "processed" / "feature_columns.json"
MANIFEST = ROOT / "results" / "same_label_manifest.csv"
REPO_ANOMALY_SWEEP = ROOT / "reports" / "anomaly" / "sweep.csv"
REPO_CLASSIFIER_TESTS = ROOT / "reports" / "classifier" / "tests_oof.csv"
OUT = ROOT / "results"

ARM_S = 30                 # power-on arming period used for the causal baseline
PERSISTENCE = (1, 2, 3)    # consecutive anomalous ticks required for an alarm
MAIN_N = 3                 # repo operating point (src/anomaly.py N_PERSIST)
BASELINES = ("arming", "heater_oracle")
POLICIES = ("repo", "psi101")
PERTURB_TESTS = 3


# ----------------------------------------------------------- repo constants
KNOWN_NAMES = {"SEED": SEED}


def repo_constants(path: Path, names: set[str]) -> dict:
    """Read constants from a repo module without importing it (the classifier
    module imports lightgbm and shap at top level). ``dict(...)`` calls with
    literal keyword values are supported. Bare names in keyword values are
    resolved from ``KNOWN_NAMES`` (e.g. ``random_state=SEED`` -> src/repro.py)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in names:
            continue
        v = node.value
        if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "dict":
            out[name] = {k.arg: (KNOWN_NAMES[k.value.id] if isinstance(k.value, ast.Name)
                                 else ast.literal_eval(k.value)) for k in v.keywords}
        else:
            out[name] = ast.literal_eval(v)
    # tuple-unpacked constants such as  OUTER_FOLDS, INNER_FOLDS = 8, 4
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Tuple)):
            for t, v in zip(node.targets[0].elts, node.value.elts):
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(v)
    missing = names - out.keys()
    assert not missing, f"{path.name} no longer defines {sorted(missing)}"
    return out


AN = repo_constants(ROOT / "src" / "anomaly.py",
                    {"CHANNELS", "SENSOR_FEATS", "TRAIN_CLEAN_SECS", "N_PERSIST",
                     "INNER_FOLDS", "REF_Q", "Q_MAIN", "IF_PARAMS"})
CL = repo_constants(ROOT / "src" / "classifier.py",
                    {"CHANNELS", "WIN", "STRIDE", "T_MAX", "DECIDE_FROM",
                     "MIN_ROWS", "CLASSES", "RF_PARAMS"})
CHANNELS = list(repo_features.CHANNELS.values())
assert CHANNELS == AN["CHANNELS"] == CL["CHANNELS"], "channel lists diverged"
assert AN["N_PERSIST"] == MAIN_N
WIN, STRIDE, T_MAX = CL["WIN"], CL["STRIDE"], CL["T_MAX"]
DECIDE_FROM, MIN_ROWS, CLASSES = CL["DECIDE_FROM"], CL["MIN_ROWS"], CL["CLASSES"]
Q = AN["Q_MAIN"]
COLS = json.loads(FEATURE_COLS.read_text(encoding="utf-8"))
CLF_COLS = COLS["classifier"]
RATIO_COLS = [c for c in CLF_COLS if c.startswith("ratio_")]
S1_GROUPS = {c: [f"{c}{s}" for s in AN["SENSOR_FEATS"]] for c in CHANNELS}
WIN_FEATS = ([f"{c}__wmean" for c in CLF_COLS] + [f"{c}__wmax" for c in CHANNELS]
             + [f"{c}__wmin" for c in CHANNELS] + [f"{c}__wdefined" for c in RATIO_COLS])
LAST_WIN_END = T_MAX - 1                  # last aligned window covers [570, 599]
STEADY_FROM_END = DECIDE_FROM + WIN - 1   # window starting at 120 ends at 149


# ------------------------------------------------------------------- data
def load_packets() -> pd.DataFrame:
    """One row per (test, second): Time, heater flag and the 8 model channels."""
    rows = []
    with PACKETS.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            rows.append({"test_id": p["test_id"], "Time": int(p["source_epoch_s"]),
                         "heat": p["heating_in_progress"], **p["channels"]})
    df = pd.DataFrame(rows)
    df[CHANNELS] = df[CHANNELS].astype(float)
    return df.sort_values(["test_id", "Time"]).reset_index(drop=True)


def load_labels() -> tuple[dict[int, str], dict[str, set[int]], pd.DataFrame]:
    """Labels and allowed supervised test sets per policy from the audit."""
    if not MANIFEST.exists():
        raise SystemExit("results/same_label_manifest.csv missing: "
                         "run `python scripts/audit_same_tests.py` first.")
    man = pd.read_csv(MANIFEST)
    ok = man["label_status"].isin(["verified_folder_and_psi101", "recovered_psi101"])
    assert (man.loc[ok, "label"] == man.loc[ok, "psi101_material"]).all()
    assert set(man.loc[ok, "label"]) <= set(CLASSES)
    policy = {
        "repo": set(man.loc[man["label_status"] == "verified_folder_and_psi101", "test_id"]),
        "psi101": set(man.loc[ok, "test_id"]),
    }
    labels = dict(zip(man.loc[ok, "test_id"], man.loc[ok, "label"]))
    return labels, policy, man


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------- features
def causal_features(raw: pd.DataFrame, baseline: str,
                    heater_on: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tick and per-window features for ONE test.

    ``raw`` holds Time and the 8 channels only. With ``baseline='arming'``
    the heater time must not be passed (structural guarantee that inference
    never sees it). Returns (ticks, windows): ``ticks`` is indexed like
    ``raw``; ``windows`` is on the complete 1 s grid, row t holding the
    aggregate of window [t - WIN + 1, t].
    """
    test = raw[["Time"] + CHANNELS].copy()
    test[CHANNELS] = test[CHANNELS].ffill()          # past values only
    t = test["Time"].to_numpy()
    if baseline == "arming":
        assert heater_on is None, "arming baseline must not see the heater flag"
        ref = test[t < t[0] + ARM_S]
        armed = t >= t[0] + ARM_S
    else:  # heater_oracle: src/features.py baseline_correct
        clean = test[t < heater_on]
        recent = clean[clean["Time"] >= heater_on - repo_features.BASELINE_SECS]
        ref = recent if len(recent) >= 10 else clean
        armed = np.ones(len(test), dtype=bool)
    test[CHANNELS] = test[CHANNELS] - ref[CHANNELS].median()
    noise = ref[CHANNELS].std()
    roll = repo_features.rolling_features(test, CHANNELS)
    ratios = repo_features.ratio_features(roll, noise)
    ticks = pd.concat([test, roll, ratios], axis=1)
    assert list(ticks[CLF_COLS].columns) == CLF_COLS
    ticks["armed"] = armed

    grid = pd.RangeIndex(t[0], t[-1] + 1)
    g = ticks.set_index("Time").reindex(grid)
    present = pd.Series(np.isin(grid, t).astype(float), index=grid)
    rows = present.rolling(WIN, min_periods=1).sum()
    f = g[CLF_COLS]
    s = f.fillna(0.0).rolling(WIN, min_periods=1).sum()
    n = f.notna().astype(float).rolling(WIN, min_periods=1).sum()
    wmean = (s / n.where(n > 0)).add_suffix("__wmean")
    wmax = g[CHANNELS].rolling(WIN, min_periods=1).max().add_suffix("__wmax")
    wmin = g[CHANNELS].rolling(WIN, min_periods=1).min().add_suffix("__wmin")
    defined = g[RATIO_COLS].notna().astype(float).mul(present, axis=0)
    wdef = (defined.rolling(WIN, min_periods=1).sum()
            .div(rows.where(rows > 0), axis=0).add_suffix("__wdefined"))
    windows = pd.concat([wmean, wmax, wmin, wdef], axis=1)[WIN_FEATS]
    windows["rows"] = rows
    windows["is_tick"] = present.astype(bool)
    return ticks, windows


def build_dataset(df: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for all tests plus evaluation-only columns (t_rel, phase)."""
    all_ticks, all_win = [], []
    for tid, raw in df.groupby("test_id", sort=True):
        heat_t = raw.loc[raw["heat"] == 1, "Time"]
        on, off = int(heat_t.min()), int(heat_t.max())
        if baseline == "arming":
            assert raw["Time"].iloc[0] + ARM_S < on, f"test {tid}: arming overlaps heating"
            ticks, win = causal_features(raw[["Time"] + CHANNELS], "arming")
        else:
            ticks, win = causal_features(raw[["Time"] + CHANNELS], "heater_oracle", on)
        for frame, tt in ((ticks, ticks["Time"].to_numpy()), (win, win.index.to_numpy())):
            frame["test_id"] = tid
            frame["t_rel"] = tt - on
            frame["phase"] = np.where(tt < on, "clean", np.where(tt <= off, "heating", "smoke"))
        all_ticks.append(ticks.reset_index(drop=True))
        all_win.append(win.rename_axis("Time").reset_index())
    return (pd.concat(all_ticks, ignore_index=True),
            pd.concat(all_win, ignore_index=True))


# ---------------------------------------------------------------- stage 1
def sustained(tid: np.ndarray, t: np.ndarray, score: np.ndarray, n: int) -> np.ndarray:
    """Min of ``score`` over the last n consecutive seconds (src/anomaly.py
    ``sustained``, generalised to any n). NaN until n consecutive ticks."""
    score = np.asarray(score, dtype=float)
    if n == 1:
        return score.copy()
    seg = np.cumsum(np.r_[True, (np.diff(t) != 1) | (np.diff(tid) != 0)])
    return (pd.Series(score).groupby(seg).rolling(n, min_periods=n).min()
            .reset_index(level=0, drop=True).sort_index().to_numpy())


def s1_matrix(ticks: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return ticks[cols].fillna(0).to_numpy()        # src/anomaly.py X_of


def fit_if(X: np.ndarray, n_jobs: int) -> IsolationForest:
    return IsolationForest(**{**AN["IF_PARAMS"], "n_jobs": n_jobs}).fit(X)


def fit_stage1(train: pd.DataFrame, n_jobs: int) -> dict:
    """Per-sensor forests, reference levels and thresholds from TRAINING tests.

    Reference levels and thresholds come from inner GroupKFold out-of-fold
    scores on training clean air, exactly as in src/anomaly.py."""
    fit_mask = ((train["phase"] == "clean") & train["armed"]
                & (train["t_rel"] >= -AN["TRAIN_CLEAN_SECS"])).to_numpy()
    cal = train[(train["phase"] == "clean") & train["armed"]].reset_index(drop=True)
    tests = np.array(sorted(train["test_id"].unique()))
    oof = {g: np.full(len(cal), np.nan) for g in S1_GROUPS}
    for tr_i, va_i in GroupKFold(AN["INNER_FOLDS"]).split(tests, groups=tests):
        tr_rows = train["test_id"].isin(tests[tr_i]).to_numpy() & fit_mask
        va_rows = cal["test_id"].isin(tests[va_i]).to_numpy()
        for g, cols in S1_GROUPS.items():
            m = fit_if(s1_matrix(train[tr_rows], cols), n_jobs)
            oof[g][va_rows] = -m.score_samples(s1_matrix(cal[va_rows], cols))
    tid, t = cal["test_id"].to_numpy(), cal["Time"].to_numpy()
    ref = {g: float(np.nanquantile(sustained(tid, t, oof[g], AN["N_PERSIST"]), AN["REF_Q"]))
           for g in S1_GROUPS}
    fused = np.max([oof[g] / ref[g] for g in S1_GROUPS], axis=0)
    iss = -cal["iss_sct"].to_numpy()               # scatter drops with smoke
    thr, iss_thr = {}, {}
    for n in PERSISTENCE:
        thr[n] = float(np.nanquantile(sustained(tid, t, fused, n), Q))
        iss_thr[n] = float(np.nanquantile(sustained(tid, t, iss, n), Q))
    models = {g: fit_if(s1_matrix(train[fit_mask], cols), n_jobs)
              for g, cols in S1_GROUPS.items()}
    return {"models": models, "ref": ref, "thr": thr, "iss_thr": iss_thr}


def score_stage1(s1: dict, ticks: pd.DataFrame) -> np.ndarray:
    return np.max([-s1["models"][g].score_samples(s1_matrix(ticks, cols)) / s1["ref"][g]
                   for g, cols in S1_GROUPS.items()], axis=0)


def alarm_events(alarm: np.ndarray, t_rel: np.ndarray, armed: np.ndarray) -> dict:
    """First alert at/after heater-on (src/anomaly.py alert_times) and false-alarm
    events (rising edges) in armed clean air."""
    after = np.flatnonzero(alarm & (t_rel >= 0))
    first = int(after[0]) if len(after) else None
    pre = alarm & (t_rel < 0) & armed
    return {"first_idx": first,
            "alert_s": float(t_rel[first]) if first is not None else np.nan,
            "fa_events": int(np.sum(pre & ~np.r_[False, pre[:-1]])),
            "clean_armed_s": int(np.sum((t_rel < 0) & armed))}


# ---------------------------------------------------------------- stage 2
def make_stage2(n_jobs: int):
    rf = RandomForestClassifier(**{**CL["RF_PARAMS"], "n_jobs": n_jobs})
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), rf)


def aligned_windows(win: pd.DataFrame) -> pd.DataFrame:
    """The src/classifier.py windows: starts 0, 10, ... T_MAX - WIN after heater-on,
    at least MIN_ROWS samples. A window starting at s ends at tick s + WIN - 1."""
    ends = np.arange(WIN - 1, T_MAX, STRIDE)
    return win[win["t_rel"].isin(ends) & (win["rows"] >= MIN_ROWS)]


def class_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true:
        return {"n": 0, "accuracy": None, "macro_f1": None, "correct": 0}
    return {"n": len(y_true), "correct": int(np.sum(np.array(y_true) == np.array(y_pred))),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro",
                                       zero_division=0))}


def tick_zone(t_rel: np.ndarray) -> np.ndarray:
    return np.select([t_rel < 0, t_rel < STEADY_FROM_END, t_rel <= LAST_WIN_END],
                     ["false_alarm", "transient", "steady_state"], "late")


# -------------------------------------------------------------- streaming
class CausalStreamer:
    """Tick-by-tick reference implementation of the deployed two-stage pipeline.

    Holds only O(WIN) state: ring buffers of the last 31 baseline-corrected
    samples and the last WIN per-tick feature rows. It never sees the heater
    flag. Used to (a) prove the batch evaluation has no lookahead and (b)
    measure single-thread CPU latency per tick.
    """

    BUF = max(repo_features.WINDOWS) + 1

    def __init__(self, s1: dict, s2, n_persist: int):
        self.s1, self.s2, self.n = s1, s2, n_persist
        self.t0 = self.t_last = None
        self.last_valid = np.full(len(CHANNELS), np.nan)
        self.arming: list[tuple[int, np.ndarray]] = []
        self.base = self.noise = None
        self.x = deque(maxlen=self.BUF)              # baseline-corrected values
        self.f = deque(maxlen=WIN)                   # per-tick feature rows
        self.p = deque(maxlen=WIN)                   # 1 = real tick, 0 = gap
        self.run: deque = deque(maxlen=n_persist)    # recent fused scores
        self.idx = {c: i for i, c in enumerate(CLF_COLS)}
        self.ch_idx = [self.idx[c] for c in CHANNELS]
        self.r_idx = [self.idx[c] for c in RATIO_COLS]
        self.g_idx = {g: [self.idx[c] for c in cols] for g, cols in S1_GROUPS.items()}

    def _push(self, t: int, xc: np.ndarray) -> np.ndarray:
        if self.t_last is not None:
            for _ in range(t - self.t_last - 1):     # gap: NaN rows, as on the grid
                self.x.append(np.full(len(CHANNELS), np.nan))
                self.f.append(np.full(len(CLF_COLS), np.nan))
                self.p.append(0.0)
                self.run.clear()
        self.t_last = t
        self.x.append(xc)
        feats = self._tick_features()
        self.f.append(feats)
        self.p.append(1.0)
        return feats

    def _tick_features(self) -> np.ndarray:
        X = np.array(self.x)
        out = {}
        for c, v in zip(CHANNELS, X[-1]):
            out[c] = v
        for w in repo_features.WINDOWS:
            mp = max(3, int(0.6 * w))
            win = X[-w:]
            valid = ~np.isnan(win)
            cnt = valid.sum(axis=0)
            tv = np.arange(len(win), dtype=float)[:, None] * np.ones((1, win.shape[1]))
            xs, ts = np.where(valid, win, 0.0), np.where(valid, tv, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = xs.sum(0) / cnt
                var = ((np.where(valid, win - mean, 0.0)) ** 2).sum(0) / (cnt - 1)
                mt = ts.sum(0) / cnt
                cov = (ts * xs).sum(0) / cnt - mt * mean
                vt = (ts * ts).sum(0) / cnt - mt * mt
                slope = np.where(vt != 0, cov / vt, np.nan)
            ok = cnt >= mp
            prev = X[-1 - w] if len(X) > w else np.full(len(CHANNELS), np.nan)
            diff = X[-1] - prev
            for i, c in enumerate(CHANNELS):
                out[f"{c}_mean{w}"] = mean[i] if ok[i] else np.nan
                out[f"{c}_std{w}"] = np.sqrt(var[i]) if ok[i] else np.nan
                out[f"{c}_slope{w}"] = slope[i] if ok[i] else np.nan
                out[f"{c}_diff{w}"] = diff[i]
        for num, den in repo_features.RATIOS:
            n = abs(out[f"{num}_mean{repo_features.RATIO_WINDOW}"])
            d = abs(out[f"{den}_mean{repo_features.RATIO_WINDOW}"])
            floor = repo_features.NOISE_K * self.noise[CHANNELS.index(den)]
            lim = max(floor, 1e-9)
            out[f"ratio_{num}_over_{den}"] = n / d if d > lim else np.nan
        return np.array([out[c] for c in CLF_COLS], dtype=float)

    def _window(self) -> np.ndarray | None:
        P = np.array(self.p)
        rows = P.sum()
        if rows < MIN_ROWS:
            return None
        F = np.array(self.f)
        valid = ~np.isnan(F)
        cnt = valid.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            wmean = np.where(cnt > 0, np.where(valid, F, 0.0).sum(0) / cnt, np.nan)
        C = F[:, self.ch_idx]
        cv = ~np.isnan(C)
        wmax = np.where(cv.any(0), np.where(cv, C, -np.inf).max(0), np.nan)
        wmin = np.where(cv.any(0), np.where(cv, C, np.inf).min(0), np.nan)
        wdef = (~np.isnan(F[:, self.r_idx]) & (P[:, None] > 0)).sum(0) / rows
        return np.concatenate([wmean, wmax, wmin, wdef])[None, :]

    def step(self, t: int, raw: np.ndarray) -> dict:
        """Consume one sample; return the fused score, alarm and Stage 2 output."""
        if self.t0 is None:
            self.t0 = t
        self.last_valid = np.where(np.isnan(raw), self.last_valid, raw)   # ffill
        v = self.last_valid.copy()
        if self.base is None:
            if t < self.t0 + ARM_S:
                self.arming.append((t, v))
                return {"armed": False}
            A = pd.DataFrame([a for _, a in self.arming], columns=CHANNELS)
            self.base = A.median().to_numpy()        # same NaN rules as the batch path
            self.noise = A.std().to_numpy()
            for ta, va in self.arming:                 # replay arming samples
                self._push(ta, va - self.base)
        feats = self._push(t, v - self.base)
        x = np.nan_to_num(feats, nan=0.0)
        fused = max(-self.s1["models"][g].score_samples(x[None, idx])[0] / self.s1["ref"][g]
                    for g, idx in self.g_idx.items())
        self.run.append(fused)
        sus = min(self.run) if len(self.run) == self.n else np.nan
        alarm = bool(sus > self.s1["thr"][self.n])
        proba = w = None
        if alarm:
            w = self._window()
            if w is not None:
                proba = self.s2.predict_proba(w)[0]
        return {"armed": True, "fused": fused, "alarm": alarm, "proba": proba, "window": w}


# ------------------------------------------------------------------- main
def summarise_stage1(rows: pd.DataFrame, prefix: str) -> dict:
    det = rows[f"{prefix}alert_s"].dropna()
    hours = rows["clean_armed_s"].sum() / 3600
    return {"tests": int(len(rows)), "detected": int(len(det)),
            "detection_rate": f"{len(det)}/{len(rows)}",
            "latency_median_s": float(det.median()) if len(det) else None,
            "latency_p90_s": float(det.quantile(0.9)) if len(det) else None,
            "false_alarm_events": int(rows[f"{prefix}fa_events"].sum()),
            "clean_air_hours": round(float(hours), 4),
            "far_per_clean_hour": float(rows[f"{prefix}fa_events"].sum() / hours),
            "tests_with_false_alarm": int((rows[f"{prefix}fa_events"] > 0).sum())}


def run(args) -> None:
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    labels, policy_sets, manifest = load_labels()
    df = load_packets()
    test_ids = sorted(df["test_id"].unique())
    assert len(test_ids) == 41
    aged = set(manifest.loc[pd.to_numeric(manifest["psi101_primary_aging_s"],
                                          errors="coerce") > T_MAX, "test_id"])
    folds = test_ids[: args.max_folds] if args.max_folds else test_ids
    if args.test_ids:
        assert set(args.test_ids) <= set(test_ids), "unknown test id"
        folds = sorted(args.test_ids)
    print(f"{len(test_ids)} tests; LOTO folds evaluated: {len(folds)}; "
          f"label policies {({k: len(v) for k, v in policy_sets.items()})}", flush=True)

    checks: dict = {}
    per_test: list[dict] = []
    tick_rows: list[pd.DataFrame] = []
    oracle_win: list[pd.DataFrame] = []
    fold_models: dict = {}

    for baseline in args.baselines:
        ticks, wins = build_dataset(df, baseline)
        if baseline == "arming":
            ticks_a, wins_a = ticks, wins
        print(f"[{baseline}] features: {len(ticks):,} ticks x {len(CLF_COLS)}, "
              f"{len(wins):,} grid windows", flush=True)

        if baseline == "arming":   # leakage self-check 1: future perturbation
            rng = np.random.default_rng(SEED)
            res = []
            for tid in test_ids[:: max(1, len(test_ids) // PERTURB_TESTS)][:PERTURB_TESTS]:
                raw = df[df["test_id"] == tid].reset_index(drop=True)
                cut = int(raw.loc[raw["heat"] == 1, "Time"].min()) + 60
                pert = raw.copy()
                after = pert["Time"] > cut
                pert.loc[after, CHANNELS] = rng.normal(0, 1e3, (int(after.sum()), len(CHANNELS)))
                a_t, a_w = causal_features(raw[["Time"] + CHANNELS], "arming")
                b_t, b_w = causal_features(pert[["Time"] + CHANNELS], "arming")
                keep_t, keep_w = ~after.to_numpy(), a_w.index <= cut
                same = (a_t.loc[keep_t, CLF_COLS].equals(b_t.loc[keep_t, CLF_COLS])
                        and a_w.loc[keep_w, WIN_FEATS].equals(b_w.loc[keep_w, WIN_FEATS]))
                changed = not a_t.loc[~keep_t, CLF_COLS].equals(b_t.loc[~keep_t, CLF_COLS])
                assert same and changed, f"lookahead detected in test {tid}"
                res.append(int(tid))
            checks["future_perturbation_tests_bit_identical"] = res
            print(f"  future-perturbation check passed on tests {res}", flush=True)

        for k, tid in enumerate(folds):
            fs = time.time()
            train_ids = [t for t in test_ids if t != tid]
            assert tid not in train_ids
            tr = ticks[ticks["test_id"] != tid].reset_index(drop=True)
            te = ticks[ticks["test_id"] == tid].reset_index(drop=True)
            s1 = fit_stage1(tr, args.n_jobs)
            fused = np.full(len(te), np.nan)
            armed = te["armed"].to_numpy()
            fused[armed] = score_stage1(s1, te[armed])
            tid_a, t_a = te["test_id"].to_numpy(), te["Time"].to_numpy()
            t_rel = te["t_rel"].to_numpy()
            iss = np.where(armed, -te["iss_sct"].to_numpy(), np.nan)

            s2 = {}
            for pol in args.policies:
                sup = [t for t in train_ids if t in policy_sets[pol]]
                assert set(sup) <= policy_sets[pol] and tid not in sup, "label leakage"
                tw = aligned_windows(wins[wins["test_id"].isin(sup)])
                y = tw["test_id"].map(labels).to_numpy()
                assert pd.notna(y).all()
                s2[pol] = make_stage2(args.n_jobs).fit(tw[WIN_FEATS].to_numpy(), y)
                if tid in policy_sets[pol]:            # oracle stage 2 (heater-aligned)
                    ow = aligned_windows(wins[wins["test_id"] == tid])
                    p = s2[pol].predict_proba(ow[WIN_FEATS].to_numpy())
                    oracle_win.append(pd.DataFrame(p, columns=s2[pol].classes_).assign(
                        baseline=baseline, policy=pol, test_id=tid,
                        win_start=ow["t_rel"].to_numpy() - WIN + 1, truth=labels[tid]))

            tw_all = wins[wins["test_id"] == tid].set_index("Time")
            for n in PERSISTENCE:
                sus = sustained(tid_a, t_a, fused, n)
                alarm = np.nan_to_num(sus, nan=-np.inf) > s1["thr"][n]
                ev = alarm_events(alarm, t_rel, armed)
                iss_alarm = (np.nan_to_num(sustained(tid_a, t_a, iss, n), nan=-np.inf)
                             > s1["iss_thr"][n])
                iev = alarm_events(iss_alarm, t_rel, armed)
                trig = None
                if ev["first_idx"] is not None:
                    xi = te.iloc[[ev["first_idx"]]]
                    trig = max(S1_GROUPS, key=lambda g: -s1["models"][g].score_samples(
                        s1_matrix(xi, S1_GROUPS[g]))[0] / s1["ref"][g])
                row = {"baseline": baseline, "n_persist": n, "test_id": int(tid),
                       "aged_over_600s": tid in aged, "alert_s": ev["alert_s"],
                       "fa_events": ev["fa_events"], "clean_armed_s": ev["clean_armed_s"],
                       "trigger_sensor": trig, "iss_alert_s": iev["alert_s"],
                       "iss_fa_events": iev["fa_events"]}
                # gated stage 2: only ticks with an active alarm and a full window
                gw = tw_all.loc[t_a[alarm]]
                gw = gw[gw["rows"] >= MIN_ROWS]
                for pol in args.policies:
                    r = {**row, "policy": pol, "label": labels.get(tid),
                         "labelled": tid in policy_sets[pol], "gated_ticks": len(gw)}
                    if len(gw):
                        p = s2[pol].predict_proba(gw[WIN_FEATS].to_numpy())
                        cls = s2[pol].classes_
                        z = tick_zone(gw["t_rel"].to_numpy())
                        tick_rows.append(pd.DataFrame({
                            "baseline": baseline, "policy": pol, "n_persist": n,
                            "test_id": tid, "t_rel": gw["t_rel"].to_numpy(), "zone": z,
                            "pred": cls[p.argmax(1)], "truth": labels.get(tid),
                            "labelled": tid in policy_sets[pol]}))
                        pdf = pd.DataFrame(p, columns=cls)
                        t_first = (t_a[ev["first_idx"]] if ev["first_idx"] is not None
                                   else None)
                        r["pred_first_alert"] = (
                            cls[p[gw.index.get_loc(t_first)].argmax()]
                            if t_first is not None and t_first in gw.index else None)
                        for name, sel in (("steady_state", z == "steady_state"),
                                          ("until_600s", (z == "transient")
                                           | (z == "steady_state"))):
                            r[f"pred_{name}"] = (pdf[sel].mean().idxmax() if sel.any()
                                                 else None)
                    per_test.append(r)
            if baseline == "arming":
                fold_models[tid] = (s1, s2)
            print(f"  [{baseline}] fold {k + 1}/{len(folds)} test {tid}: "
                  f"{time.time() - fs:.1f}s", flush=True)

    res = pd.DataFrame(per_test)
    ticks_g = pd.concat(tick_rows, ignore_index=True)
    orw = pd.concat(oracle_win, ignore_index=True)

    # ---------------------------------------------- streaming equivalence + latency
    lat_rows, eq = [], []
    if "arming" in args.baselines and args.stream_tests:
        pol = "psi101" if "psi101" in args.policies else args.policies[0]
        chosen = folds[:: max(1, len(folds) // args.stream_tests)][: args.stream_tests]
        for tid in chosen:
            s1, s2 = fold_models[tid]
            for g in s1["models"].values():
                g.set_params(n_jobs=1)
            clf = s2[pol]
            clf[-1].set_params(n_jobs=1)
            raw = df[df["test_id"] == tid].reset_index(drop=True)
            te = ticks_a[ticks_a["test_id"] == tid].reset_index(drop=True)
            armed = te["armed"].to_numpy()
            b_fused = np.full(len(te), np.nan)
            b_fused[armed] = score_stage1(s1, te[armed])
            b_alarm = np.nan_to_num(sustained(te["test_id"].to_numpy(), te["Time"].to_numpy(),
                                              b_fused, MAIN_N), nan=-np.inf) > s1["thr"][MAIN_N]
            wt = wins_a[wins_a["test_id"] == tid].set_index("Time")
            scale = wt.loc[wt["rows"] >= MIN_ROWS, WIN_FEATS].abs().max().clip(lower=1.0)
            scale = scale.fillna(1.0).to_numpy()
            st = CausalStreamer(s1, clf, MAIN_N)
            s_fused, s_alarm, dmax, fmax, nan_mis, same_cls, n2 = [], [], 0.0, 0.0, 0, 0, 0
            for i, r in enumerate(raw[["Time"] + CHANNELS].itertuples(index=False)):
                t0 = time.perf_counter()
                out = st.step(int(r[0]), np.array(r[1:], dtype=float))
                ms = (time.perf_counter() - t0) * 1e3
                if not out["armed"]:
                    s_fused.append(np.nan)
                    s_alarm.append(False)
                    continue
                s_fused.append(out["fused"])
                s_alarm.append(out["alarm"])
                ran2 = out["proba"] is not None
                if ran2 and wt.loc[int(r[0]), "rows"] >= MIN_ROWS:
                    bw = wt.loc[[int(r[0])], WIN_FEATS].to_numpy()
                    bp = clf.predict_proba(bw)[0]
                    dmax = max(dmax, float(np.abs(bp - out["proba"]).max()))
                    nan_mis += int(np.sum(np.isnan(bw) != np.isnan(out["window"])))
                    fmax = max(fmax, float(np.nanmax(np.abs(bw - out["window"]) / scale)))
                    same_cls += int(bp.argmax() == out["proba"].argmax())
                    n2 += 1
                lat_rows.append({"test_id": int(tid), "t_rel": int(te["t_rel"].iloc[i]),
                                 "stage2_ran": ran2, "latency_ms": ms})
            s_fused, s_alarm = np.array(s_fused), np.array(s_alarm)
            fdiff = float(np.nanmax(np.abs(s_fused - b_fused)))
            agree = float(np.mean(s_alarm[armed] == b_alarm[armed]))
            eq.append({"test_id": int(tid), "max_abs_fused_score_diff": fdiff,
                       "alarm_agreement": agree, "stage2_ticks_compared": n2,
                       "max_scaled_window_feature_diff": fmax,
                       "window_feature_nan_mismatches": nan_mis,
                       "max_abs_stage2_proba_diff": dmax,
                       "stage2_argmax_agreement": same_cls / n2 if n2 else None})
            print(f"  streamed test {tid}: |dscore|max={fdiff:.2e} alarm agree={agree:.4f} "
                  f"|dfeat|/scale max={fmax:.1e} |dproba|max={dmax:.3f}", flush=True)
        checks["streaming_vs_batch"] = eq
        checks["streaming_note"] = (
            "Window feature differences are scaled by each feature's largest |value| in "
            "the test (min 1). They are pandas rolling round-off: the online add/remove "
            "algorithm leaves a residue of about 1e-7 of the channel magnitude where the "
            "exact value is 0 (e.g. std of a constant channel), which can move a few RF "
            "trees across near-zero split thresholds.")
    lat = pd.DataFrame(lat_rows)

    # ------------------------------------------------------------- metrics
    metrics = build_metrics(args, res, ticks_g, orw, lat, checks, policy_sets, aged,
                            folds, time.time() - t_start)
    res.to_csv(OUT / "chained_loto_per_test.csv", index=False)
    if len(lat):
        lat.to_csv(OUT / "chained_loto_latency_ms.csv", index=False)
    (OUT / "chained_loto_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                   encoding="utf-8")
    (OUT / "chained_loto_summary.md").write_text(render_markdown(metrics), encoding="utf-8")
    print(render_markdown(metrics))
    print(f"Saved results/ in {time.time() - t_start:.0f}s")
    assert all(e["alarm_agreement"] >= 0.999 and e["window_feature_nan_mismatches"] == 0
               and e["max_scaled_window_feature_diff"] < 1e-5 for e in eq), \
        "streaming implementation disagrees with batch evaluation"


def pct(s: pd.Series, q: float):
    return float(s.quantile(q)) if len(s) else None


def build_metrics(args, res, ticks_g, orw, lat, checks, policy_sets, aged, folds,
                  runtime_s) -> dict:
    m: dict = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(runtime_s, 1),
            "folds_evaluated": len(folds), "loto_complete": len(folds) == 41,
            "seed": SEED, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu": cpu_name(), "inputs_sha256": {
                PACKETS.name: sha256(PACKETS), MANIFEST.name: sha256(MANIFEST)},
            "config": {"arm_s": ARM_S, "persistence_ticks": list(PERSISTENCE),
                       "main_persistence": MAIN_N, "threshold_quantile": Q,
                       "ref_quantile": AN["REF_Q"], "inner_folds": AN["INNER_FOLDS"],
                       "train_clean_secs": AN["TRAIN_CLEAN_SECS"],
                       "if_params": AN["IF_PARAMS"], "rf_params": CL["RF_PARAMS"],
                       "stage2_imputer": "median, fit per fold",
                       "window_s": WIN, "stride_s": STRIDE, "t_max_s": T_MAX,
                       "decide_from_s": DECIDE_FROM, "min_rows": MIN_ROWS,
                       "classes": CLASSES,
                       "zones": {"transient": f"window start in [0, {DECIDE_FROM}) s",
                                 "steady_state": f"window start >= {DECIDE_FROM} s and "
                                                 f"end <= {LAST_WIN_END} s",
                                 "late": f"window end > {LAST_WIN_END} s",
                                 "false_alarm": "tick before heater-on"}},
            "label_policies": {k: sorted(int(x) for x in v) for k, v in policy_sets.items()},
            "tests_primary_aging_over_600s": sorted(int(x) for x in aged),
        },
        "leakage_checks": checks,
        "stage1": {}, "oracle_stage2": {}, "chained": {}, "latency_ms": {},
        "repo_reference": repo_reference(),
    }
    for (b, n), g in res[res["policy"] == res["policy"].iloc[0]].groupby(
            ["baseline", "n_persist"]):
        key = f"{b}/n{n}"
        m["stage1"][key] = {
            "per_sensor_iforest": summarise_stage1(g, ""),
            "iss_scatter_recalibrated": summarise_stage1(g, "iss_"),
            "trigger_sensor_counts": g["trigger_sensor"].value_counts().to_dict(),
            "median_lead_vs_iss_s": float((g["iss_alert_s"] - g["alert_s"]).median()),
        }
    for (b, pol), g in orw.groupby(["baseline", "policy"]):
        cls = [c for c in CLASSES if c in g.columns]
        g = g.assign(pred=g[cls].to_numpy().argmax(1))
        g["pred"] = np.array(cls)[g["pred"]]
        zone = g["win_start"] >= DECIDE_FROM
        tl = g[zone].groupby("test_id")
        tp = tl[cls].mean().idxmax(axis=1)
        truth = tl["truth"].first()
        m["oracle_stage2"][f"{b}/{pol}"] = {
            "test_level_windows_120_600s": class_metrics(truth, tp),
            "test_level_excluding_aged_runs": class_metrics(
                truth[~truth.index.isin(aged)], tp[~tp.index.isin(aged)]),
            "window_level_120_600s": class_metrics(g.loc[zone, "truth"], g.loc[zone, "pred"]),
            "window_level_0_120s": class_metrics(g.loc[~zone, "truth"], g.loc[~zone, "pred"]),
        }
    for (b, pol, n), g in res.groupby(["baseline", "policy", "n_persist"]):
        lab = g[g["labelled"]]
        tk = ticks_g[(ticks_g["baseline"] == b) & (ticks_g["policy"] == pol)
                     & (ticks_g["n_persist"] == n)]
        tkl = tk[tk["labelled"]]
        entry = {"labelled_tests": int(len(lab)),
                 "detected_labelled": int(lab["alert_s"].notna().sum())}
        for d in ("first_alert", "until_600s", "steady_state"):
            col = f"pred_{d}"
            ok = lab[col].notna() if col in lab else pd.Series(False, index=lab.index)
            entry[f"test_level_{d}"] = class_metrics(lab.loc[ok, "label"], lab.loc[ok, col])
            entry[f"test_level_{d}"]["tests_without_prediction"] = int((~ok).sum())
            entry[f"detected_and_identified_{d}"] = (
                f"{int((lab[col] == lab['label']).sum()) if col in lab else 0}/{len(lab)}")
        na = lab[~lab["test_id"].isin(aged)]
        if "pred_steady_state" in na:
            ok = na["pred_steady_state"].notna()
            entry["test_level_steady_state_excluding_aged_runs"] = class_metrics(
                na.loc[ok, "label"], na.loc[ok, "pred_steady_state"])
        zones = {}
        for z in ("transient", "steady_state", "late"):
            zz = tkl[tkl["zone"] == z]
            zones[z] = class_metrics(zz["truth"], zz["pred"])
            zones[z]["errors"] = int(len(zz) - zones[z]["correct"])
        errs = sum(zones[z]["errors"] for z in ("transient", "steady_state"))
        for z in ("transient", "steady_state"):
            zones[z]["share_of_errors_until_600s"] = (zones[z]["errors"] / errs) if errs else None
        zones["false_alarm_gated_ticks"] = int((tk["zone"] == "false_alarm").sum())
        zones["false_alarm_predictions"] = tk.loc[tk["zone"] == "false_alarm",
                                                  "pred"].value_counts().to_dict()
        entry["gated_tick_level"] = zones
        if pol == "repo":
            unl = g[~g["labelled"]]
            entry["unlabelled_tests_chained_hint"] = {
                int(r.test_id): r.pred_steady_state for r in unl.itertuples()
                if hasattr(r, "pred_steady_state")}
        m["chained"][f"{b}/{pol}/n{n}"] = entry
    if len(lat):
        for name, sel in (("all_armed_ticks", lat["latency_ms"].notna()),
                          ("stage1_only", ~lat["stage2_ran"]),
                          ("stage1_plus_stage2", lat["stage2_ran"])):
            s = lat.loc[sel, "latency_ms"]
            m["latency_ms"][name] = {"n_ticks": int(len(s)),
                                     "mean": float(s.mean()) if len(s) else None,
                                     "p50": pct(s, 0.5), "p99": pct(s, 0.99),
                                     "max": float(s.max()) if len(s) else None}
        m["latency_ms"]["streamed_tests"] = sorted(int(x) for x in lat["test_id"].unique())
        m["latency_ms"]["threads"] = 1
        m["latency_ms"]["budget_ms_at_1hz"] = 1000
    return m


def repo_reference() -> dict:
    """Numbers the repo currently reports (for side-by-side tables)."""
    out = {}
    if REPO_ANOMALY_SWEEP.exists():
        s = pd.read_csv(REPO_ANOMALY_SWEEP)
        r = s[(s["q"] == 1.0) & (s["detector"] == "per_sensor")].iloc[0]
        out["stage1_per_sensor_8fold_heater_baseline"] = {
            "detected": f"{int(r['detected'])}/{int(r['tests'])}",
            "latency_median_s": float(r["delay_median_s"]),
            "latency_p90_s": float(r["delay_p90_s"]),
            "far_per_clean_hour": float(r["fa_per_hr"]),
            "iss_latency_median_s": float(r["iss_delay_median_s"]),
            "iss_far_per_clean_hour": float(r["iss_fa_per_hr"])}
    if REPO_CLASSIFIER_TESTS.exists():
        t = pd.read_csv(REPO_CLASSIFIER_TESTS)
        for mdl in t["model"].unique():
            x = t[t["model"] == mdl]
            out[f"stage2_{mdl}_loto38_native_nan"] = class_metrics(x["material"], x["predicted"])
    return out


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# --------------------------------------------------------------- markdown
def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(m: dict) -> str:
    meta = m["meta"]
    L = ["# Chained causal LOTO evaluation (SAME, 1 Hz)", "",
         f"Generated {meta['generated_utc']}; {meta['folds_evaluated']} LOTO folds "
         f"(complete: {meta['loto_complete']}); seed {meta['seed']}; "
         f"scikit-learn {meta['scikit_learn']}; runtime {meta['runtime_s']} s.", "",
         f"Baselines: `arming` = causal, first {meta['config']['arm_s']} s after power-on "
         "(primary); `heater_oracle` = repo baseline (last 120 s before heater-on; needs "
         "the event time, so not deployable). Thresholds at q = "
         f"{meta['config']['threshold_quantile']} on training-fold clean air.", "",
         "## Detection (Stage 1 = end-to-end trigger)", "",
         "| Baseline | N ticks | Detected | Median latency (s) | P90 (s) | FAR / clean h "
         "| ISS scatter detected | ISS median (s) | ISS FAR / h |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in m["stage1"].items():
        b, n = k.split("/")
        a, i = v["per_sensor_iforest"], v["iss_scatter_recalibrated"]
        L.append(f"| {b} | {n[1:]} | {a['detection_rate']} | {fmt(a['latency_median_s'])} | "
                 f"{fmt(a['latency_p90_s'])} | {fmt(a['far_per_clean_hour'], 2)} | "
                 f"{i['detection_rate']} | {fmt(i['latency_median_s'])} | "
                 f"{fmt(i['far_per_clean_hour'], 2)} |")
    ref = m["repo_reference"].get("stage1_per_sensor_8fold_heater_baseline")
    if ref:
        L += ["", f"Repo-reported standalone Stage 1 (8-fold, heater baseline, N=3): "
              f"{ref['detected']} detected, median {ref['latency_median_s']} s, "
              f"P90 {ref['latency_p90_s']} s, FAR {ref['far_per_clean_hour']:.2f}/h; "
              f"ISS {ref['iss_latency_median_s']} s, {ref['iss_far_per_clean_hour']:.2f}/h."]
    L += ["", "## Material identification", "",
          "Oracle Stage 2 = heater-aligned windows starting 120-570 s, probabilities "
          "averaged per test (repo protocol). Chained = Stage 2 run only on ticks with an "
          "active Stage 1 alarm.", "",
          "| Baseline / labels | Oracle acc | Oracle macro-F1 | Oracle acc excl. 720 s-aged |",
          "|---|---|---|---|"]
    for k, v in m["oracle_stage2"].items():
        a = v["test_level_windows_120_600s"]
        e = v["test_level_excluding_aged_runs"]
        L.append(f"| {k} | {fmt(a['accuracy'], 3)} ({a['correct']}/{a['n']}) | "
                 f"{fmt(a['macro_f1'], 3)} | {fmt(e['accuracy'], 3)} ({e['correct']}/{e['n']}) |")
    L += ["", "| Baseline / labels / N | First-alert acc | Until-600 s acc | Steady-state acc "
          "| Steady macro-F1 | Steady acc excl. aged | Tick acc transient | Tick acc steady "
          "| Share of errors in transient |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in m["chained"].items():
        f, u, s = (v["test_level_first_alert"], v["test_level_until_600s"],
                   v["test_level_steady_state"])
        ex = v.get("test_level_steady_state_excluding_aged_runs", {})
        z = v["gated_tick_level"]
        L.append(f"| {k} | {fmt(f['accuracy'], 3)} ({f['correct']}/{f['n']}) | "
                 f"{fmt(u['accuracy'], 3)} | {fmt(s['accuracy'], 3)} ({s['correct']}/{s['n']}) | "
                 f"{fmt(s['macro_f1'], 3)} | {fmt(ex.get('accuracy'), 3)} | "
                 f"{fmt(z['transient']['accuracy'], 3)} | {fmt(z['steady_state']['accuracy'], 3)} "
                 f"| {fmt(z['transient']['share_of_errors_until_600s'], 2)} |")
    if m["latency_ms"]:
        L += ["", "## Per-tick CPU latency (single thread, streaming implementation)", "",
              f"CPU: {meta['cpu']}. Streamed tests: {m['latency_ms']['streamed_tests']}.", "",
              "| Ticks | n | Mean (ms) | P50 (ms) | P99 (ms) | Max (ms) |",
              "|---|---|---|---|---|---|"]
        for k in ("all_armed_ticks", "stage1_only", "stage1_plus_stage2"):
            v = m["latency_ms"][k]
            L.append(f"| {k} | {v['n_ticks']} | {fmt(v['mean'], 2)} | {fmt(v['p50'], 2)} | "
                     f"{fmt(v['p99'], 2)} | {fmt(v['max'], 2)} |")
    c = m["leakage_checks"]
    L += ["", "## Leakage self-checks", "",
          f"- Future perturbation (features at or before the cut bit-identical): tests "
          f"{c.get('future_perturbation_tests_bit_identical')}."]
    for e in c.get("streaming_vs_batch", []):
        L.append(f"- Streaming vs batch, test {e['test_id']}: max |score diff| "
                 f"{e['max_abs_fused_score_diff']:.1e}, alarm agreement "
                 f"{e['alarm_agreement']:.4f}, max scaled window feature diff "
                 f"{e['max_scaled_window_feature_diff']:.1e}, max |proba diff| "
                 f"{e['max_abs_stage2_proba_diff']:.3f}, Stage 2 argmax agreement "
                 f"{fmt(e['stage2_argmax_agreement'], 4)}.")
    if c.get("streaming_note"):
        L.append(f"- {c['streaming_note']}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all 41 (default)")
    ap.add_argument("--test-ids", type=int, nargs="+", help="evaluate only these folds")
    ap.add_argument("--stream-tests", type=int, default=4,
                    help="tests replayed tick-by-tick for equivalence + latency")
    ap.add_argument("--n-jobs", type=int, default=-1)
    seed_everything()
    run(ap.parse_args())


if __name__ == "__main__":
    main()
