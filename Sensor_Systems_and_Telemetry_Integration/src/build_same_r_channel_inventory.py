"""Build ml_lead_handoff/SAME_R_CHANNEL_INVENTORY.csv and SAME_R_DUSTTRAK_CHECK.json.

The channel inventory lists every column exactly as named in the original SAME-R
HKP housekeeping CSVs. Bracketed tags such as [AT08] or [PN23] are copied
verbatim from the source headers; their meaning is NOT interpreted here.
Statistics cover all 43 raw files as supplied (before the ML loader's
duplicate/drop rules).

Usage (from the package root):
    python src/build_same_r_channel_inventory.py
"""
from pathlib import Path
import csv, json, re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "ml_lead_handoff" / "raw_same_r_hkp"
ADAPTER = ROOT / "ml_lead_handoff" / "ML_LEAD_SAME_R_ADAPTER.json"
OUT_INV = ROOT / "ml_lead_handoff" / "SAME_R_CHANNEL_INVENTORY.csv"
OUT_DT = ROOT / "ml_lead_handoff" / "SAME_R_DUSTTRAK_CHECK.json"

HEADER_RE = re.compile(r"^(?P<base>[^\[\(]+?)(?:\[(?P<tag>[^\]]+)\])?(?:\((?P<unit>[^\)]+)\))?$")


def load_raw():
    files = sorted(RAW.glob("*/*.csv"))
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype=str)
        d["__file"] = f.name
        d["__material_folder"] = f.parent.name
        frames.append(d)
    return files, frames


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def build():
    files, frames = load_raw()
    headers = list(frames[0].columns.drop(["__file", "__material_folder"]))
    for fr in frames[1:]:
        h = list(fr.columns.drop(["__file", "__material_folder"]))
        if h != headers:
            raise SystemExit("SAME-R raw files do not share one header; inventory would be ambiguous")
    data = pd.concat(frames, ignore_index=True)
    model_map = json.loads(ADAPTER.read_text(encoding="utf-8"))["model_channels"]

    rows = []
    for i, col in enumerate(headers, start=1):
        m = HEADER_RE.match(col)
        base, tag, unit = (m.group("base"), m.group("tag"), m.group("unit")) if m else (col, None, None)
        raw = data[col]
        vals = _num(raw)
        numeric = vals.dropna()
        distinct = numeric.unique()
        if numeric.empty:
            kind = "non_numeric"
        elif col in ("Time",):
            kind = "time_epoch_s"
        elif set(np.round(distinct, 9)) <= {0.0, 1.0}:
            kind = "binary_flag"
        elif unit:
            kind = "measurement"
        elif len(distinct) <= 20 and np.all(np.mod(distinct, 1) == 0):
            kind = "state_or_counter"
        else:
            kind = "numeric_other"
        rows.append({
            "column_index": i,
            "source_column": col,
            "base_name": base.strip(),
            "bracket_tag_verbatim": tag or "",
            "unit_from_header": unit or "",
            "value_kind": kind,
            "numeric_count": int(numeric.size),
            "non_numeric_nonblank_count": int(raw.notna().sum() - numeric.size),
            "min": "" if numeric.empty else float(numeric.min()),
            "max": "" if numeric.empty else float(numeric.max()),
            "distinct_values": int(len(distinct)),
            "ml_model_input_key": model_map.get(col, ""),
        })

    with open(OUT_INV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # DustTrak consistency check (data-internal only; not a calibration).
    def channel_check(x):
        meas = _num(data[f"DustTrak{x}Measurement(MgPerM3)"])
        volts = _num(data[f"DustTrak{x}AnalogMeasurement(Volts)"])
        power = _num(data[f"DustTrak{x}Power"])
        state = _num(data[f"DustTrak{x}State"])
        ok = meas.notna() & volts.notna() & (power == 1) & (state == 6)
        sig = ok & (volts > 0.05) & (volts < 4.99)
        ratio = (meas[sig] / volts[sig])
        return {
            "rows_numeric": int(meas.notna().sum()),
            "rows_powered_state6": int(ok.sum()),
            "pearson_r_reading_vs_analog_volts": round(float(np.corrcoef(meas[ok], volts[ok])[0, 1]), 4),
            "reading_per_volt_median": round(float(ratio.median()), 3) if not ratio.empty else None,
            "reading_per_volt_iqr": [round(float(q), 3) for q in ratio.quantile([0.25, 0.75])] if not ratio.empty else None,
            "rows_reading_when_powered_off_nonzero": int(((power == 0) & (meas != 0) & meas.notna()).sum()),
            "rows_analog_at_or_below_minus_9_9_V": int((volts <= -9.9).sum()),
            "rows_analog_at_or_above_4_99_V": int((volts >= 4.99).sum()),
            "rows_negative_reading": int((meas < 0).sum()),
            "reading_min": float(meas.min()), "reading_median": float(meas.median()), "reading_max": float(meas.max()),
        }

    a, b = channel_check("A"), channel_check("B")
    amod = (_num(data["DustTrakAMeasurement(MgPerM3)"]).dropna())
    report = {
        "scope": "Internal-consistency check of DustTrak channels across all 43 SAME-R raw files. This is not a calibration.",
        "DustTrakB": b,
        "DustTrakA": {**a, "share_of_readings_at_or_above_10000": round(float((amod >= 10000).mean()), 4)},
        "findings": [
            "DustTrak B reading tracks its own analog output closely (see pearson_r), at a steady ~20 mg/m3 per volt, i.e. a 0-5 V analog output spanning 0-100 mg/m3. Reading is 0 whenever the unit is powered off. The channel is internally consistent and operating.",
            "DustTrak B analog output clips at 5 V (about 100 mg/m3); rows at or above 4.99 V should be treated as saturated on the analog side. Rows at -10 V analog are faults.",
            "DustTrak A readings are mostly coded values of 10000 and above that do not track its analog output; they are not usable as mass concentrations. DustTrak A is not an ML model input.",
            "Calibration for smoke is NOT established by this check. Photometric mass monitors report mass relative to their factory calibration aerosol; NIST (Cleary) calibrated the SAME instruments, and smoke-specific factors are in the SAME publications (Yang et al. 2015, Aerosol Sci. Technol. 49; Meyer et al. 2018, Fire Safety J.). Until those factors are applied, present dtb as a relative mass-monitor reading, not a calibrated smoke concentration."
        ],
    }
    OUT_DT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rows, report


if __name__ == "__main__":
    rows, report = build()
    tagged = sum(1 for r in rows if r["bracket_tag_verbatim"])
    print(json.dumps({"channels": len(rows), "with_bracket_tag": tagged,
                      "ml_inputs": sum(1 for r in rows if r["ml_model_input_key"]),
                      "dusttrak_b_r": report["DustTrakB"]["pearson_r_reading_vs_analog_volts"],
                      "dusttrak_b_mg_per_volt": report["DustTrakB"]["reading_per_volt_median"]}, indent=2))
