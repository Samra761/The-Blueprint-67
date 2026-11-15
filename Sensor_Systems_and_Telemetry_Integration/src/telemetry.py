"""Wide telemetry packet builder and experiment-table importer.

One packet groups all available variables for an experiment/run at a timestamp.
If a source has no timestamp, a source-row key is used and timestamp_utc remains
null; no time value is fabricated. Variables are not automatically physical sensors.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
import pandas as pd

class TelemetryError(ValueError):
    pass

def load_registry(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    return {v["variable_id"]:v for v in obj["variables"]}

def _is_missing(v):
    return v is None or (isinstance(v,float) and math.isnan(v)) or str(v).strip().lower() in {"","na","n/a","null","none","nan"}

def read_source_table(path):
    """Read CSV, JSON (records/list/envelope), or Excel into a DataFrame."""
    path=Path(path)
    suffix=path.suffix.lower()
    if suffix==".csv": return pd.read_csv(path,dtype="object")
    if suffix in {".xlsx", ".xls"}: return pd.read_excel(path,dtype="object")
    if suffix==".json":
        obj=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj,list): records=obj
        elif isinstance(obj,dict):
            records=next((obj[k] for k in ("observations","data","records","rows") if isinstance(obj.get(k),list)),None)
            if records is None: raise TelemetryError("JSON must contain a list or an observations/data/records/rows array")
        else: raise TelemetryError("Unsupported JSON root; expected list or object envelope")
        return pd.DataFrame(records,dtype="object")
    raise TelemetryError(f"Unsupported input format: {suffix}. Use CSV, JSON, XLSX or XLS.")

def import_table(path, registry_path, experiment_col=None, record_col=None, psi_col="psi_id", value_type_col="value_type", source_col="source", unit_overrides=None):
    path=Path(path); reg=load_registry(registry_path); df=read_source_table(path)
    experiment_col=experiment_col or next((c for c in ["experiment_id","campaign","flight","platform"] if c in df),psi_col)
    record_col=record_col or next((c for c in ["record_id","source_record_id"] if c in df),None)
    unit_overrides=unit_overrides or {}; out=[]; errors=[]; stem=path.stem.removeprefix("source_").split("_")[0].upper()
    excluded=re.compile(r"(^|_)(id|no|row|index)$",re.I)
    context_fields={"air_flow_settings_cm_s","record_id","source_record_id","test_id","test_no","as_run_no","sample_no","flex_id","same_run","test_date","test_gmt","gmt_day_2009","flight","sample","campaign","platform","gravity","gravity_level","material","material_raw","material_family","fuel","psi_id","source_xlsx_row","value_type","quality_flags","source_ref","source"}
    for ix,row in df.iterrows():
        exp=row.get(experiment_col) or row.get(psi_col) or "unknown"
        rec=row.get(record_col) if record_col else None
        if _is_missing(exp):
            errors.append({"row":int(ix)+2,"field":experiment_col,"error":"missing_experiment_identifier"}); continue
        for col in df.columns:
            if excluded.search(col) or (col in context_fields and col != "air_flow_settings_cm_s") or col in {"test_date","test_gmt","is_aged_reading","use_for_particle_size_model","use_for_extinction_model","extinguished","gn2_suppression_used","vitiated_atmosphere","log_join_ok","ug_sustained_spread"}: continue
            raw=row[col]
            if _is_missing(raw): continue
            raw_text=str(raw).strip()
            # Accept a single numeric token only. Do not strip commas: values such as
            # "10, 5" are multi-step flow settings, not the number 105.
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw_text):
                continue
            try: value=float(raw_text)
            except (ValueError,TypeError): continue
            if not math.isfinite(value):
                errors.append({"row":int(ix)+2,"field":col,"error":"non_finite_numeric_value"}); continue
            vid=f"{stem}::{col}"
            if vid not in reg:
                # Numeric metadata/settings not in the canonical registry are not emitted as measurements.
                # They remain available in source_row_metadata where relevant.
                continue
            meta=reg[vid]
            out.append({"observation_id":f"{path.stem}|row{ix+2}|{rec or str(ix+2)}|{col}","source_row_key":f"{path.name}:{ix+2}",
              "experiment_id":str(exp),"psi_id":None if _is_missing(row.get(psi_col)) else str(row.get(psi_col)),
              "record_id":None if _is_missing(rec) else str(rec),"run_id":None if _is_missing(row.get("run_id")) else str(row.get("run_id")),
              "variable_id":vid,"variable_name":col,"value":value,"unit":unit_overrides.get(col,meta["unit"]),
              "timestamp_utc":None if _is_missing(row.get("timestamp_utc")) else str(row.get("timestamp_utc")),
              "elapsed_time_s":None if _is_missing(row.get("elapsed_time_s")) else row.get("elapsed_time_s"),
              "value_type":None if _is_missing(row.get(value_type_col)) else str(row.get(value_type_col)),
              "quality_flags":None if _is_missing(row.get("quality_flags")) else str(row.get("quality_flags")),
              "source_file":path.name,"source_reference":None if _is_missing(row.get(source_col)) else str(row.get(source_col)),
              "source_row_key":f"{path.name}:{ix+2}",
              "source_row_metadata":{k:str(row[k]) for k in df.columns if k in context_fields and not _is_missing(row.get(k))},
              "measurement_class":meta["measurement_class"],"is_instrument_sensor":False})
    return out,errors

def validate_observations(rows, registry):
    errors=[]; allowed={"reported_experimental_variable","experimental_setting","derived_result"}; seen=set()
    for i,r in enumerate(rows):
        for k in ["experiment_id","variable_id","variable_name","unit","source_file"]:
            if _is_missing(r.get(k)): errors.append({"row":i,"field":k,"error":"required_field_missing"})
        if r.get("variable_id") not in registry: errors.append({"row":i,"field":"variable_id","error":"unknown_variable_id"})
        if r.get("measurement_class") not in allowed: errors.append({"row":i,"field":"measurement_class","error":"invalid_class"})
        try:
            v=float(r.get("value"))
            if not math.isfinite(v): raise ValueError
        except (ValueError,TypeError): errors.append({"row":i,"field":"value","error":"value_must_be_finite_numeric"})
        if r.get("timestamp_utc") is not None:
            try: pd.to_datetime(r["timestamp_utc"],utc=True,errors="raise")
            except Exception: errors.append({"row":i,"field":"timestamp_utc","error":"invalid_timestamp"})
        key=r.get("observation_id")
        if key in seen: errors.append({"row":i,"field":"observation_id","error":"duplicate_observation_id"})
        seen.add(key)
    return {"ok":not errors,"records_checked":len(rows),"error_count":len(errors),"errors":errors}

def build_wide_packets(observations, registry=None):
    """Return (packet_envelope: Dict, wide_dataframe: DataFrame).

    Timestamped rows group by source + experiment + PSI + run + timestamp.
    Untimestamped experiment-table rows group by source_row_key (fallback:
    observation ID row token), keeping timestamp_utc=null.
    """
    groups={}
    for obs in observations:
        ts=obs.get("timestamp_utc")
        if ts:
            key=(obs.get("source_file"),obs.get("experiment_id"),obs.get("psi_id"),obs.get("run_id"),ts)
        else:
            row_key=obs.get("source_row_key")
            if not row_key:
                parts=str(obs.get("observation_id","")).split("|")
                row_key="|".join(parts[:2])
            key=(obs.get("source_file"),obs.get("experiment_id"),obs.get("psi_id"),obs.get("run_id"),"ROW:"+str(row_key))
        if key not in groups:
            groups[key]={"packet_id":None,"experiment_id":obs.get("experiment_id"),"psi_id":obs.get("psi_id"),
              "run_id":obs.get("run_id"),"timestamp_utc":ts,"source_file":obs.get("source_file"),
              "source_reference":obs.get("source_reference"),"source_row_metadata":obs.get("source_row_metadata",{}),
              "record_ids":[],"readings":{},"units":{},
              "variable_names":{},"measurement_classes":{},"reading_metadata":{},
              "time_basis":"source_timestamp" if ts else "no_timestamp_source_row"}
        p=groups[key]
        if obs.get("record_id") and obs["record_id"] not in p["record_ids"]: p["record_ids"].append(obs["record_id"])
        vid=obs["variable_id"]
        if vid in p["readings"]:
            # Keep first value and make duplicate collision explicit.
            p.setdefault("duplicate_variable_ids",[]).append(vid)
            continue
        p["readings"][vid]=float(obs["value"]); p["units"][vid]=obs.get("unit")
        p["variable_names"][vid]=obs.get("variable_name"); p["measurement_classes"][vid]=obs.get("measurement_class")
        p["reading_metadata"][vid]={"value_type":obs.get("value_type"),"quality_flags":obs.get("quality_flags"),
          "source_row_key":obs.get("source_row_key"),"source_reference":obs.get("source_reference"),
          "is_instrument_sensor":False}
    packets=[]
    for i,(key,p) in enumerate(sorted(groups.items(),key=lambda kv:tuple("" if x is None else str(x) for x in kv[0]))):
        p["packet_id"]=f"PKT-{i+1:06d}"
        p["record_ids"]=p["record_ids"] or None
        packets.append(p)
    envelope={"schema_version":"1.0","format":"wide_packet_per_timestamp","packet_count":len(packets),
      "grouping_policy":"one packet per source/experiment/PSI/run/timestamp; when timestamp is absent, one packet per source row",
      "timestamp_policy":"timestamp_utc is null when not present in source; no synthetic timestamps",
      "data_semantics":"Reported experimental variables and derived results; not asserted to be physical sensor channels.",
      "packets":packets}
    df=packets_to_dataframe(packets)
    return envelope,df

def packets_to_dataframe(packets):
    """Flatten packet metadata and variable IDs into wide DataFrame columns."""
    meta_cols=["packet_id","experiment_id","psi_id","run_id","timestamp_utc","source_file","source_reference","record_ids","time_basis"]
    variable_ids=sorted({vid for p in packets for vid in p.get("readings",{})})
    rows=[]
    for p in packets:
        row={k:p.get(k) for k in meta_cols}
        for vid in variable_ids: row[vid]=p.get("readings",{}).get(vid)
        rows.append(row)
    return pd.DataFrame(rows,columns=meta_cols+variable_ids)

def export_packets_json(packet_data, path):
    """Dump packet envelope (or packet list) as standards-compliant JSON."""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(packet_data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def export_json(rows,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
