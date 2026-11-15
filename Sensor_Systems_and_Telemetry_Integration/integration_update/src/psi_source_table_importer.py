from __future__ import annotations
import csv, json, re, hashlib
from pathlib import Path
from typing import Any

MISSING = {"", "na", "n/a", "none", "not applicable", "not_applicable"}
SENTINELS = {"or", "over range", "over-range", "?", "-999", "-1.#ind", "#value!"}
NUMBER = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z°%²³/.-]+)?$")

def decode_csv(path: Path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "utf-8"):
        try:
            return raw, raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return raw, raw.decode("cp1252", errors="replace"), "cp1252-replace"

def cell_status(value: Any) -> str:
    s = "" if value is None else str(value).strip()
    if s.lower() in MISSING: return "missing"
    if s.lower() in SENTINELS: return "sentinel_or_unresolved"
    return "present"

def parse_scalar(value: Any):
    s = "" if value is None else str(value).strip()
    m = NUMBER.fullmatch(s)
    if not m: return None
    return float(m.group(1)), (m.group(2) or "")

def si_convert(value: float, unit: str, accession: str, column: str):
    u = unit.lower().replace(" ", "")
    if accession == "PSI-69" and column.lower().startswith("burning rate"):
        return value * 1e-6, "m²/s", "mm²/s to m²/s (unit correction confirmed in supplied team update)"
    rules = {
        "mmhg": (133.322387415, "Pa", "mmHg to Pa"),
        "atm": (101325.0, "Pa", "atm to Pa"),
        "bar": (100000.0, "Pa", "bar to Pa"),
        "cm": (0.01, "m", "cm to m"),
        "mm": (0.001, "m", "mm to m"),
        "cm/s": (0.01, "m/s", "cm/s to m/s"),
        "mm/s": (0.001, "m/s", "mm/s to m/s"),
        "mg/s": (1e-6, "kg/s", "mg/s to kg/s"),
        "mg/min": (1e-6/60, "kg/s", "mg/min to kg/s"),
        "%": (0.01, "fraction", "percent to fraction"),
        "percent": (0.01, "fraction", "percent to fraction"),
    }
    if u in rules:
        factor, si_unit, note = rules[u]
        return value * factor, si_unit, note
    if u in {"°c", "degc", "c"}:
        return value + 273.15, "K", "°C to K"
    return None, None, None

def header_unit(header: str, accession: str):
    if accession == "PSI-69" and header.lower().startswith("burning rate"):
        return "mm²/s"
    m = re.search(r"\(([^()]*)\)", header)
    if m: return m.group(1).strip()
    if ";" in header: return header.split(";", 1)[1].strip()
    return ""

def import_source_table(path: str | Path, accession: str, source_url: str | None = None):
    path = Path(path)
    raw, text, encoding = decode_csv(path)
    sha = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(text.splitlines())
    headers = [str(x).strip() for x in (reader.fieldnames or [])]
    packets, numeric_candidates = [], []
    for source_row, row in enumerate(reader, start=2):
        row = {str(k).strip(): ("" if v is None else str(v)) for k, v in row.items() if k is not None}
        if not any(str(v).strip() for v in row.values()):
            continue
        statuses = {k: cell_status(v) for k, v in row.items()}
        packet = {
            "packet_id": f"{accession}::{path.name}::row-{source_row}",
            "accession": accession, "source_file": path.name,
            "source_row_number_1based": source_row,
            "timestamp_utc": None,
            "timestamp_status": "not_provided_at_sample_level",
            "source_url": source_url or f"https://psi.nasa.gov/physci/repo/data/investigations/{accession}",
            "source_sha256": sha, "source_fields": row,
            "field_status": statuses,
            "packet_semantics": "experiment_or_source_row_not_live_telemetry"
        }
        packets.append(packet)
        for column, raw_value in row.items():
            parsed = parse_scalar(raw_value)
            if parsed is None: continue
            value, cell_unit = parsed
            unit = cell_unit or header_unit(column, accession)
            si_value, si_unit, conversion = si_convert(value, unit, accession, column)
            numeric_candidates.append({
                "accession": accession, "source_file": path.name,
                "source_row_number_1based": source_row,
                "source_column": column, "raw_value": raw_value,
                "numeric_value": value, "original_unit": unit,
                "si_value": si_value, "si_unit": si_unit,
                "conversion_rule": conversion,
                "classification": "numeric_source_candidate_not_verified_sensor",
                "source_sha256": sha
            })
    return packets, numeric_candidates, {"encoding":encoding, "sha256":sha, "columns":len(headers), "packets":len(packets)}

def import_directory(folder: str | Path, accession_by_filename: dict[str,str]):
    all_packets, all_numeric = [], []
    for path in sorted(Path(folder).glob("*.csv")):
        accession = accession_by_filename[path.name]
        packets, numeric, _ = import_source_table(path, accession)
        all_packets.extend(packets); all_numeric.extend(numeric)
    return all_packets, all_numeric

def export_jsonl(records, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record, ensure_ascii=False) + "\n")

def export_csv(records, path: str | Path):
    if not records: return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
