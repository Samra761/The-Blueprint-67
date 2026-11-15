#!/usr/bin/env python3
"""Build PSI-100 SAFFIRE-III telemetry packets from the original workbook.

Run from the Sensor_Systems_and_Telemetry_Integration root:
    python src/build_psi100_saffire_iii_packets.py
"""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/psi100/PSI-100_Raw_Data_Saffire-III_1_Hz_Telemetry_Data.xlsx"
OUT = ROOT / "data/processed/saffire_iii_psi100"
TIME_COLUMNS = {
    "RelativeSeconds", "CygnusTimestamp", "SaffireTimestamp",
    "Cyg-Saf TS (ms)", "Saffire Repeat?", "Row to Row Delta T (ms)"
}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def finite_number(value: Any):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source workbook: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    workbook = load_workbook(SOURCE, read_only=True, data_only=True)
    if "RawData" not in workbook.sheetnames:
        raise ValueError("Expected RawData worksheet is absent")
    ws = workbook["RawData"]
    headers = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    channel_headers = [h for h in headers if h is not None and h not in TIME_COLUMNS]
    registry = []
    for header in channel_headers:
        match = re.search(r"\(([^()]*)\)", str(header))
        unit = match.group(1).strip() if match else None
        registry.append({
            "variable_id": "SAFFIRE3::" + slug(str(header)),
            "source_column": str(header),
            "display_name": str(header),
            "unit": unit,
            "measurement_class": "telemetry_channel",
            "is_instrument_sensor": False,
            "instrument_identity_status": (
                "header-level channel name only; exact hardware identity and calibration not verified"
            ),
            "unit_status": "explicit_in_source_header" if unit else "not_stated_in_source_header",
            "source_sheet": "RawData",
            "source_file": SOURCE.name,
        })
    registry_obj = {"registry_version": "1.0.0", "source": "PSI-100 RawData worksheet", "variables": registry}
    (OUT / "channel_registry.json").write_text(json.dumps(registry_obj, indent=2) + "\n", encoding="utf-8")

    jsonl_path = OUT / "PSI100_RawData_packets.jsonl"
    csv_path = OUT / "PSI100_RawData_normalized.csv"
    fields = [
        "packet_id", "experiment_id", "psi_id", "source_sheet", "source_file", "source_row",
        "timestamp_utc", "timestamp_source", "source_epoch_ms", "relative_seconds_source",
        "saffire_timestamp_ms", "cyg_saf_timestamp_ms", "saffire_repeat_source",
        "row_to_row_delta_t_ms", "quality_flags",
    ] + [v["variable_id"] for v in registry]

    packets = timestamped = untimestamped = invalid_rows = duplicate_timestamps = 0
    seen = set()
    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for source_row, values in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            row = dict(zip(headers, values))
            if not any(value is not None for value in values):
                continue
            epoch_raw = row.get("CygnusTimestamp")
            try:
                epoch_ms = float(epoch_raw)
            except (TypeError, ValueError):
                epoch_ms = float("nan")
            flags = []
            timestamp = None
            if math.isfinite(epoch_ms) and 1e12 <= epoch_ms <= 2e12:
                timestamp = datetime.fromtimestamp(epoch_ms / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                timestamped += 1
                if epoch_ms in seen:
                    duplicate_timestamps += 1
                seen.add(epoch_ms)
            else:
                flags.append("invalid_cygnus_timestamp_source_value")
                invalid_rows += 1
                untimestamped += 1

            channels = {
                variable["variable_id"]: finite_number(row.get(variable["source_column"]))
                for variable in registry
            }
            packet = {
                "schema_version": "1.0",
                "packet_id": f"PSI100-RawData-{packets + 1:05d}",
                "experiment_id": "SAFFIRE-III",
                "psi_id": "PSI-100",
                "source_sheet": "RawData",
                "source_file": SOURCE.name,
                "source_row": source_row,
                "timestamp_utc": timestamp,
                "timestamp_source": "CygnusTimestamp_epoch_milliseconds" if timestamp else None,
                "time_basis": "source_epoch_milliseconds_to_utc" if timestamp else "invalid_or_missing_source_timestamp",
                "source_epoch_ms": finite_number(epoch_raw),
                "relative_seconds_source": finite_number(row.get("RelativeSeconds")),
                "saffire_timestamp_ms": finite_number(row.get("SaffireTimestamp")),
                "cyg_saf_timestamp_ms": finite_number(row.get("Cyg-Saf TS (ms)")),
                "saffire_repeat_source": row.get("Saffire Repeat?"),
                "row_to_row_delta_t_ms": finite_number(row.get("Row to Row Delta T (ms)")),
                "channels": channels,
                "quality_flags": flags,
            }
            jf.write(json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n")
            csv_row = {
                "packet_id": packet["packet_id"], "experiment_id": packet["experiment_id"],
                "psi_id": packet["psi_id"], "source_sheet": packet["source_sheet"],
                "source_file": packet["source_file"], "source_row": source_row,
                "timestamp_utc": timestamp, "timestamp_source": packet["timestamp_source"],
                "source_epoch_ms": packet["source_epoch_ms"],
                "relative_seconds_source": packet["relative_seconds_source"],
                "saffire_timestamp_ms": packet["saffire_timestamp_ms"],
                "cyg_saf_timestamp_ms": packet["cyg_saf_timestamp_ms"],
                "saffire_repeat_source": packet["saffire_repeat_source"],
                "row_to_row_delta_t_ms": packet["row_to_row_delta_t_ms"],
                "quality_flags": ";".join(flags),
            }
            csv_row.update(channels)
            writer.writerow(csv_row)
            packets += 1

    summary = {
        "dataset": "PSI-100 SAFFIRE-III RawData telemetry stream",
        "source_file": SOURCE.name,
        "source_sheet": "RawData",
        "packets": packets,
        "channels": len(registry),
        "timestamped_packets": timestamped,
        "untimestamped_packets": untimestamped,
        "unique_valid_source_timestamps": len(seen),
        "duplicate_timestamp_rows": duplicate_timestamps,
        "invalid_timestamp_rows_preserved_and_flagged": invalid_rows,
        "timestamp_rule": "CygnusTimestamp is treated as Unix epoch milliseconds only within the plausible epoch-ms range; invalid source values remain preserved and timestamp_utc is null.",
        "unit_rule": "Units are assigned only when explicitly stated in the source header; otherwise null.",
        "identity_rule": "Source channel labels are preserved verbatim; hardware identity and calibration are not inferred.",
        "overlap_rule": "RawData is the canonical combined workbook stream. Other workbook sheets are projections/views and must not be added as independent packet counts. The detailed cleaned CSV is a corroborating export, not an additional stream.",
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
