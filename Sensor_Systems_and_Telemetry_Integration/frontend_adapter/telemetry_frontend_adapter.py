"""Frontend adapter for NASA wide packets and telemetry-channel packets.

The adapter preserves source timestamps, provenance, nulls and quality flags.
It never substitutes the host clock or invents measurements.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


_SAME_R_CHANNEL_META = {
    "iss_obs": {"name": "ISS detector obscuration", "unit": "V", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "iss_sct": {"name": "ISS detector scatter", "unit": "V", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "sts": {"name": "STS detector", "unit": "V", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "ion_a": {"name": "Ion detector A", "unit": "V", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "ion_b": {"name": "Ion detector B", "unit": "V", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "ptrak": {"name": "P-Trak particle count", "unit": "pt/cc", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "dtb": {"name": "DustTrak B relative reading", "unit": "mg/m^3 (source label; not smoke-calibrated)", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
    "plume_t": {"name": "Gas plume temperature", "unit": "degC", "measurement_class": "telemetry_channel", "is_instrument_sensor": False, "instrument_identity_status": "channel mapping label; exact hardware identity not independently confirmed"},
}


def _packet_list(packet_data: Any) -> List[Dict[str, Any]]:
    if isinstance(packet_data, list):
        return [p for p in packet_data if isinstance(p, dict)]
    if isinstance(packet_data, dict):
        for key in ("packets", "telemetry_packets", "data"):
            value = packet_data.get(key)
            if isinstance(value, list):
                return [p for p in value if isinstance(p, dict)]
        if "readings" in packet_data or "channels" in packet_data:
            return [packet_data]
    raise ValueError("Expected a packet list or an object containing 'packets'.")


def _reading_parts(raw: Any, variable_id: str, registry_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    registry_item = registry_item or {}
    if isinstance(raw, dict):
        value = raw.get("value", raw.get("reading", raw.get("measurement")))
        unit = raw.get("unit", raw.get("units", registry_item.get("unit")))
        name = raw.get("name", raw.get("display_name", registry_item.get("display_name", registry_item.get("name", variable_id))))
        measurement_class = raw.get("measurement_class", raw.get("class", registry_item.get("measurement_class")))
        item = {"value": value, "unit": unit, "name": name, "measurement_class": measurement_class}
    else:
        item = {
            "value": raw,
            "unit": registry_item.get("unit"),
            "name": registry_item.get("display_name", registry_item.get("name", variable_id)),
            "measurement_class": registry_item.get("measurement_class"),
        }
    for key in ("source_column", "instrument_identity_status", "unit_status", "source_sheet", "is_instrument_sensor"):
        if registry_item.get(key) is not None:
            item[key] = registry_item[key]
    return item


def to_frontend_payload(packet_data: Any) -> Dict[str, Any]:
    """Convert wide packets or {channels: ...} telemetry packets for a frontend.

    Optional top-level `channel_registry` is a mapping/list of channel metadata.
    Missing readings and missing timestamps remain null.
    """
    packets = _packet_list(packet_data)
    registry_raw = packet_data.get("channel_registry", {}) if isinstance(packet_data, dict) else {}
    if isinstance(registry_raw, list):
        channel_registry = {
            str(x.get("variable_id")): x for x in registry_raw
            if isinstance(x, dict) and x.get("variable_id")
        }
    elif isinstance(registry_raw, dict) and "variables" in registry_raw:
        variables = registry_raw.get("variables") or []
        channel_registry = {
            str(x.get("variable_id")): x for x in variables
            if isinstance(x, dict) and x.get("variable_id")
        }
    elif isinstance(registry_raw, dict):
        channel_registry = registry_raw
    else:
        channel_registry = {}

    variable_meta: Dict[str, Dict[str, Any]] = {}
    normalized: List[Dict[str, Any]] = []

    for index, packet in enumerate(packets):
        readings = packet.get("readings")
        channel_mode = False
        if not isinstance(readings, dict):
            readings = packet.get("channels")
            channel_mode = isinstance(readings, dict)
        if not isinstance(readings, dict):
            reserved = {
                "timestamp_utc", "timestamp_source", "time_basis", "source", "experiment",
                "experiment_id", "run_id", "psi", "psi_id", "provenance", "packet_id",
                "record_id", "source_file", "source_reference", "source_row_metadata",
                "quality_flags", "source_epoch_s", "elapsed_test_time_s", "time_string",
                "material", "candidate_materials", "label_verified", "heating_in_progress",
                "source_sheet", "source_row", "source_epoch_ms", "relative_seconds_source",
                "saffire_timestamp_ms", "cyg_saf_timestamp_ms", "saffire_repeat_source",
                "row_to_row_delta_t_ms", "schema_version", "packet_type",
            }
            readings = {k: v for k, v in packet.items() if k not in reserved}

        clean_readings: Dict[str, Any] = {}
        packet_reading_metadata = packet.get("reading_metadata") or {}
        packet_flags = packet.get("quality_flags") or []
        for variable_id, raw in readings.items():
            variable_id = str(variable_id)
            reg = channel_registry.get(variable_id)
            if reg is None:
                reg = _SAME_R_CHANNEL_META.get(variable_id)
            item = _reading_parts(raw, variable_id, reg)
            item["unit"] = item.get("unit") or (packet.get("units") or {}).get(variable_id)
            item["name"] = item.get("name") or (packet.get("variable_names") or {}).get(variable_id) or variable_id
            item["measurement_class"] = item.get("measurement_class") or (packet.get("measurement_classes") or {}).get(variable_id)
            source_meta = packet_reading_metadata.get(variable_id)
            if source_meta is None and packet_flags:
                channel_flags = [
                    flag for flag in packet_flags
                    if str(flag).startswith(variable_id + "_")
                    or str(flag).startswith(variable_id + ":")
                    or str(flag).startswith(variable_id + "-")
                ]
                if channel_flags:
                    source_meta = {"quality_flags": channel_flags}
            item["source_metadata"] = source_meta
            clean_readings[variable_id] = item
            meta = variable_meta.setdefault(variable_id, {
                "id": variable_id, "name": item["name"], "unit": item["unit"],
                "measurement_class": item["measurement_class"],
            })
            for field in ("name", "unit", "measurement_class"):
                if meta.get(field) is None and item.get(field) is not None:
                    meta[field] = item[field]

        normalized_packet = {
            "index": index,
            "packet_id": packet.get("packet_id", packet.get("record_id", index + 1)),
            "timestamp_utc": packet.get("timestamp_utc"),
            "timestamp_source": packet.get("timestamp_source"),
            "time_basis": packet.get("time_basis", "source_timestamp" if packet.get("timestamp_utc") else "untimed_record"),
            "source": packet.get("source", packet.get("source_file")),
            "experiment": packet.get("experiment", packet.get("experiment_id")),
            "run_id": packet.get("run_id"),
            "psi": packet.get("psi", packet.get("psi_id")),
            "provenance": packet.get("provenance"),
            "source_file": packet.get("source_file"),
            "source_reference": packet.get("source_reference"),
            "source_sheet": packet.get("source_sheet"),
            "source_row": packet.get("source_row"),
            "source_row_metadata": packet.get("source_row_metadata"),
            "source_epoch_s": packet.get("source_epoch_s"),
            "source_epoch_ms": packet.get("source_epoch_ms"),
            "elapsed_test_time_s": packet.get("elapsed_test_time_s"),
            "time_string": packet.get("time_string"),
            "material": packet.get("material"),
            "candidate_materials": packet.get("candidate_materials"),
            "label_verified": packet.get("label_verified"),
            "heating_in_progress": packet.get("heating_in_progress"),
            "quality_flags": packet_flags,
            "readings": clean_readings,
        }
        normalized.append(normalized_packet)

    source = None
    if isinstance(packet_data, dict):
        metadata = packet_data.get("metadata", {})
        source = packet_data.get("source") or (metadata.get("source") if isinstance(metadata, dict) else None)
    timestamped = [p for p in normalized if p.get("timestamp_utc")]
    def _time_key(packet):
        try:
            return datetime.fromisoformat(str(packet["timestamp_utc"]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
    latest = max(timestamped, key=_time_key) if timestamped else None
    return {
        "schema": "nasa.telemetry.frontend.v1",
        "source": source,
        "packet_count": len(normalized),
        "timestamped_packet_count": sum(p["timestamp_utc"] is not None for p in normalized),
        "untimestamped_packet_count": sum(p["timestamp_utc"] is None for p in normalized),
        "variables": list(variable_meta.values()),
        "packets": normalized,
        "latest": latest,
        "last_record": normalized[-1] if normalized else None,
    }


def load_and_adapt_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return to_frontend_payload(json.load(f))


def load_and_adapt_jsonl(path: str, channel_registry_path: Optional[str] = None) -> Dict[str, Any]:
    """Load newline-delimited packet JSON; optionally load a variable registry sidecar."""
    packets = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                packets.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    registry = {}
    if channel_registry_path:
        with open(channel_registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    return to_frontend_payload({
        "source": Path(path).name,
        "channel_registry": registry,
        "packets": packets,
    })


def write_frontend_json(packet_data: Any, output_path: str) -> Dict[str, Any]:
    payload = to_frontend_payload(packet_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    return payload


def latest_value(payload: Dict[str, Any], variable_id: str) -> Tuple[Optional[Any], Optional[str]]:
    latest = payload.get("latest") or {}
    reading = (latest.get("readings") or {}).get(variable_id)
    if not isinstance(reading, dict):
        return None, None
    return reading.get("value"), reading.get("unit")


def format_source_time(timestamp_utc: Optional[str]) -> str:
    if not timestamp_utc:
        return "Timestamp not provided by source"
    try:
        dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except (TypeError, ValueError):
        return str(timestamp_utc)
