"""Build telemetry outputs for every configured processed NASA source table."""
from pathlib import Path
import json
import sys
from telemetry import load_registry, import_table, validate_observations, build_wide_packets, export_packets_json, packets_to_dataframe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend_adapter"))
from telemetry_frontend_adapter import write_frontend_json  # noqa: E402
REGISTRY = ROOT / "config" / "variable_registry.json"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output" / "all_sources"
SOURCE_FILES = [
    "source_SAME_PSI-101-102_smoke_particles_v1.0.csv",
    "source_BASS_PSI-26_test_log_v1.0.csv",
    "source_microgravity_flammability_unified_master_v1.0.csv",
    "source_FLEX_PSI-69_clean_v1.0.csv",
    "source_BASS-II_flame_spread_microgravity_v1.0.csv",
    "source_SPICE_PSI-107_smoke_points_v1.0.csv",
    "source_SAFFIRE_I-III_PSI-98-100_v1.0.csv",
]

def main():
    registry = load_registry(REGISTRY)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"source_count": 0, "observations": 0, "packets": 0, "datasets": []}
    all_packets = []
    all_errors = []
    for name in SOURCE_FILES:
        path = PROCESSED / name
        observations, import_errors = import_table(path, REGISTRY)
        validation = validate_observations(observations, registry)
        envelope, frame = build_wide_packets(observations, registry)
        export_packets_json(envelope, OUT / f"{path.stem}_packets.json")
        frame.to_csv(OUT / f"{path.stem}_packets.csv", index=False)
        item = {"dataset": name, "observations": len(observations), "packets": envelope["packet_count"],
                "import_errors": len(import_errors), "validation": validation}
        summary["datasets"].append(item)
        summary["source_count"] += 1
        summary["observations"] += len(observations)
        summary["packets"] += envelope["packet_count"]
        all_errors.extend({"dataset": name, **e} for e in import_errors)
        all_packets.extend(envelope["packets"])
    combined = {"schema_version": "1.1", "format": "wide_packet_per_timestamp",
                "source_count": summary["source_count"], "packet_count": len(all_packets),
                "grouping_policy": "Packets are grouped within source file, experiment, PSI, run and source timestamp; untimed rows remain separate.",
                "timestamp_policy": "No sample timestamps are invented. Experiment dates/times remain source_row_metadata.",
                "data_semantics": "NASA-derived reported experimental variables and derived results; not asserted to be physical sensor channels.",
                "packets": all_packets}
    export_packets_json(combined, OUT / "all_sources_wide_packets.json")
    combined_frame = packets_to_dataframe(all_packets)
    combined_frame.to_csv(OUT / "all_sources_wide_packets.csv", index=False)
    write_frontend_json(combined, str(OUT / "frontend_payload.json"))
    summary["combined_packet_file"] = (OUT / "all_sources_wide_packets.json").relative_to(ROOT).as_posix()
    summary["import_errors"] = all_errors
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("source_count", "observations", "packets")}, indent=2))
    if all_errors or any(not d["validation"]["ok"] for d in summary["datasets"]):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
