"""CLI: import a CSV and export wide packets (JSON + CSV)."""
from pathlib import Path
import json, sys
from telemetry import load_registry, import_table, validate_observations, build_wide_packets, export_packets_json

ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)<2:
    raise SystemExit("Usage: python src/run_import.py path/to/input.csv [output.json]")
src=Path(sys.argv[1])
out=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/"data/processed/imported_wide_packets.json"
registry_path=ROOT/"config/variable_registry.json"
registry=load_registry(registry_path)
observations,import_errors=import_table(src,registry_path)
validation=validate_observations(observations,registry)
packet_data,wide_df=build_wide_packets(observations,registry)
export_packets_json(packet_data,out)
csv_out=out.with_suffix(".csv")
wide_df.to_csv(csv_out,index=False)
report={"input":str(src),"json_output":str(out),"csv_output":str(csv_out),
        "observations":len(observations),"packets":packet_data["packet_count"],
        "import_errors":len(import_errors),"validation":validation,
        "import_error_details":import_errors}
report_path=out.with_name(out.stem+"_validation.json")
report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({"observations":len(observations),"packets":packet_data["packet_count"],
 "import_errors":len(import_errors),"validation_ok":validation["ok"],
 "json_output":str(out),"csv_output":str(csv_out),"report":str(report_path)},indent=2))
if not validation["ok"]: raise SystemExit(2)
