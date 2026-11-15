#!/usr/bin/env python3
"""Validate and promote SAME-R source epoch seconds to ISO-8601 UTC.

Run from the Sensor_Systems_and_Telemetry_Integration root:
    python src/promote_same_r_source_epoch_utc.py

The original source_epoch_s is retained. Every converted value is checked
against the source TimeString GMT label before the file is replaced.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "ml_lead_handoff/SAME_R_WIDE_PACKETS.jsonl"


def main() -> None:
    if not PACKETS.exists():
        raise FileNotFoundError(PACKETS)
    fd, tmp_name = tempfile.mkstemp(prefix="same_r_", suffix=".jsonl", dir=PACKETS.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, PACKETS.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                packet = json.loads(line)
                epoch = float(packet["source_epoch_s"])
                expected_label = datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y/%j/%H:%M:%S")
                if packet.get("time_string") != expected_label:
                    raise ValueError(
                        f"GMT label mismatch at line {line_number}: "
                        f"{packet.get('time_string')} != {expected_label}"
                    )
                packet["timestamp_utc"] = datetime.fromtimestamp(
                    epoch, timezone.utc
                ).isoformat(timespec="seconds").replace("+00:00", "Z")
                packet["timestamp_source"] = "source_epoch_s_unix_seconds"
                packet["time_basis"] = "source_epoch_s_unix_utc"
                out.write(json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
        os.replace(tmp_name, PACKETS)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    print(f"Promoted {count} SAME-R packets; every TimeString GMT label matched.")


if __name__ == "__main__":
    main()
