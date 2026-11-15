"""Minimal Streamlit consumer example for Sensor Systems & Telemetry Integration.

This is a reference integration component, not the project dashboard UI.
"""
import json
import streamlit as st
from telemetry_frontend_adapter import to_frontend_payload, load_and_adapt_jsonl

st.set_page_config(page_title="NASA Telemetry Connector", layout="wide")
st.title("NASA telemetry connector")
st.caption("Load validated packet JSON or JSONL. This example does not create live sensor data.")

uploaded = st.file_uploader("Upload telemetry packets", type=["json", "jsonl"])
registry_upload = st.file_uploader("Optional channel registry JSON (for SAFFIRE-III)", type=["json"], key="registry")

if uploaded is None:
    st.info("No source data loaded. Show 'No source data' rather than demo sensor values.")
else:
    try:
        if uploaded.name.lower().endswith(".jsonl"):
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                packet_path = Path(d) / uploaded.name
                packet_path.write_bytes(uploaded.getvalue())
                registry_path = None
                if registry_upload is not None:
                    registry_path = Path(d) / "channel_registry.json"
                    registry_path.write_bytes(registry_upload.getvalue())
                payload = load_and_adapt_jsonl(str(packet_path), str(registry_path) if registry_path else None)
        else:
            payload = to_frontend_payload(json.load(uploaded))

        st.success(f"Loaded {payload['packet_count']:,} packets · {len(payload['variables'])} variables")
        st.write({
            "timestamped_packets": payload["timestamped_packet_count"],
            "untimestamped_packets": payload["untimestamped_packet_count"],
        })
        st.subheader("Latest source record")
        if payload["latest"]:
            st.json(payload["latest"])
        else:
            st.info("No source timestamp is available; do not substitute the computer clock.")
        st.subheader("Registered variables present in this file")
        st.dataframe(payload["variables"], use_container_width=True)
        st.download_button(
            "Download frontend-ready JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name="frontend_telemetry.json",
            mime="application/json",
        )
        # Frontend owner: use payload['packets'] for replay and payload['latest']
        # only when the source provides a real timestamp.
    except Exception as exc:
        st.error(f"Could not load telemetry file: {exc}")
