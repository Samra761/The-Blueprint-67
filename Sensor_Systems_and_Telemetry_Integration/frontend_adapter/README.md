# Frontend Adapter

This directory contains the adapter that exposes telemetry packet data in a frontend-friendly structure, plus a Streamlit integration example and adapter tests. It is a handoff/interface component; it is not the frontend application itself.

## Files

- `telemetry_frontend_adapter.py` — adapter functions for preparing the telemetry payload.
- `streamlit_integration_example.py` — example usage pattern for a Streamlit consumer.
- `test_adapter.py` — adapter tests.

## Integration boundary

The frontend owner remains responsible for UI pages, dashboard behavior, visualization, and application integration. The adapter does not create or own the canonical backend API. Connect it to the agreed upstream data source/API when the Data Engineering lead confirms the contract.

## Test

From the module root:

```bash
python -m pytest -q frontend_adapter/test_adapter.py
```

Preserve nulls and quality flags. Do not convert missing values to zero or present unverified/calibration-dependent values as confirmed measurements.
