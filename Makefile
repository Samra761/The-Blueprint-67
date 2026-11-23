# Flame in Freefall - common entry points (run from the project root).
# Windows without make: run the python commands shown for each target.
PY ?= python

.PHONY: help verify test audit pipeline chained sensors full dryrun dashboard

help:
	@echo "make verify     fast integrity check, no training (~10 s)"
	@echo "make test       unit tests (root + sensor workspace)"
	@echo "make audit      re-run the SAME label audit (writes results/same_label_*)"
	@echo "make pipeline   raw CSVs -> features -> Stage 1 -> Stage 2 (needs data/raw)"
	@echo "make sensors    per-sensor sensitivity study (~1.5 h)"
	@echo "make chained    chained causal LOTO-41 evaluation (~1.3 h)"
	@echo "make full       everything above, in order (hours)"
	@echo "make dryrun     print the full-reproduction commands"
	@echo "make dashboard  launch the Streamlit dashboard"

verify:
	$(PY) scripts/verify_repo.py

test:
	$(PY) -m pytest -q tests
	cd Sensor_Systems_and_Telemetry_Integration && PYTHONPATH=integration_update/src \
		$(PY) -m pytest -q tests frontend_adapter/test_adapter.py integration_update/tests

audit:
	$(PY) scripts/audit_same_tests.py

pipeline:
	bash scripts/run_pipeline.sh

sensors:
	$(PY) src/sensor_sensitivity.py

chained:
	$(PY) src/evaluate_chained.py

full:
	$(PY) scripts/verify_repo.py --full

dryrun:
	$(PY) scripts/verify_repo.py --full --dry-run --skip-tests

dashboard:
	streamlit run dashboard/app.py
