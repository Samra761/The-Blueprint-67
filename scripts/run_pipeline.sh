#!/usr/bin/env bash
# Rebuild everything from raw CSVs in data/raw/<material>/
# Usage:  bash scripts/run_pipeline.sh            (from the project root)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "▶ Phase 0  load + phase-label SAME CSVs";   python src/load.py
echo "▶ Phase 1  features";                       python src/features.py
echo "▶ Phase 2  anomaly detector (~10-15 min)";   python src/anomaly.py
echo "▶ Phase 3  material classifier";            python src/classifier.py
echo "✔ Done. Reports in reports/, models in models/"
echo "  Launch the dashboard:  streamlit run dashboard/app.py"
