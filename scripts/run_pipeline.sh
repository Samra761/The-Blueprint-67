#!/usr/bin/env bash
# Rebuild the legacy Stage 1 / Stage 2 artifacts from raw CSVs in data/raw/<material>/
# Usage:  bash scripts/run_pipeline.sh            (from anywhere)
#
# Raw SAME CSVs are not committed (see .gitignore); fetch them first with
#   python src/fetch_psi.py           (configure .env from .env.example)
# The Task 3-5 studies (label audit, sensor sensitivity, chained LOTO) read the
# committed SAME-R packet stream instead and do not need data/raw. For the
# complete reproduction use:  python scripts/verify_repo.py --full
set -euo pipefail
cd "$(dirname "$0")/.."

if ! ls data/raw/*/*.csv >/dev/null 2>&1; then
  echo "No raw CSVs in data/raw/<material>/. Run: python src/fetch_psi.py" >&2
  exit 1
fi

echo "▶ Phase 0  load + phase-label SAME CSVs (+ label-manifest check)"; python src/load.py
echo "▶ Phase 1  features";                                            python src/features.py
echo "▶ Phase 2  anomaly detector (~10-15 min)";                        python src/anomaly.py
echo "▶ Phase 3  material classifier";                                 python src/classifier.py
echo "✔ Done. Reports in reports/, models in models/"
echo "  Verify:    python scripts/verify_repo.py"
echo "  Dashboard: streamlit run dashboard/app.py"
