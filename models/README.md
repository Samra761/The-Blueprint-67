# models/

Trained model binaries are **git-ignored** (they're large and fully reproducible).
Run `bash scripts/run_pipeline.sh` to regenerate them.

| File | Built by | Model | Size |
|---|---|---|---|
| `anomaly_iforest.joblib` | `src/anomaly.py` | 8 per-sensor Isolation Forests (smoke early warning) | ~26 MB |
| `material_classifier.joblib` | `src/classifier.py` | LightGBM, 5 classes (burning material) | ~2.7 MB |

The FLEX burn-time and flame-outcome models run in the browser and are embedded in `dashboard/app.py`.

To version these files in git instead, use Git LFS (see `.gitattributes`).
