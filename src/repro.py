"""Single source of truth for reproducibility settings.

Every stochastic component in the repository (Isolation Forests, Random
Forests, LightGBM, the leakage self-check perturbation and figure jitter)
takes its seed from ``SEED`` below. Nothing else in ``src/`` hard-codes a
seed.

Why the canonical seed is 0
---------------------------
All committed artifacts were generated with seed 0:

* ``results/chained_loto_metrics.json``  ->  ``meta.seed == 0``
* ``reports/anomaly/``, ``reports/classifier/``, ``reports/sensor_sensitivity/``
  (``random_state=0`` in the code that produced them).

Changing the seed changes the forests and therefore the reported numbers, so
the committed tables are only reproducible with ``SEED = 0``. To run a
seed-robustness study without editing code, set the environment variable::

    FIF_SEED=42 python src/evaluate_chained.py

``scripts/verify_repo.py`` warns when the active seed differs from the seed
recorded in the committed results.
"""
from __future__ import annotations

import os
import random

import numpy as np

CANONICAL_SEED = 0
SEED = int(os.environ.get("FIF_SEED", CANONICAL_SEED))


def seed_everything(seed: int = SEED) -> int:
    """Seed Python's and NumPy's global RNGs (defensive: the pipeline passes
    explicit ``random_state`` everywhere, so results do not depend on this)."""
    random.seed(seed)
    np.random.seed(seed)
    return seed
