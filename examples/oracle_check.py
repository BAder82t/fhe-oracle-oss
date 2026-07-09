# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CI-ready precision check with diagnostics-on-FAIL.

Real, runnable version of the script github_action.yml describes.
Replace plaintext_fn/fhe_fn with your model.

Run:
    pip install cma numpy
    python examples/oracle_check.py
    echo $?   # 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle
from fhe_oracle.report import to_markdown


# 1. Your plaintext reference implementation.
#    Replace this with your real model's plaintext prediction.
def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))


# 2. A stand-in for your FHE-compiled version.
#    Replace this with your real FHE predict function.
def fhe_fn(x):
    v = float(np.sum(np.asarray(x) ** 2))
    base = 1e-5 * v
    amp = 100.0 if v > 8.0 else 1.0
    return plaintext_fn(x) + base * amp


if __name__ == "__main__":
    oracle = FHEOracle(
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
        input_dim=4,
        input_bounds=[(-3.0, 3.0)] * 4,
        seed=int(os.environ.get("ORACLE_SEED", "0")),
    )
    result = oracle.run(
        n_trials=int(os.environ.get("ORACLE_N_TRIALS", "500")),
        threshold=float(os.environ.get("ORACLE_THRESHOLD", "0.01")),
    )

    diagnostics = {}
    if result.verdict == "FAIL":
        shrunk = oracle.shrink(result, max_evals=200)  # smallest triggering input
        diagnostics["shrunk_input"] = [round(v, 4) for v in shrunk.shrunk_input]
        diagnostics["shrink_reduction"] = (
            f"{100.0 * (1.0 - shrunk.shrunk_norm / shrunk.original_norm):.1f}%"
            if shrunk.original_norm > 0
            else "n/a"
        )

    print(to_markdown(result, diagnostics=diagnostics))
    sys.exit(0 if result.verdict == "PASS" else 1)
