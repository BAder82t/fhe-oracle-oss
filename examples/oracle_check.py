# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CI-ready precision check -- the one-call version.

Replace plaintext_fn/fhe_fn/input_bounds with your model. Everything
else (probe, search, shrink, localize, report) happens inside check().

Run:
    pip install cma numpy
    python examples/oracle_check.py
    echo $?   # 0 = PASS, 1 = FAIL

Equivalent from the shell, no Python file needed at the call site:
    fhe-oracle check examples/oracle_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import check


# 1. Your plaintext reference implementation.
def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))


# 2. A stand-in for your FHE-compiled version. Replace with your real
#    FHE predict function.
def fhe_fn(x):
    v = float(np.sum(np.asarray(x) ** 2))
    base = 1e-5 * v
    amp = 100.0 if v > 8.0 else 1.0
    return plaintext_fn(x) + base * amp


input_bounds = [(-3.0, 3.0)] * 4
n_trials = int(os.environ.get("ORACLE_N_TRIALS", "500"))
threshold = float(os.environ.get("ORACLE_THRESHOLD", "0.01"))
seed = int(os.environ.get("ORACLE_SEED", "0"))


if __name__ == "__main__":
    result = check(
        plaintext_fn, fhe_fn, input_bounds,
        n_trials=n_trials, threshold=threshold, seed=seed,
    )
    print(result.report)
    sys.exit(0 if result.oracle_result.verdict == "PASS" else 1)
