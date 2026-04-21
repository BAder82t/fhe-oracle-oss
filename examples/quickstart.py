# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""30-second quickstart for FHE Oracle.

Run:
    pip install cma numpy
    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle


# 1. Your plaintext reference implementation.
def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))


# 2. A stand-in for your FHE-compiled version.
#    Replace this with your real FHE predict function.
#    Here: noise that scales with input norm^2 (a CKKS depth-noise
#    pattern), with a hot zone that inflates the error 100x when
#    |x|^2 > 8.
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
        seed=0,
    )
    result = oracle.run(n_trials=300, threshold=1e-3)
    print(result)
    print(f"verdict     = {result.verdict}")
    print(f"max_error   = {result.max_error:.6e}")
    print(f"worst_input = {[round(v, 3) for v in result.worst_input]}")
    sys.exit(0 if result.verdict == "PASS" else 1)
