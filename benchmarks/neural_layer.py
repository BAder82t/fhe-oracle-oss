# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Neural layer (dense + Chebyshev sigmoid) FHE precision benchmark."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle


def main() -> int:
    rng = np.random.default_rng(11)
    d = 10
    hidden = 4
    W = rng.normal(0.0, 0.5, size=(hidden, d))
    b = rng.normal(0.0, 0.1, size=hidden)

    def plain_fn(x: list[float]) -> list[float]:
        arr = np.asarray(x, dtype=np.float64)
        z = W @ arr + b
        return (1.0 / (1.0 + np.exp(-z))).tolist()

    def fhe_fn(x: list[float]) -> list[float]:
        # Approx sigmoid via degree-3 Chebyshev. Introduces polynomial
        # approximation error that grows at the edges of the domain —
        # the CMA-ES oracle should gravitate there.
        arr = np.asarray(x, dtype=np.float64)
        z = W @ arr + b
        z_clip = np.clip(z / 5.0, -1.0, 1.0)
        approx = 0.5 + 0.5 * (1.5 * z_clip - 0.5 * z_clip ** 3)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        noise = local.normal(0.0, 1e-5, size=hidden)
        return (approx + noise).tolist()

    print("=" * 60)
    print("FHE Oracle benchmark: dense + Chebyshev sigmoid")
    print("=" * 60)

    oracle = FHEOracle(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_fn,
        input_dim=d,
        input_bounds=[(-3.0, 3.0)] * d,
        seed=11,
    )
    t0 = time.perf_counter()
    result = oracle.run(n_trials=500, threshold=1e-3)
    print(f"Verdict:    {result.verdict}")
    print(f"Max error:  {result.max_error:.6e}")
    print(f"Elapsed:    {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
