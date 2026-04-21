# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Polynomial evaluation FHE precision benchmark.

Evaluates a depth-4 polynomial circuit under a mocked (or real) FHE
backend and compares CMA-ES adversarial search to random sampling.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle


def plaintext_poly(x: list[float]) -> float:
    """Evaluate p(x) = sum_i c_i * x_i^2 * x_{i+1}, depth-4 in CKKS."""
    arr = np.asarray(x, dtype=np.float64)
    coeffs = np.linspace(0.5, 1.5, len(arr) - 1)
    out = float(np.sum(coeffs * arr[:-1] ** 2 * arr[1:]))
    return out


def mock_fhe_poly(x: list[float]) -> float:
    """Mock CKKS evaluation of plaintext_poly with calibrated noise.

    Noise scales with the magnitude of intermediate products — the
    failure mode real CKKS circuits exhibit at high multiplicative
    depth.
    """
    arr = np.asarray(x, dtype=np.float64)
    plain = plaintext_poly(x)

    seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
    local = np.random.default_rng(seed)
    base_noise = float(local.normal(0.0, 5e-5))

    # Noise amplification scales with max |intermediate product|.
    intermediates = arr[:-1] ** 2 * arr[1:]
    max_magnitude = float(np.max(np.abs(intermediates))) if intermediates.size else 0.0
    amp = 1.0 + 20.0 * max(0.0, max_magnitude - 1.0)
    return plain + base_noise * amp


def main() -> int:
    d = 6
    bounds = [(-2.0, 2.0)] * d
    n_trials = 500

    print("=" * 60)
    print("FHE Oracle benchmark: polynomial evaluation (depth 4)")
    print(f"Input dim={d}, using mock FHE function")
    print("=" * 60)

    rng = np.random.default_rng(7)

    print(f"\n[1/2] Random sampling ({n_trials} trials)...")
    t0 = time.perf_counter()
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best_random = 0.0
    for _ in range(n_trials):
        x = rng.uniform(lows, highs).tolist()
        err = abs(plaintext_poly(x) - mock_fhe_poly(x))
        if err > best_random:
            best_random = err
    print(
        f"      max error: {best_random:.6e}  elapsed: {time.perf_counter() - t0:.2f}s"
    )

    print(f"\n[2/2] FHE Oracle (CMA-ES, {n_trials} trials)...")
    oracle = FHEOracle(
        plaintext_fn=plaintext_poly,
        fhe_fn=mock_fhe_poly,
        input_dim=d,
        input_bounds=bounds,
        seed=7,
    )
    result = oracle.run(n_trials=n_trials, threshold=1e-3)
    print(
        f"      max error: {result.max_error:.6e}  elapsed: {result.elapsed_seconds:.2f}s"
    )

    print("\n" + "=" * 60)
    print(f"Verdict: {result.verdict}")
    if best_random > 0:
        print(f"Oracle / random ratio: {result.max_error / best_random:.1f}x")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
