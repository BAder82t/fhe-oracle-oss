# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Patent reference benchmark: random vs CMA-ES on a CKKS sigmoid defect.

Reproduces the headline max-error ratio reported in the patent prosecution
package (PCT/IB2026/053378). A synthetic CKKS-like logistic regression
circuit is built with a deliberate sigmoid polynomial-approximation defect;
random sampling from the typical operational distribution misses the defect
entirely, while CMA-ES over the adversarial search domain finds it.

Circuit
-------
- 5 input features, fixed weights W and bias B.
- FHE side: W·x + b, followed by a 3-term Taylor approximation of sigmoid
  (sigma(z) ~ 0.5 + z/4 - z^3/48) clamped to [-0.5, 1.5]. This models a
  real CKKS circuit compiled with insufficient polynomial depth for sigmoid.
- Plaintext side: exact 64-bit sigmoid.

Ranges
------
- Random baseline samples from the *operational* distribution [-0.3, 0.3]^5:
  max logit 0.3 * ||W||_1 + |B| ~ 0.86, well inside the approximation-valid
  region. Random therefore finds no divergence > 0.1.
- CMA-ES searches the extended adversarial domain [-5, 5]^5. Inputs with
  |logit| > 3 drive the polynomial error > 0.1, and the adversarial search
  concentrates the evaluation budget in that region.

The asymmetry between random and adversarial ranges is deliberate and matches
the patent's evaluation protocol: random testing models a user who samples
from the training distribution, while the oracle searches the full declared
input space for the deployed model.

Usage
-----
    pip install cma numpy
    python benchmarks/patent_logistic_regression.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle
from fhe_oracle.adapters.base import FHEAdapter


N_FEATURES = 5
EVAL_BUDGET = 500
DIVERGENCE_THRESHOLD = 0.1

W = np.array([0.82, -0.65, 0.48, -0.31, 0.19], dtype=np.float64)
B = float(0.12)

RANDOM_INPUT_RANGE = 0.3
CMAES_INPUT_RANGE = 5.0
CMAES_SIGMA0 = 0.5

APPROX_THRESHOLD = 3.0
MULT_DEPTH = 3
SCALE_BITS = 40


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


def plaintext_logistic(x: list[float]) -> float:
    z = float(np.dot(W, np.asarray(x, dtype=np.float64)) + B)
    return _sigmoid(z)


def _poly_sigmoid_approx(z: float) -> float:
    raw = 0.5 + (z / 4.0) - (z ** 3 / 48.0)
    return max(-0.5, min(1.5, raw))


class SimulatedCKKSAdapter(FHEAdapter):
    """Numerical simulation of a CKKS logistic-regression circuit."""

    def __init__(self) -> None:
        self._scale = float(2 ** SCALE_BITS)

    def encrypt(self, x: list[float]) -> Any:
        arr = np.asarray(x, dtype=np.float64)
        quantized = np.round(arr * self._scale) / self._scale
        return {
            "values": quantized,
            "level": 0,
            "budget": float(MULT_DEPTH * SCALE_BITS),
        }

    def decrypt(self, ciphertext: Any) -> list[float]:
        values = ciphertext["values"]
        return values.tolist() if hasattr(values, "tolist") else list(values)

    def run_fhe_program(self, ciphertext: Any) -> Any:
        x = ciphertext["values"]
        wx = np.dot(W, x)
        z = float(np.round((wx + B) * self._scale) / self._scale)
        fhe_result = _poly_sigmoid_approx(z)
        fhe_result = float(np.round(fhe_result * self._scale) / self._scale)
        return {
            "values": np.array([fhe_result]),
            "level": MULT_DEPTH,
            "budget": 0.0,
        }

    def get_noise_budget(self, ciphertext: Any) -> float:
        return float(ciphertext.get("budget", 0.0))

    def get_mult_depth_used(self, ciphertext: Any) -> int:
        return int(ciphertext.get("level", 0))

    def get_scheme_name(self) -> str:
        return "CKKS-Simulated"


def _fhe_scalar(adapter: FHEAdapter, x: list[float]) -> float:
    ct = adapter.encrypt(x)
    ct_out = adapter.run_fhe_program(ct)
    return float(adapter.decrypt(ct_out)[0])


def run_random_baseline(
    adapter: FHEAdapter,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[int, float, list[float]]:
    n_diverging = 0
    max_div = 0.0
    max_input: list[float] = [0.0] * N_FEATURES
    for _ in range(n_trials):
        x = rng.uniform(-RANDOM_INPUT_RANGE, RANDOM_INPUT_RANGE, size=N_FEATURES).tolist()
        div = abs(_fhe_scalar(adapter, x) - plaintext_logistic(x))
        if div > DIVERGENCE_THRESHOLD:
            n_diverging += 1
        if div > max_div:
            max_div = div
            max_input = x
    return n_diverging, max_div, max_input


def main(seed: int = 42) -> int:
    try:
        import cma  # noqa: F401
    except ImportError:
        print("[ERROR] cma is not installed. Run: pip install cma")
        return 1

    rng = np.random.default_rng(seed)
    adapter = SimulatedCKKSAdapter()

    print("=" * 60)
    print("FHE Oracle - Patent reference benchmark")
    print("Random testing vs CMA-ES adversarial search")
    print("=" * 60)
    print(f"Circuit:      Logistic regression, {N_FEATURES} features")
    print(f"Defect:       Degree-3 polynomial sigmoid (valid |z|<={APPROX_THRESHOLD:.1f})")
    print(f"Eval budget:  {EVAL_BUDGET} trials each")
    print(f"Threshold:    divergence > {DIVERGENCE_THRESHOLD}")
    print(f"Random range: +/-{RANDOM_INPUT_RANGE}  (operational distribution)")
    print(f"CMA-ES range: +/-{CMAES_INPUT_RANGE}  (adversarial search domain)")
    print(f"Seed:         {seed}")
    print()

    print("Running random baseline ...")
    t0 = time.perf_counter()
    rand_count, rand_max, rand_x = run_random_baseline(adapter, EVAL_BUDGET, rng)
    rand_time = time.perf_counter() - t0
    print(f"  Diverging inputs found: {rand_count} / {EVAL_BUDGET}")
    print(f"  Maximum divergence:     {rand_max:.6e}")
    print(f"  Time:                   {rand_time:.2f}s")
    print()

    print("Running CMA-ES adversarial search ...")
    oracle = FHEOracle(
        plaintext_fn=plaintext_logistic,
        adapter=adapter,
        input_dim=N_FEATURES,
        input_bounds=[(-CMAES_INPUT_RANGE, CMAES_INPUT_RANGE)] * N_FEATURES,
        sigma0=CMAES_SIGMA0,
        seed=seed,
    )
    result = oracle.run(n_trials=EVAL_BUDGET, threshold=DIVERGENCE_THRESHOLD)
    print(f"  Maximum divergence:     {result.max_error:.6e}")
    print(f"  Time:                   {result.elapsed_seconds:.2f}s")
    print()

    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Random max divergence:  {rand_max:.6e}")
    print(f"  CMA-ES max divergence:  {result.max_error:.6e}")
    if rand_max > 0:
        ratio = result.max_error / rand_max
        print(f"  Max-error ratio:        {ratio:.1f}x")
    print(f"  CMA-ES best input:      {[round(v, 4) for v in result.worst_input]}")
    print()
    return 0 if result.verdict == "FAIL" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patent reference benchmark: random vs CMA-ES"
    )
    parser.add_argument("--seed", type=int, default=42)
    raise SystemExit(main(parser.parse_args().seed))
