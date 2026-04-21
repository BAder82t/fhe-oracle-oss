# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""High-dimensional scaling sweep (B2).

Extends Table 3 of the paper to higher d. Paper shows CMA-ES advantage
declining from 1.89x (d=4) to 1.18x (d=32). Conjectured parity at
d in [50, 100]. This sweep tests d in {8, 16, 32, 64, 128} with four
configurations:

  full_cma  : separable=False, random_floor=0.0   (baseline CMA-ES)
  sep_cma   : separable=True,  random_floor=0.0   (diagonal covariance)
  full_warm : separable=False, random_floor=0.3   (A1 stack)
  sep_warm  : separable=True,  random_floor=0.3   (A1 + sep)

Plus a uniform-random baseline at every (d, budget) cell for ratios.

Circuits:
  lr_mock    : logistic regression mock at variable d (synthetic data)
  poly_mock  : depth-4 polynomial mock at variable d

Budget scales with d: B=500 for d<=32, B=1000 for d=64, B=2000 for d=128.

10 seeds per cell -> 4 configs x 2 circuits x 5 dims x 10 seeds = 400
oracle cells + 2 x 5 x 10 = 100 random baseline cells. All mocks;
total wall-clock under 2 minutes on a laptop.

Output: benchmarks/results/highdim_sweep.csv
Columns: config, circuit, d, budget, seed, max_error, wall_clock_s,
         n_trials
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle import FHEOracle


DIMENSIONS = [8, 16, 32, 64, 128]
SEEDS = list(range(10))
CONFIGS: dict[str, dict[str, Any]] = {
    "full_cma":  {"separable": False, "random_floor": 0.0},
    "sep_cma":   {"separable": True,  "random_floor": 0.0},
    "full_warm": {"separable": False, "random_floor": 0.3},
    "sep_warm":  {"separable": True,  "random_floor": 0.3},
}


def budget_for_dim(d: int) -> int:
    if d <= 32:
        return 500
    if d <= 64:
        return 1000
    return 2000


# --- Logistic regression mock at arbitrary d ---

def make_lr_mock(d: int, rng_seed: int = 42) -> dict[str, Any]:
    """Synthetic logistic-regression mock circuit at dimension d.

    Matches the ablation_heuristics.make_circuit1 pattern but scales to
    variable d. The noise amplification trigger uses ||x||_2^2 / d so
    the failure mode has comparable density across dimensions.
    """
    rng = np.random.default_rng(rng_seed)
    n_samples = max(200, 20 * d)
    X = rng.normal(0.0, 1.0, size=(n_samples, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)

    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b
    w_final = w.copy()
    b_final = float(b)

    def plain(x):
        arr = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w_final, arr) + b_final)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    def fhe(x):
        arr = np.asarray(x, dtype=np.float64)
        pval = plain(arr)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        noise = float(local.normal(0.0, 1e-4))
        # Density-normalised magnitude trigger.
        z_proxy = float(np.dot(arr, arr)) / max(1, d)
        if z_proxy > 0.5 and abs(pval - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 0.5)
            noise *= amp
        return pval + noise

    return {
        "name": "lr_mock",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


# --- Depth-4 polynomial mock at arbitrary d ---

def make_poly_mock(d: int) -> dict[str, Any]:
    """Depth-4 polynomial p(x) = sum_i c_i * x_i^2 * x_{i+1} at variable d."""
    coeffs = np.linspace(0.5, 1.5, d - 1)

    def plain(x):
        arr = np.asarray(x, dtype=np.float64)
        return float(np.sum(coeffs * arr[:-1] ** 2 * arr[1:]))

    def fhe(x):
        arr = np.asarray(x, dtype=np.float64)
        plain_val = plain(arr)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        base_noise = float(local.normal(0.0, 5e-5))
        intermediates = arr[:-1] ** 2 * arr[1:]
        max_mag = (
            float(np.max(np.abs(intermediates))) if intermediates.size else 0.0
        )
        amp = 1.0 + 20.0 * max(0.0, max_mag - 1.0)
        return plain_val + base_noise * amp

    return {
        "name": "poly_mock",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-2.0, 2.0)] * d,
    }


CIRCUIT_BUILDERS: dict[str, Callable[[int], dict[str, Any]]] = {
    "lr_mock":   make_lr_mock,
    "poly_mock": make_poly_mock,
}


def run_random_baseline(
    circuit: dict[str, Any], budget: int, seed: int
) -> tuple[float, float, int]:
    """Uniform random sampling baseline. Returns (max_error, wall_s, n_evals)."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in circuit["bounds"]])
    highs = np.array([hi for _, hi in circuit["bounds"]])
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    t0 = time.perf_counter()
    best = 0.0
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            p = plain(x)
            f = fhe(x)
        except Exception:
            continue
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
        f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
        n = min(p_arr.size, f_arr.size)
        if n == 0:
            continue
        err = float(np.max(np.abs(p_arr[:n] - f_arr[:n])))
        if err > best:
            best = err
    return best, time.perf_counter() - t0, budget


def run_oracle(
    config_name: str,
    cfg: dict[str, Any],
    circuit: dict[str, Any],
    budget: int,
    seed: int,
) -> tuple[float, float, int]:
    """Run FHEOracle with the given config. Returns (max_error, wall_s, n_trials)."""
    oracle = FHEOracle(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        input_dim=circuit["d"],
        input_bounds=circuit["bounds"],
        seed=seed,
        separable=cfg["separable"],
        random_floor=cfg["random_floor"],
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=budget, threshold=0.0)
    wall = time.perf_counter() - t0
    return res.max_error, wall, res.n_trials


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "highdim_sweep.csv")

    total_cells = (
        len(CONFIGS) * len(CIRCUIT_BUILDERS) * len(DIMENSIONS) * len(SEEDS)
        + len(CIRCUIT_BUILDERS) * len(DIMENSIONS) * len(SEEDS)
    )
    print("High-dimensional scaling sweep (B2)")
    print(f"  Dimensions  : {DIMENSIONS}")
    print(f"  Configs     : {list(CONFIGS.keys())} + uniform_random baseline")
    print(f"  Circuits    : {list(CIRCUIT_BUILDERS.keys())}")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Total cells : {total_cells}")
    print(f"  Output      : {out_path}")
    print("=" * 75)

    t_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    cell = 0

    for circ_name, builder in CIRCUIT_BUILDERS.items():
        for d in DIMENSIONS:
            budget = budget_for_dim(d)
            circuit = builder(d)

            # Random baseline at (circuit, d).
            for seed in SEEDS:
                cell += 1
                err, wall, n_evals = run_random_baseline(circuit, budget, seed)
                rows.append({
                    "config": "uniform_random",
                    "circuit": circ_name,
                    "d": d,
                    "budget": budget,
                    "seed": seed,
                    "max_error": err,
                    "wall_clock_s": wall,
                    "n_trials": n_evals,
                })
                if cell % 20 == 0:
                    print(
                        f"  [{cell:4d}/{total_cells}] "
                        f"{circ_name} d={d} random seed={seed} "
                        f"err={err:.4e} t={wall:.2f}s"
                    )

            # Oracle configs at (circuit, d).
            for cfg_name, cfg in CONFIGS.items():
                for seed in SEEDS:
                    cell += 1
                    err, wall, n_trials = run_oracle(
                        cfg_name, cfg, circuit, budget, seed
                    )
                    rows.append({
                        "config": cfg_name,
                        "circuit": circ_name,
                        "d": d,
                        "budget": budget,
                        "seed": seed,
                        "max_error": err,
                        "wall_clock_s": wall,
                        "n_trials": n_trials,
                    })
                    if cell % 20 == 0:
                        print(
                            f"  [{cell:4d}/{total_cells}] "
                            f"{circ_name} d={d} {cfg_name} seed={seed} "
                            f"err={err:.4e} t={wall:.2f}s"
                        )

    elapsed = time.perf_counter() - t_start
    print("=" * 75)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    fieldnames = [
        "config", "circuit", "d", "budget", "seed",
        "max_error", "wall_clock_s", "n_trials",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
