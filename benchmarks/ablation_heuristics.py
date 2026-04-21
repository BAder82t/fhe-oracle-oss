# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""S0 Heuristic Lesion Audit — 9-config paired matrix on mock circuits.

Implements the S0 experiment from
`research/experiment-plan/S0-heuristic-lesion-audit.md`.

Configs (dossier §3.1):
- FULL : weights=(1.0, 0.5, 0.3), seeds=("mm","ds","nt")
- DIV  : weights=(1.0, 0.0, 0.0), seeds=()
- -N   : weights=(1.0, 0.0, 0.3), seeds=("mm","ds","nt")
- -D   : weights=(1.0, 0.5, 0.0), seeds=("mm","ds","nt")
- -ND  : weights=(1.0, 0.0, 0.0), seeds=("mm","ds","nt")
- -S   : weights=(1.0, 0.5, 0.3), seeds=()
- -MM  : weights=(1.0, 0.5, 0.3), seeds=("ds","nt")
- -DS  : weights=(1.0, 0.5, 0.3), seeds=("mm","nt")
- -NT  : weights=(1.0, 0.5, 0.3), seeds=("mm","ds")

Mock circuits (3 of 5 S0 targets; TenSEAL LR + WDBC deferred
pending their adapters per dossier 02 §2.3 and 03 §3.3):
- Circuit 1: logistic regression (d=8, bounds=[-3,3]^8)
- Circuit 2: depth-4 polynomial (d=6, bounds=[-2,2]^6)
- Circuit 3: dense + Chebyshev sigmoid (d=10, bounds=[-3,3]^10)

Output CSV: benchmarks/results/ablation_heuristics.csv
Columns: config, circuit, seed, weights, seeds_used, max_error,
         worst_input, wall_clock_s, n_trials, verdict
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
from fhe_oracle.fitness import DivergenceFitness

from polynomial_eval import mock_fhe_poly, plaintext_poly


CONFIGS: dict[str, dict[str, Any]] = {
    "FULL": {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "ds", "nt")},
    "DIV":  {"w": (1.0, 0.0, 0.0), "seeds": ()},
    "-N":   {"w": (1.0, 0.0, 0.3), "seeds": ("mm", "ds", "nt")},
    "-D":   {"w": (1.0, 0.5, 0.0), "seeds": ("mm", "ds", "nt")},
    "-ND":  {"w": (1.0, 0.0, 0.0), "seeds": ("mm", "ds", "nt")},
    "-S":   {"w": (1.0, 0.5, 0.3), "seeds": ()},
    "-MM":  {"w": (1.0, 0.5, 0.3), "seeds": ("ds", "nt")},
    "-DS":  {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "nt")},
    "-NT":  {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "ds")},
}


# --- Circuit 1: logistic regression (d=8) ---

def make_circuit1():
    rng = np.random.default_rng(42)
    d = 8
    n_samples = 200
    X = rng.normal(0.0, 1.0, size=(n_samples, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    # Fit LR (copied from benchmarks/logistic_regression.py _fit_logistic)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b

    def plain(x: list[float]) -> float:
        z = float(np.dot(w, x) + b)
        return 1.0 / (1.0 + np.exp(-z))

    def fhe(x: list[float]) -> float:
        arr = np.asarray(x, dtype=np.float64)
        pval = plain(x)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        noise = float(local.normal(0.0, 1e-4))
        z_proxy = float(np.dot(arr, arr))
        if z_proxy > 4.0 and abs(pval - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 4.0)
            noise *= amp
        return pval + noise

    return {
        "name": "circuit1_lr",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


# --- Circuit 2: depth-4 polynomial (d=6) ---

def make_circuit2():
    d = 6
    return {
        "name": "circuit2_poly",
        "plain": plaintext_poly,
        "fhe": mock_fhe_poly,
        "d": d,
        "bounds": [(-2.0, 2.0)] * d,
    }


# --- Circuit 3: dense + Chebyshev sigmoid (d=10) ---

def make_circuit3():
    rng = np.random.default_rng(11)
    d = 10
    hidden = 4
    W = rng.normal(0.0, 0.5, size=(hidden, d))
    b = rng.normal(0.0, 0.1, size=hidden)

    def plain(x: list[float]) -> list[float]:
        arr = np.asarray(x, dtype=np.float64)
        z = W @ arr + b
        return (1.0 / (1.0 + np.exp(-z))).tolist()

    def fhe(x: list[float]) -> list[float]:
        arr = np.asarray(x, dtype=np.float64)
        z = W @ arr + b
        z_clip = np.clip(z / 5.0, -1.0, 1.0)
        approx = 0.5 + 0.5 * (1.5 * z_clip - 0.5 * z_clip ** 3)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        noise = local.normal(0.0, 1e-5, size=hidden)
        return (approx + noise).tolist()

    return {
        "name": "circuit3_cheb",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


def run_one_cell(
    config_name: str,
    cfg: dict[str, Any],
    circuit: dict[str, Any],
    seed: int,
    n_trials: int,
) -> dict[str, Any]:
    """Run a single (config, circuit, seed) cell.

    Uses a custom fitness that:
    - Applies the config's weights to a divergence/noise/depth combo.
      For mock adapters we don't have a real noise probe, so the noise
      and depth components are stubbed as functions of ||x|| and
      ||x||_inf respectively — the _lesion_ (setting weights to 0)
      still works correctly, which is the S0 experiment's point.
    - Is seeded with the config's heuristic subset via inject().
    """
    w_div, w_noise, w_depth = cfg["w"]
    seed_heuristics = cfg["seeds"]
    d = circuit["d"]
    bounds = circuit["bounds"]
    plain_fn = circuit["plain"]
    fhe_fn = circuit["fhe"]

    # Build a fitness that applies the per-config weights.
    class CfgFitness:
        def score(self, x):
            arr = np.asarray(x, dtype=np.float64)
            try:
                p = plain_fn(x)
                f = fhe_fn(x)
            except Exception:
                return 0.0
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            divergence = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
            # Mock noise/depth terms (monotone in input magnitude).
            # These are what get lesioned when w_noise/w_depth = 0.
            noise_term = min(1.0, float(np.linalg.norm(arr) / (np.sqrt(d) * 3.0)))
            depth_term = min(1.0, float(np.max(np.abs(arr)) / 3.0))
            return (
                w_div * divergence
                + w_noise * noise_term
                + w_depth * depth_term
            )

    k = 10 if seed_heuristics else 0

    oracle = FHEOracle(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_fn,
        input_dim=d,
        input_bounds=bounds,
        fitness=CfgFitness(),
        seed=seed,
        use_heuristic_seeds=bool(seed_heuristics),
        heuristic_which=tuple(seed_heuristics) if seed_heuristics else ("mm", "ds", "nt"),
        heuristic_k=k,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=n_trials, threshold=0.0)
    wall = time.perf_counter() - t0

    return {
        "config": config_name,
        "circuit": circuit["name"],
        "seed": seed,
        "weights": f"({w_div},{w_noise},{w_depth})",
        "seeds_used": ",".join(seed_heuristics) if seed_heuristics else "",
        "max_error": res.max_error,
        "worst_input": str(res.worst_input),
        "wall_clock_s": wall,
        "n_trials": res.n_trials,
        "verdict": res.verdict,
    }


def main(n_trials: int = 100, seeds: list[int] | None = None) -> int:
    if seeds is None:
        seeds = list(range(10))
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_heuristics.csv")

    total = len(CONFIGS) * len(circuits) * len(seeds)
    print(f"S0 Heuristic Lesion Audit")
    print(f"  Configs:  {len(CONFIGS)}  ({list(CONFIGS.keys())})")
    print(f"  Circuits: {len(circuits)} ({[c['name'] for c in circuits]})")
    print(f"  Seeds:    {len(seeds)} ({seeds})")
    print(f"  Budget:   B={n_trials} per cell")
    print(f"  Total:    {total} cells")
    print(f"  Output:   {out_path}")
    print("=" * 70)

    t_start = time.perf_counter()
    rows = []
    cell_count = 0
    for cfg_name, cfg in CONFIGS.items():
        for circuit in circuits:
            for seed in seeds:
                cell_count += 1
                row = run_one_cell(cfg_name, cfg, circuit, seed, n_trials)
                rows.append(row)
                print(
                    f"[{cell_count:3d}/{total}] "
                    f"{cfg_name:>5s}  {circuit['name']:<16s} "
                    f"seed={seed}  max_err={row['max_error']:.4e} "
                    f"t={row['wall_clock_s']:.2f}s"
                )

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Writing {len(rows)} rows to {out_path}")

    fieldnames = [
        "config", "circuit", "seed", "weights", "seeds_used",
        "max_error", "worst_input", "wall_clock_s", "n_trials", "verdict",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    raise SystemExit(main(n_trials=n))
