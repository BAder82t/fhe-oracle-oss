# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Hybrid random-floor + CMA-ES warm-start sweep (A1 + A4).

Runs ρ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} × 3 circuits × 10 seeds × B=500.

Circuits reuse the three mock circuits from ablation_heuristics.py:
- circuit1_lr       (LR hot-zone, d=8)
- circuit2_poly     (depth-4 polynomial, d=6)
- circuit3_cheb     (dense + Chebyshev sigmoid, d=10)

Primary target: circuit3_cheb at ρ=0.3 should lift wins/10 above 0/10
(the paper's plateau-trap result at `tex:1142-1168`).

Outputs: benchmarks/results/hybrid_warmstart.csv
Columns: rho, circuit, seed, max_error, worst_input, wall_clock_s,
         n_trials, verdict, cert_budget_rand, cert_hits, cert_mu_hat,
         cert_p_discovery
"""

from __future__ import annotations

import csv
import os
import sys
import time
from statistics import median

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle import FHEOracle
from ablation_heuristics import make_circuit1, make_circuit2, make_circuit3


RHOS = [0.0, 0.1, 0.3, 0.5, 1.0]


def run_one_cell(rho: float, circuit: dict, seed: int, n_trials: int) -> dict:
    """Run a single (rho, circuit, seed) cell."""
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    d = circuit["d"]
    bounds = circuit["bounds"]

    oracle = FHEOracle(
        plaintext_fn=plain,
        fhe_fn=fhe,
        input_dim=d,
        input_bounds=bounds,
        seed=seed,
        random_floor=rho,
        warm_start=True,
        warm_sigma_scale=0.2,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=n_trials, threshold=0.0)
    wall = time.perf_counter() - t0

    cert = res.coverage_certificate
    return {
        "rho": rho,
        "circuit": circuit["name"],
        "seed": seed,
        "max_error": res.max_error,
        "worst_input": str(res.worst_input),
        "wall_clock_s": wall,
        "n_trials": res.n_trials,
        "verdict": res.verdict,
        "cert_budget_rand": cert.budget_rand if cert else 0,
        "cert_hits": cert.hits if cert else 0,
        "cert_mu_hat": cert.mu_hat if cert else 0.0,
        "cert_p_discovery": cert.p_discovery if cert else 0.0,
    }


def main(n_trials: int = 500, seeds: list[int] | None = None) -> int:
    if seeds is None:
        seeds = list(range(10))
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hybrid_warmstart.csv")

    total = len(RHOS) * len(circuits) * len(seeds)
    print("Hybrid random-floor + CMA-ES warm-start sweep (A1 + A4)")
    print(f"  ρ values: {RHOS}")
    print(f"  Circuits: {len(circuits)} ({[c['name'] for c in circuits]})")
    print(f"  Seeds:    {len(seeds)} ({seeds})")
    print(f"  Budget:   B={n_trials}")
    print(f"  Total:    {total} cells")
    print(f"  Output:   {out_path}")
    print("=" * 80)

    t_start = time.perf_counter()
    rows = []
    cell = 0
    for rho in RHOS:
        for circuit in circuits:
            for seed in seeds:
                cell += 1
                row = run_one_cell(rho, circuit, seed, n_trials)
                rows.append(row)
                print(
                    f"[{cell:3d}/{total}] ρ={rho:.1f}  "
                    f"{row['circuit']:<16s} seed={seed} "
                    f"max_err={row['max_error']:.4e} "
                    f"n={row['n_trials']} t={row['wall_clock_s']:.2f}s"
                )

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    fieldnames = [
        "rho", "circuit", "seed", "max_error", "worst_input",
        "wall_clock_s", "n_trials", "verdict",
        "cert_budget_rand", "cert_hits", "cert_mu_hat", "cert_p_discovery",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # --- Summary table ---
    print()
    print("Summary: median max_error and wins vs ρ=0.0 per (ρ, circuit)")
    print("=" * 80)
    print(
        f"{'ρ':>5s}  {'circuit':<16s}  {'median_err':>14s}  "
        f"{'mean_err':>14s}  {'wins/10':>8s}"
    )
    print("-" * 80)

    for circuit_name in sorted({c["name"] for c in circuits}):
        baseline_errs = {
            r["seed"]: r["max_error"]
            for r in rows
            if r["rho"] == 0.0 and r["circuit"] == circuit_name
        }
        for rho in RHOS:
            cell_errs = [
                r["max_error"]
                for r in rows
                if r["rho"] == rho and r["circuit"] == circuit_name
            ]
            cell_seeds = [
                r["seed"]
                for r in rows
                if r["rho"] == rho and r["circuit"] == circuit_name
            ]
            if rho == 0.0:
                wins = "-"
            else:
                wins_count = sum(
                    1
                    for s, e in zip(cell_seeds, cell_errs)
                    if e > baseline_errs.get(s, 0.0)
                )
                wins = f"{wins_count}/10"
            print(
                f"{rho:>5.1f}  {circuit_name:<16s}  "
                f"{median(cell_errs):>14.6e}  "
                f"{sum(cell_errs) / len(cell_errs):>14.6e}  "
                f"{wins:>8s}"
            )
        print("-" * 80)

    return 0


if __name__ == "__main__":
    n = 500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    raise SystemExit(main(n_trials=n))
