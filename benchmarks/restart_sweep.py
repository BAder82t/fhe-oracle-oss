# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""IPOP/BIPOP restart sweep (A2).

5 configurations × 3 circuits × 10 seeds × B=500:

  baseline  : ρ=0.0, restarts=0                    — pre-A1 pure CMA-ES
  a1_only   : ρ=0.3, restarts=0                    — A1 hybrid, no restarts
  ipop_bare : ρ=0.0, restarts=3, bipop=False       — IPOP only, no floor
  a1_ipop   : ρ=0.3, restarts=3, bipop=False       — A1 + IPOP composed
  a1_bipop  : ρ=0.3, restarts=3, bipop=True        — A1 + BIPOP composed

Primary target: Chebyshev wins/10 ≥ 9 under `a1_ipop` or `a1_bipop`
(up from 7/10 at A1-only). A2 gate criterion.

Secondary: check LR doesn't regress further vs baseline.

Outputs: benchmarks/results/restart_sweep.csv
Columns: config_name, rho, restarts, bipop, circuit, seed,
         max_error, n_restarts_used, wall_clock_s, n_trials,
         cert_budget_rand, cert_hits, cert_mu_hat, cert_p_discovery
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


CONFIGS = {
    "baseline":  dict(rho=0.0, restarts=0, bipop=False),
    "a1_only":   dict(rho=0.3, restarts=0, bipop=False),
    "ipop_bare": dict(rho=0.0, restarts=3, bipop=False),
    "a1_ipop":   dict(rho=0.3, restarts=3, bipop=False),
    "a1_bipop":  dict(rho=0.3, restarts=3, bipop=True),
}


def run_one_cell(config_name: str, cfg: dict, circuit: dict,
                 seed: int, n_trials: int) -> dict:
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
        random_floor=cfg["rho"],
        warm_start=True,
        warm_sigma_scale=0.2,
        restarts=cfg["restarts"],
        bipop=cfg["bipop"],
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=n_trials, threshold=0.0)
    wall = time.perf_counter() - t0

    cert = res.coverage_certificate
    return {
        "config_name": config_name,
        "rho": cfg["rho"],
        "restarts": cfg["restarts"],
        "bipop": cfg["bipop"],
        "circuit": circuit["name"],
        "seed": seed,
        "max_error": res.max_error,
        "n_restarts_used": res.n_restarts_used,
        "wall_clock_s": wall,
        "n_trials": res.n_trials,
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
    out_path = os.path.join(out_dir, "restart_sweep.csv")

    total = len(CONFIGS) * len(circuits) * len(seeds)
    print("IPOP/BIPOP restart sweep (A2)")
    print(f"  Configs:  {len(CONFIGS)} ({list(CONFIGS.keys())})")
    print(f"  Circuits: {len(circuits)} ({[c['name'] for c in circuits]})")
    print(f"  Seeds:    {len(seeds)} ({seeds})")
    print(f"  Budget:   B={n_trials}")
    print(f"  Total:    {total} cells")
    print(f"  Output:   {out_path}")
    print("=" * 80)

    t_start = time.perf_counter()
    rows = []
    cell = 0
    for cfg_name, cfg in CONFIGS.items():
        for circuit in circuits:
            for seed in seeds:
                cell += 1
                row = run_one_cell(cfg_name, cfg, circuit, seed, n_trials)
                rows.append(row)
                print(
                    f"[{cell:3d}/{total}] {cfg_name:<10s} "
                    f"{row['circuit']:<16s} seed={seed} "
                    f"max_err={row['max_error']:.4e} "
                    f"restarts={row['n_restarts_used']} "
                    f"n={row['n_trials']} t={row['wall_clock_s']:.2f}s"
                )

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    fieldnames = [
        "config_name", "rho", "restarts", "bipop",
        "circuit", "seed", "max_error", "n_restarts_used",
        "wall_clock_s", "n_trials",
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
    print(
        "Summary: median/mean max_error per (config, circuit); "
        "wins vs baseline and vs a1_only"
    )
    print("=" * 110)
    print(
        f"{'config':<10s}  {'circuit':<16s}  "
        f"{'median_err':>12s}  {'mean_err':>12s}  "
        f"{'wins/base':>10s}  {'wins/a1':>8s}"
    )
    print("-" * 110)
    for circuit_name in sorted({c["name"] for c in circuits}):
        baseline = {
            r["seed"]: r["max_error"]
            for r in rows
            if r["config_name"] == "baseline" and r["circuit"] == circuit_name
        }
        a1_only = {
            r["seed"]: r["max_error"]
            for r in rows
            if r["config_name"] == "a1_only" and r["circuit"] == circuit_name
        }
        for cfg_name in CONFIGS:
            cell_rows = [
                r for r in rows
                if r["config_name"] == cfg_name and r["circuit"] == circuit_name
            ]
            errs = [r["max_error"] for r in cell_rows]
            if cfg_name == "baseline":
                wins_base = "-"
            else:
                wins_base_count = sum(
                    1 for r in cell_rows
                    if r["max_error"] > baseline.get(r["seed"], 0.0)
                )
                wins_base = f"{wins_base_count}/10"
            if cfg_name == "a1_only":
                wins_a1 = "-"
            else:
                wins_a1_count = sum(
                    1 for r in cell_rows
                    if r["max_error"] > a1_only.get(r["seed"], 0.0)
                )
                wins_a1 = f"{wins_a1_count}/10"
            print(
                f"{cfg_name:<10s}  {circuit_name:<16s}  "
                f"{median(errs):>12.4e}  "
                f"{sum(errs)/len(errs):>12.4e}  "
                f"{wins_base:>10s}  {wins_a1:>8s}"
            )
        print("-" * 110)

    return 0


if __name__ == "__main__":
    n = 500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    raise SystemExit(main(n_trials=n))
