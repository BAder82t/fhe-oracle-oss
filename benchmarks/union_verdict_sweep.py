# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Union verdict sweep (A3).

Configurations on WDBC mock (d=30) — the primary target:
  wdbc_oracle   : FHEOracle only, ρ=0.3, B=500
  wdbc_empirical: EmpiricalSearch only, B=500, jitter=0.1
  wdbc_hybrid   : run_hybrid with oracle B=250 + empirical B=250
  wdbc_hybrid_eq: run_hybrid with oracle B=500 + empirical B=500

Regression checks on 3 prior mock circuits (no empirical data):
  circuit1_lr / circuit2_poly / circuit3_cheb under oracle_only at
  ρ=0.3 — must match A1 baseline numbers.

10 seeds per config.

Outputs: benchmarks/results/union_verdict_sweep.csv
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
from fhe_oracle.empirical import EmpiricalSearch
from fhe_oracle.hybrid import run_hybrid
from ablation_heuristics import make_circuit1, make_circuit2, make_circuit3
from wdbc_mock import build_wdbc_circuit


WDBC_THRESHOLD = 0.01    # spec §Key design decision 2
WDBC_JITTER = 0.1         # spec §Key design decision 3
SEEDS = list(range(10))


def run_wdbc_configs(seeds: list[int], budget_full: int, budget_half: int) -> list[dict]:
    plain, fhe, data, w, b, d = build_wdbc_circuit(random_state=42)
    bounds = [(-3.0, 3.0)] * d

    rows = []
    for seed in seeds:
        # --- wdbc_oracle: oracle only, full budget, ρ=0.3 ---
        t0 = time.perf_counter()
        oracle = FHEOracle(
            plaintext_fn=plain, fhe_fn=fhe,
            input_dim=d, input_bounds=bounds,
            seed=seed, random_floor=0.3,
        )
        o_res = oracle.run(n_trials=budget_full, threshold=WDBC_THRESHOLD)
        t_oracle = time.perf_counter() - t0
        rows.append({
            "config": "wdbc_oracle", "circuit": "wdbc_lr", "seed": seed,
            "max_error": o_res.max_error, "verdict": o_res.verdict,
            "source": "oracle",
            "oracle_max_error": o_res.max_error,
            "empirical_max_error": 0.0,
            "wall_clock_s": t_oracle,
        })

        # --- wdbc_empirical: empirical only, full budget ---
        from fhe_oracle.hybrid import _default_divergence_fn
        div_fn = _default_divergence_fn(plain, fhe)
        t0 = time.perf_counter()
        emp = EmpiricalSearch(
            divergence_fn=div_fn, data=data,
            threshold=WDBC_THRESHOLD, budget=budget_full,
            jitter_std=WDBC_JITTER, seed=seed + 100,
        )
        e_res = emp.run()
        t_emp = time.perf_counter() - t0
        rows.append({
            "config": "wdbc_empirical", "circuit": "wdbc_lr", "seed": seed,
            "max_error": e_res.max_error, "verdict": e_res.verdict,
            "source": "empirical",
            "oracle_max_error": 0.0,
            "empirical_max_error": e_res.max_error,
            "wall_clock_s": t_emp,
        })

        # --- wdbc_hybrid (equal split half budget each) ---
        t0 = time.perf_counter()
        h_res = run_hybrid(
            plaintext_fn=plain, fhe_fn=fhe,
            input_dim=d, input_bounds=bounds,
            threshold=WDBC_THRESHOLD,
            oracle_budget=budget_half, oracle_seed=seed,
            random_floor=0.3,
            data=data, empirical_budget=budget_half,
            jitter_std=WDBC_JITTER, empirical_seed=seed + 100,
        )
        t_h = time.perf_counter() - t0
        rows.append({
            "config": "wdbc_hybrid", "circuit": "wdbc_lr", "seed": seed,
            "max_error": h_res.max_error, "verdict": h_res.union_verdict,
            "source": h_res.source,
            "oracle_max_error": h_res.oracle_result.max_error,
            "empirical_max_error": h_res.empirical_result.max_error,
            "wall_clock_s": t_h,
        })

        # --- wdbc_hybrid_eq (full budget each, 2x total) ---
        t0 = time.perf_counter()
        h2_res = run_hybrid(
            plaintext_fn=plain, fhe_fn=fhe,
            input_dim=d, input_bounds=bounds,
            threshold=WDBC_THRESHOLD,
            oracle_budget=budget_full, oracle_seed=seed,
            random_floor=0.3,
            data=data, empirical_budget=budget_full,
            jitter_std=WDBC_JITTER, empirical_seed=seed + 100,
        )
        t_h2 = time.perf_counter() - t0
        rows.append({
            "config": "wdbc_hybrid_eq", "circuit": "wdbc_lr", "seed": seed,
            "max_error": h2_res.max_error, "verdict": h2_res.union_verdict,
            "source": h2_res.source,
            "oracle_max_error": h2_res.oracle_result.max_error,
            "empirical_max_error": h2_res.empirical_result.max_error,
            "wall_clock_s": t_h2,
        })

    return rows


def run_regression_configs(seeds: list[int], budget: int) -> list[dict]:
    """Oracle-only on the 3 prior mock circuits — A1 baseline regression check."""
    rows = []
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]
    for circuit in circuits:
        for seed in seeds:
            oracle = FHEOracle(
                plaintext_fn=circuit["plain"],
                fhe_fn=circuit["fhe"],
                input_dim=circuit["d"],
                input_bounds=circuit["bounds"],
                seed=seed,
                random_floor=0.3,
            )
            t0 = time.perf_counter()
            res = oracle.run(n_trials=budget, threshold=0.0)
            rows.append({
                "config": "oracle_only_a1",
                "circuit": circuit["name"],
                "seed": seed,
                "max_error": res.max_error,
                "verdict": res.verdict,
                "source": "oracle",
                "oracle_max_error": res.max_error,
                "empirical_max_error": 0.0,
                "wall_clock_s": time.perf_counter() - t0,
            })
    return rows


def main(budget_full: int = 500, budget_half: int = 250) -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "union_verdict_sweep.csv")

    print("Union verdict sweep (A3)")
    print(f"  Full budget : {budget_full}")
    print(f"  Half budget : {budget_half}")
    print(f"  Threshold   : {WDBC_THRESHOLD}")
    print(f"  Jitter      : {WDBC_JITTER}")
    print(f"  Seeds       : {SEEDS}")
    print("=" * 80)

    t_start = time.perf_counter()

    print("[1/2] WDBC (4 configs × 10 seeds)...")
    wdbc_rows = run_wdbc_configs(SEEDS, budget_full, budget_half)
    print(f"       {len(wdbc_rows)} cells")

    print("[2/2] Regression on circuits 1-3 (1 config × 3 circuits × 10 seeds)...")
    reg_rows = run_regression_configs(SEEDS, budget_full)
    print(f"       {len(reg_rows)} cells")

    all_rows = wdbc_rows + reg_rows
    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    fieldnames = [
        "config", "circuit", "seed", "max_error", "verdict", "source",
        "oracle_max_error", "empirical_max_error", "wall_clock_s",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {out_path}")

    # --- Summary table ---
    print()
    print("WDBC summary")
    print("=" * 95)
    print(
        f"{'config':<18s}  {'median_err':>12s}  {'mean_err':>12s}  "
        f"{'FAIL/10':>8s}  {'oracle_max_median':>18s}  {'empirical_max_median':>20s}"
    )
    print("-" * 95)
    for cfg in ["wdbc_oracle", "wdbc_empirical", "wdbc_hybrid", "wdbc_hybrid_eq"]:
        cell_rows = [r for r in wdbc_rows if r["config"] == cfg]
        errs = [r["max_error"] for r in cell_rows]
        fails = sum(1 for r in cell_rows if r["verdict"] == "FAIL")
        o_errs = [r["oracle_max_error"] for r in cell_rows]
        e_errs = [r["empirical_max_error"] for r in cell_rows]
        print(
            f"{cfg:<18s}  {median(errs):>12.4e}  "
            f"{sum(errs) / len(errs):>12.4e}  "
            f"{fails:>3d}/10  "
            f"{median(o_errs):>18.4e}  "
            f"{median(e_errs):>20.4e}"
        )
    print("-" * 95)

    # Regression summary for circuits 1-3
    print()
    print("Regression on circuits 1-3 (oracle_only_a1, should match A1 baseline)")
    print("=" * 60)
    for circuit_name in sorted({r["circuit"] for r in reg_rows}):
        cell_rows = [r for r in reg_rows if r["circuit"] == circuit_name]
        errs = [r["max_error"] for r in cell_rows]
        print(
            f"  {circuit_name:<16s}  median={median(errs):.4e}  "
            f"mean={sum(errs) / len(errs):.4e}"
        )

    return 0


if __name__ == "__main__":
    bf = 500
    bh = 250
    if len(sys.argv) > 1:
        bf = int(sys.argv[1])
        bh = bf // 2
    raise SystemExit(main(budget_full=bf, budget_half=bh))
