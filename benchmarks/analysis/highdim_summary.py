# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse high-dimensional scaling sweep (B2).

Reads benchmarks/results/highdim_sweep.csv; emits
benchmarks/results/highdim_summary.csv.

Per (config, circuit, d):
  - median_max_error, mean_max_error
  - ratio vs uniform_random baseline (median of paired per-seed ratios)
  - wins vs random (out of n_seeds)
  - ratio vs full_cma (for non-full_cma configs)
  - wins vs full_cma

Primary questions answered by the printout:
  1. At what d does full_cma reach parity with random?
  2. Does sep_cma maintain advantage at d=64 and d=128?
  3. Does sep_warm dominate at high d?
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "highdim_sweep.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "highdim_summary.csv"
)


def _load() -> list[dict]:
    with open(CSV_IN) as fh:
        return list(csv.DictReader(fh))


def _seed_errs(rows: list[dict], config: str, circuit: str, d: int) -> np.ndarray:
    cells = [
        r for r in rows
        if r["config"] == config
        and r["circuit"] == circuit
        and int(r["d"]) == d
    ]
    by_seed = {int(r["seed"]): float(r["max_error"]) for r in cells}
    seeds = sorted(by_seed.keys())
    return np.array([by_seed[s] for s in seeds], dtype=np.float64)


def _paired_ratio(num: np.ndarray, den: np.ndarray) -> float:
    """Median paired ratio: median of (num_i / max(den_i, eps))."""
    if num.size == 0 or den.size == 0:
        return float("nan")
    n = min(num.size, den.size)
    eps = 1e-12
    denom = np.where(den[:n] > eps, den[:n], eps)
    ratios = num[:n] / denom
    return float(np.median(ratios))


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing: {CSV_IN}. Run benchmarks/highdim_sweep.py first.")
        return 1

    rows = _load()
    circuits = sorted({r["circuit"] for r in rows})
    dims = sorted({int(r["d"]) for r in rows})
    oracle_configs = ["full_cma", "sep_cma", "full_warm", "sep_warm"]

    summary_rows: list[dict] = []

    print("=" * 105)
    print("High-dim scaling sweep — median max_error + paired ratios")
    print("=" * 105)
    header = (
        f"{'circuit':<10s} {'d':>4s} {'config':<12s} "
        f"{'median_err':>12s} {'mean_err':>12s} "
        f"{'R/random':>10s} {'wins/rand':>11s} "
        f"{'R/full_cma':>11s} {'wins/full':>11s}"
    )
    print(header)
    print("-" * 105)

    for circuit in circuits:
        for d in dims:
            rand_errs = _seed_errs(rows, "uniform_random", circuit, d)
            full_errs = _seed_errs(rows, "full_cma", circuit, d)
            # Random baseline row
            summary_rows.append({
                "config": "uniform_random",
                "circuit": circuit,
                "d": d,
                "median_err": float(np.median(rand_errs)) if rand_errs.size else 0.0,
                "mean_err": float(np.mean(rand_errs)) if rand_errs.size else 0.0,
                "ratio_vs_random": 1.0,
                "wins_vs_random": 0,
                "ratio_vs_full_cma": "-",
                "wins_vs_full_cma": "-",
            })
            print(
                f"{circuit:<10s} {d:>4d} {'uniform_random':<12s} "
                f"{float(np.median(rand_errs)):>12.4e} "
                f"{float(np.mean(rand_errs)):>12.4e} "
                f"{'1.000':>10s} {'-':>11s} "
                f"{'-':>11s} {'-':>11s}"
            )
            for cfg in oracle_configs:
                errs = _seed_errs(rows, cfg, circuit, d)
                if errs.size == 0:
                    continue
                r_rand = _paired_ratio(errs, rand_errs)
                wins_rand = int(np.sum(errs > rand_errs))
                if cfg == "full_cma":
                    r_full = 1.0
                    wins_full = 0
                    r_full_str = "1.000"
                    wins_full_str = "-"
                else:
                    r_full = _paired_ratio(errs, full_errs)
                    wins_full = int(np.sum(errs > full_errs))
                    r_full_str = f"{r_full:.3f}"
                    wins_full_str = f"{wins_full}/{errs.size}"
                summary_rows.append({
                    "config": cfg,
                    "circuit": circuit,
                    "d": d,
                    "median_err": float(np.median(errs)),
                    "mean_err": float(np.mean(errs)),
                    "ratio_vs_random": r_rand,
                    "wins_vs_random": wins_rand,
                    "ratio_vs_full_cma": r_full_str,
                    "wins_vs_full_cma": wins_full_str,
                })
                print(
                    f"{circuit:<10s} {d:>4d} {cfg:<12s} "
                    f"{float(np.median(errs)):>12.4e} "
                    f"{float(np.mean(errs)):>12.4e} "
                    f"{r_rand:>10.3f} {wins_rand:>4d}/{errs.size:<6d} "
                    f"{r_full_str:>11s} {wins_full_str:>11s}"
                )
            print("-" * 105)

    print()
    print("Parity point analysis (full_cma vs uniform_random):")
    print("-" * 60)
    for circuit in circuits:
        print(f"  {circuit}:")
        for d in dims:
            r_row = next(
                (x for x in summary_rows
                 if x["config"] == "full_cma"
                 and x["circuit"] == circuit
                 and x["d"] == d),
                None,
            )
            if r_row is None:
                continue
            flag = ""
            r = r_row["ratio_vs_random"]
            if r < 1.10:
                flag = " <-- below 1.10x parity zone"
            if r < 1.0:
                flag = " <-- BELOW random"
            print(
                f"    d={d:>4d}  R(full_cma/random)={r:.3f}"
                f"  wins={r_row['wins_vs_random']}/10{flag}"
            )
        print()

    print("Sep-CMA-ES advantage at high d (sep_cma vs full_cma):")
    print("-" * 60)
    for circuit in circuits:
        print(f"  {circuit}:")
        for d in dims:
            r_row = next(
                (x for x in summary_rows
                 if x["config"] == "sep_cma"
                 and x["circuit"] == circuit
                 and x["d"] == d),
                None,
            )
            if r_row is None:
                continue
            flag = ""
            r = r_row["ratio_vs_full_cma"]
            try:
                rv = float(r)
                if rv >= 1.10:
                    flag = " <-- sep beats full (>=1.10x)"
                elif rv >= 1.0:
                    flag = " <-- sep matches full"
                else:
                    flag = " <-- full beats sep"
            except Exception:
                rv = None
            print(
                f"    d={d:>4d}  R(sep_cma/full_cma)={r}"
                f"  wins={r_row['wins_vs_full_cma']}{flag}"
            )
        print()

    fieldnames = [
        "config", "circuit", "d",
        "median_err", "mean_err",
        "ratio_vs_random", "wins_vs_random",
        "ratio_vs_full_cma", "wins_vs_full_cma",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {len(summary_rows)} rows to {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
