# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse restart sweep (A2) with paired Wilcoxon + Holm correction.

Reads benchmarks/results/restart_sweep.csv; emits
benchmarks/results/restart_summary.csv.

For each (config, circuit) with config != baseline, compare against both
`baseline` (pre-A1 CMA-ES) and `a1_only` (A1 hybrid, no restarts).

Verdict rule (same as S0/A1):
  LOAD_BEARING: median_ratio >= 1.10 AND wins >= 7/10 AND p_holm < 0.05
  INERT:        ratio in [0.95, 1.05] AND 3 <= wins <= 7
  INCONCLUSIVE: otherwise
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "restart_sweep.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "restart_summary.csv"
)


def holm_correction(p_values: list[float]) -> list[float]:
    if HAVE_STATSMODELS:
        _, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")
        return p_holm.tolist()
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    out = [0.0] * n
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, max(prev, p * (n - rank)))
        out[idx] = adj
        prev = adj
    return out


def classify_verdict(median_ratio: float, wins: int, p_holm: float) -> str:
    if median_ratio >= 1.10 and wins >= 7 and p_holm < 0.05:
        return "LOAD_BEARING"
    if 0.95 <= median_ratio <= 1.05 and 3 <= wins <= 7:
        return "INERT"
    return "INCONCLUSIVE"


def paired_wilcoxon(cell: np.ndarray, ref: np.ndarray) -> float:
    try:
        if np.all(cell == ref):
            return 1.0
        stat, p = wilcoxon(cell, ref, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing: {CSV_IN}. Run benchmarks/restart_sweep.py first.")
        return 1

    errors: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with open(CSV_IN) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            errors[(row["config_name"], row["circuit"])][int(row["seed"])] = float(
                row["max_error"]
            )

    configs = sorted({cfg for (cfg, _) in errors.keys()})
    circuits = sorted({c for (_, c) in errors.keys()})
    summary_rows: list[dict] = []
    p_vs_baseline: list[float] = []
    p_vs_a1: list[float] = []

    for cfg in configs:
        if cfg == "baseline":
            continue
        for circuit in circuits:
            base = errors.get(("baseline", circuit), {})
            a1 = errors.get(("a1_only", circuit), {})
            cell = errors.get((cfg, circuit), {})
            if not base or not cell:
                continue
            seeds = sorted(set(base.keys()) & set(cell.keys()))
            base_vals = np.array([base[s] for s in seeds])
            cell_vals = np.array([cell[s] for s in seeds])
            denom_b = np.where(base_vals > 1e-12, base_vals, 1e-12)
            ratio_vs_base = float(np.median(cell_vals / denom_b))
            wins_vs_base = int(np.sum(cell_vals > base_vals))
            p_base = paired_wilcoxon(cell_vals, base_vals)

            if a1 and cfg != "a1_only":
                seeds_a1 = sorted(set(a1.keys()) & set(cell.keys()))
                a1_vals = np.array([a1[s] for s in seeds_a1])
                cell_a1 = np.array([cell[s] for s in seeds_a1])
                denom_a = np.where(a1_vals > 1e-12, a1_vals, 1e-12)
                ratio_vs_a1 = float(np.median(cell_a1 / denom_a))
                wins_vs_a1 = int(np.sum(cell_a1 > a1_vals))
                p_a1 = paired_wilcoxon(cell_a1, a1_vals)
            else:
                ratio_vs_a1 = 1.0
                wins_vs_a1 = 0
                p_a1 = 1.0

            summary_rows.append(
                {
                    "config": cfg,
                    "circuit": circuit,
                    "n_seeds": len(seeds),
                    "baseline_median_err": float(np.median(base_vals)),
                    "cell_median_err": float(np.median(cell_vals)),
                    "ratio_vs_baseline": ratio_vs_base,
                    "wins_vs_baseline_of_10": wins_vs_base,
                    "p_vs_baseline_uncorr": p_base,
                    "ratio_vs_a1_only": ratio_vs_a1,
                    "wins_vs_a1_only_of_10": wins_vs_a1,
                    "p_vs_a1_uncorr": p_a1,
                }
            )
            p_vs_baseline.append(p_base)
            p_vs_a1.append(p_a1)

    p_holm_base = holm_correction(p_vs_baseline) if p_vs_baseline else []
    p_holm_a1 = holm_correction(p_vs_a1) if p_vs_a1 else []

    for row, ph_b, ph_a in zip(summary_rows, p_holm_base, p_holm_a1):
        row["p_vs_baseline_holm"] = float(ph_b)
        row["p_vs_a1_holm"] = float(ph_a)
        row["verdict_vs_baseline"] = classify_verdict(
            row["ratio_vs_baseline"], row["wins_vs_baseline_of_10"], ph_b
        )
        row["verdict_vs_a1"] = classify_verdict(
            row["ratio_vs_a1_only"], row["wins_vs_a1_only_of_10"], ph_a
        )

    fieldnames = [
        "config", "circuit", "n_seeds",
        "baseline_median_err", "cell_median_err",
        "ratio_vs_baseline", "wins_vs_baseline_of_10",
        "p_vs_baseline_uncorr", "p_vs_baseline_holm", "verdict_vs_baseline",
        "ratio_vs_a1_only", "wins_vs_a1_only_of_10",
        "p_vs_a1_uncorr", "p_vs_a1_holm", "verdict_vs_a1",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {len(summary_rows)} rows to {CSV_OUT}\n")

    # Human-readable
    print("=" * 120)
    print(
        f"{'config':<10s}  {'circuit':<16s}  "
        f"{'base_med':>10s}  {'cell_med':>10s}  "
        f"{'R(/base)':>9s}  {'wins/bs':>7s}  {'v_base':<14s}  "
        f"{'R(/a1)':>8s}  {'wins/a1':>7s}  {'v_a1':<14s}"
    )
    print("=" * 120)
    by_circuit: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_circuit[row["circuit"]].append(row)
    for circuit in sorted(by_circuit):
        for row in sorted(by_circuit[circuit], key=lambda r: r["config"]):
            print(
                f"{row['config']:<10s}  {row['circuit']:<16s}  "
                f"{row['baseline_median_err']:>10.4e}  "
                f"{row['cell_median_err']:>10.4e}  "
                f"{row['ratio_vs_baseline']:>9.3f}  "
                f"{row['wins_vs_baseline_of_10']:>3d}/10  "
                f"{row['verdict_vs_baseline']:<14s}  "
                f"{row['ratio_vs_a1_only']:>8.3f}  "
                f"{row['wins_vs_a1_only_of_10']:>3d}/10  "
                f"{row['verdict_vs_a1']:<14s}"
            )
        print("-" * 120)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
