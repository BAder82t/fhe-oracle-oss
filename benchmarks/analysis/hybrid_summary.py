# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse hybrid-warm-start sweep with paired Wilcoxon + Holm correction.

Reads benchmarks/results/hybrid_warmstart.csv and emits:
- benchmarks/results/hybrid_summary.csv — per (ρ, circuit):
  median_max_error, mean_max_error, median_ratio_vs_baseline,
  wins_vs_baseline, p_uncorrected, p_holm, verdict.

Verdict rule (dossier §4 / A1 §2):
- LOAD_BEARING: median_ratio >= 1.10 AND wins >= 7/10 AND p_holm < 0.05
- INERT:        ratio in [0.95, 1.05] AND 3 <= wins <= 7
- INCONCLUSIVE: otherwise
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
    os.path.dirname(__file__), "..", "results", "hybrid_warmstart.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "hybrid_summary.csv"
)

BASELINE_RHO = 0.0


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


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing input: {CSV_IN}")
        print("Run benchmarks/hybrid_warmstart.py first.")
        return 1

    errors: dict[tuple[float, str], dict[int, float]] = defaultdict(dict)
    with open(CSV_IN) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            errors[(float(row["rho"]), row["circuit"])][int(row["seed"])] = float(
                row["max_error"]
            )

    rhos_all = sorted({r for (r, _) in errors.keys()})
    circuits = sorted({c for (_, c) in errors.keys()})
    summary_rows: list[dict] = []
    p_values: list[float] = []

    for rho in rhos_all:
        if rho == BASELINE_RHO:
            continue
        for circuit in circuits:
            base = errors.get((BASELINE_RHO, circuit), {})
            cell = errors.get((rho, circuit), {})
            if not base or not cell:
                continue
            seeds = sorted(set(base.keys()) & set(cell.keys()))
            if len(seeds) < 3:
                continue
            base_vals = np.array([base[s] for s in seeds])
            cell_vals = np.array([cell[s] for s in seeds])
            # Guard against divide-by-zero when baseline is flat
            denom = np.where(base_vals > 1e-12, base_vals, 1e-12)
            ratios = cell_vals / denom
            median_ratio = float(np.median(ratios))
            wins = int(np.sum(cell_vals > base_vals))

            try:
                if np.all(cell_vals == base_vals):
                    p_uncorr = 1.0
                else:
                    stat, p_uncorr = wilcoxon(
                        cell_vals, base_vals, alternative="greater"
                    )
                    p_uncorr = float(p_uncorr)
            except ValueError:
                p_uncorr = 1.0

            summary_rows.append(
                {
                    "rho": rho,
                    "circuit": circuit,
                    "n_seeds": len(seeds),
                    "baseline_median_err": float(np.median(base_vals)),
                    "cell_median_err": float(np.median(cell_vals)),
                    "median_ratio_cell_over_baseline": median_ratio,
                    "wins_cell_over_baseline_of_10": wins,
                    "p_uncorrected": p_uncorr,
                }
            )
            p_values.append(p_uncorr)

    if p_values:
        p_holm = holm_correction(p_values)
    else:
        p_holm = []

    for row, p_h in zip(summary_rows, p_holm):
        row["p_holm"] = float(p_h)
        row["verdict"] = classify_verdict(
            row["median_ratio_cell_over_baseline"],
            row["wins_cell_over_baseline_of_10"],
            row["p_holm"],
        )

    fieldnames = [
        "rho", "circuit", "n_seeds",
        "baseline_median_err", "cell_median_err",
        "median_ratio_cell_over_baseline", "wins_cell_over_baseline_of_10",
        "p_uncorrected", "p_holm", "verdict",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {len(summary_rows)} rows to {CSV_OUT}\n")

    # Human-readable table
    print("=" * 100)
    print(
        f"{'ρ':>5s}  {'circuit':<16s}  "
        f"{'base_med':>12s}  {'cell_med':>12s}  "
        f"{'R(cell/base)':>12s}  {'wins':>5s}  "
        f"{'p_holm':>9s}  {'verdict':<14s}"
    )
    print("=" * 100)
    by_circuit: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_circuit[row["circuit"]].append(row)
    for circuit in sorted(by_circuit):
        for row in sorted(by_circuit[circuit], key=lambda r: r["rho"]):
            print(
                f"{row['rho']:>5.1f}  {row['circuit']:<16s}  "
                f"{row['baseline_median_err']:>12.4e}  "
                f"{row['cell_median_err']:>12.4e}  "
                f"{row['median_ratio_cell_over_baseline']:>12.3f}  "
                f"{row['wins_cell_over_baseline_of_10']:>2d}/10  "
                f"{row['p_holm']:>9.3e}  {row['verdict']:<14s}"
            )
        print("-" * 100)

    # Verdict counts
    print()
    verdict_counts: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in summary_rows:
        verdict_counts[row["rho"]][row["verdict"]] += 1
    print(f"{'ρ':>5s}  {'LOAD_BEARING':>14s}  {'INERT':>8s}  {'INCONCLUSIVE':>14s}")
    print("-" * 50)
    for rho in sorted(verdict_counts):
        vc = verdict_counts[rho]
        print(
            f"{rho:>5.1f}  {vc.get('LOAD_BEARING', 0):>14d}  "
            f"{vc.get('INERT', 0):>8d}  {vc.get('INCONCLUSIVE', 0):>14d}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
