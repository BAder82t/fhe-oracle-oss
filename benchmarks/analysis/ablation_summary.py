# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse S0 ablation results with paired Wilcoxon + Holm correction.

Reads benchmarks/results/ablation_heuristics.csv and emits:
- benchmarks/results/ablation_summary.csv — per (config, circuit):
  median_ratio_vs_FULL, wins_vs_FULL, median_ratio_vs_DIV, wins_vs_DIV,
  p_vs_FULL, p_holm, verdict.

Verdict rule (dossier §4 / S0 §2):
- LOAD_BEARING: FULL/lesion median ratio >= 1.10 AND wins >= 7/10 AND p_holm < 0.05
- INERT:        FULL/lesion median ratio in [0.95, 1.05] AND 3 <= wins <= 7
- INCONCLUSIVE: otherwise (need 20-seed replication)
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "ablation_heuristics.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "ablation_summary.csv"
)

LESION_CONFIGS = ["DIV", "-N", "-D", "-ND", "-S", "-MM", "-DS", "-NT"]


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction (BH-step-down), fallback if statsmodels missing."""
    if HAVE_STATSMODELS:
        _, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")
        return p_holm.tolist()
    # Manual Holm: sort p-values, multiply each by (n - rank + 1), cap at 1.
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    out = [0.0] * n
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, max(prev, p * (n - rank)))
        out[idx] = adj
        prev = adj
    return out


def classify_verdict(
    median_ratio_vs_FULL: float, wins_of_10: int, p_holm: float
) -> str:
    """Dossier §4 verdict rule."""
    if median_ratio_vs_FULL >= 1.10 and wins_of_10 >= 7 and p_holm < 0.05:
        return "LOAD_BEARING"
    if 0.95 <= median_ratio_vs_FULL <= 1.05 and 3 <= wins_of_10 <= 7:
        return "INERT"
    return "INCONCLUSIVE"


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing input: {CSV_IN}")
        print("Run benchmarks/ablation_heuristics.py first.")
        return 1

    # errors[(cfg, circuit)][seed] = max_error
    errors: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with open(CSV_IN) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            errors[(row["config"], row["circuit"])][int(row["seed"])] = float(
                row["max_error"]
            )

    circuits = sorted({c for (_, c) in errors.keys()})
    summary_rows = []
    p_per_cfg_circuit = []

    for cfg in LESION_CONFIGS:
        for circuit in circuits:
            full_errs = errors.get(("FULL", circuit), {})
            lesion_errs = errors.get((cfg, circuit), {})
            if not full_errs or not lesion_errs:
                continue
            seeds = sorted(set(full_errs.keys()) & set(lesion_errs.keys()))
            if len(seeds) < 3:
                continue

            full_vals = np.array([full_errs[s] for s in seeds])
            lesion_vals = np.array([lesion_errs[s] for s in seeds])

            # Primary: median ratio FULL/lesion. >=1.10 means FULL is stronger → lesion is load-bearing.
            # Zero-lesion guard: clamp denominator.
            denom = np.where(lesion_vals > 1e-12, lesion_vals, 1e-12)
            ratios_vs_full = full_vals / denom
            median_ratio_vs_FULL = float(np.median(ratios_vs_full))

            # DIV is the paper's "current" baseline — also useful to track.
            div_errs = errors.get(("DIV", circuit), {})
            if div_errs and cfg != "DIV":
                div_seeds = sorted(set(div_errs.keys()) & set(lesion_errs.keys()))
                div_vals = np.array([div_errs[s] for s in div_seeds])
                les_for_div = np.array([lesion_errs[s] for s in div_seeds])
                denom_div = np.where(les_for_div > 1e-12, les_for_div, 1e-12)
                ratios_vs_div = les_for_div / np.where(
                    div_vals > 1e-12, div_vals, 1e-12
                )
                median_ratio_vs_DIV = float(np.median(ratios_vs_div))
            else:
                median_ratio_vs_DIV = 1.0

            wins_vs_FULL = int(np.sum(full_vals > lesion_vals))
            wins_vs_DIV = 0
            if div_errs and cfg != "DIV":
                div_seeds = sorted(set(div_errs.keys()) & set(lesion_errs.keys()))
                div_vals = np.array([div_errs[s] for s in div_seeds])
                les_for_div = np.array([lesion_errs[s] for s in div_seeds])
                wins_vs_DIV = int(np.sum(les_for_div > div_vals))

            # Paired one-sided Wilcoxon: H1 full > lesion.
            try:
                if np.all(full_vals == lesion_vals):
                    p_uncorrected = 1.0
                else:
                    stat, p_uncorrected = wilcoxon(
                        full_vals, lesion_vals, alternative="greater"
                    )
                    p_uncorrected = float(p_uncorrected)
            except ValueError:
                p_uncorrected = 1.0

            summary_rows.append(
                {
                    "config": cfg,
                    "circuit": circuit,
                    "n_seeds": len(seeds),
                    "full_median_err": float(np.median(full_vals)),
                    "lesion_median_err": float(np.median(lesion_vals)),
                    "median_ratio_full_over_lesion": median_ratio_vs_FULL,
                    "median_ratio_lesion_over_div": median_ratio_vs_DIV,
                    "wins_full_over_lesion_of_10": wins_vs_FULL,
                    "wins_lesion_over_div_of_10": wins_vs_DIV,
                    "p_uncorrected": p_uncorrected,
                }
            )
            p_per_cfg_circuit.append(p_uncorrected)

    # Apply Holm correction across all (cfg, circuit) comparisons.
    if p_per_cfg_circuit:
        p_holm = holm_correction(p_per_cfg_circuit)
    else:
        p_holm = []

    for row, p_h in zip(summary_rows, p_holm):
        row["p_holm"] = float(p_h)
        row["verdict"] = classify_verdict(
            row["median_ratio_full_over_lesion"],
            row["wins_full_over_lesion_of_10"],
            row["p_holm"],
        )

    # Emit CSV.
    fieldnames = [
        "config", "circuit", "n_seeds",
        "full_median_err", "lesion_median_err",
        "median_ratio_full_over_lesion", "median_ratio_lesion_over_div",
        "wins_full_over_lesion_of_10", "wins_lesion_over_div_of_10",
        "p_uncorrected", "p_holm", "verdict",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {len(summary_rows)} summary rows to {CSV_OUT}")

    # Print human-readable summary.
    print()
    print("=" * 110)
    print(
        f"{'CFG':>5s} {'CIRCUIT':<18s} {'FULL_med':>10s} {'LES_med':>10s} "
        f"{'R(F/L)':>8s} {'wins':>6s} {'p_holm':>9s} {'verdict':<15s}"
    )
    print("=" * 110)
    by_circuit = defaultdict(list)
    for row in summary_rows:
        by_circuit[row["circuit"]].append(row)
    for circuit in sorted(by_circuit.keys()):
        for row in by_circuit[circuit]:
            print(
                f"{row['config']:>5s} {row['circuit']:<18s} "
                f"{row['full_median_err']:>10.4e} {row['lesion_median_err']:>10.4e} "
                f"{row['median_ratio_full_over_lesion']:>8.3f} "
                f"{row['wins_full_over_lesion_of_10']:>3d}/10  "
                f"{row['p_holm']:>9.3e} {row['verdict']:<15s}"
            )
        print("-" * 110)

    # Count verdicts per config.
    print()
    verdict_counts = defaultdict(lambda: defaultdict(int))
    for row in summary_rows:
        verdict_counts[row["config"]][row["verdict"]] += 1
    print(f"{'CFG':>5s} {'LOAD_BEARING':>14s} {'INERT':>8s} {'INCONCLUSIVE':>14s}")
    print("-" * 60)
    for cfg in LESION_CONFIGS:
        vc = verdict_counts[cfg]
        print(
            f"{cfg:>5s} {vc.get('LOAD_BEARING', 0):>14d} "
            f"{vc.get('INERT', 0):>8d} {vc.get('INCONCLUSIVE', 0):>14d}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
