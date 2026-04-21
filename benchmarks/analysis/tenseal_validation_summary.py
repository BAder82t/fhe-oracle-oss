# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse TenSEAL real-CKKS validation (B1).

Reads benchmarks/results/tenseal_validation.csv + tenseal_ablation.csv;
emits benchmarks/results/tenseal_validation_summary.csv + stdout report
comparing our measurements to paper Tables 4 and 5.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


CSV_VALIDATION = os.path.join(
    os.path.dirname(__file__), "..", "results", "tenseal_validation.csv"
)
CSV_ABLATION = os.path.join(
    os.path.dirname(__file__), "..", "results", "tenseal_ablation.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "tenseal_validation_summary.csv"
)

PAPER_EXPECTATIONS = {
    "lr_matched":       {"ratio_mean": 2.28, "wins": 7, "note": "Table 4 row 1"},
    "cheb_matched":     {"ratio_mean": 0.68, "wins": 0, "note": "Table 4 row 3"},
    "cheb_warm":        {"ratio_mean": None, "wins": None, "note": "new (A1 mitigation)"},
    "wdbc_matched":     {"ratio_mean": 1.74, "ratio_median": 1.14, "wins": 5, "note": "Table 5 matched"},
    "wdbc_asymmetric":  {"ratio_mean": 0.41, "ratio_median": 0.25, "wins": 1, "note": "Table 5 asymmetric"},
    "wdbc_hybrid":      {"note": "new (A3 union verdict)"},
}


def main() -> int:
    if not os.path.exists(CSV_VALIDATION):
        print(f"Missing {CSV_VALIDATION}; run benchmarks/tenseal_validation.py")
        return 1

    rows: list[dict] = []
    with open(CSV_VALIDATION) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r)

    summary_rows = []
    print("=" * 110)
    print("Real-CKKS validation vs paper")
    print("=" * 110)
    print(
        f"{'experiment':<18s}  {'n':>3s}  {'our median':>12s}  {'our mean':>12s}  "
        f"{'our wins':>10s}  {'paper note':<28s}"
    )
    print("-" * 110)

    for exp, exp_rows in sorted(by_exp.items()):
        if exp in ("wdbc_asymmetric", "wdbc_hybrid"):
            # Oracle vs empirical
            o_vals = np.array([float(r["oracle_max_error"]) for r in exp_rows])
            e_vals = np.array([float(r["empirical_max_error"]) for r in exp_rows])
            ratios = o_vals / np.where(e_vals > 1e-12, e_vals, 1e-12)
            median_ratio = float(np.median(ratios))
            mean_ratio = float(np.mean(ratios))
            wins = int(np.sum(o_vals > e_vals))
            paper = PAPER_EXPECTATIONS.get(exp, {})
            paper_str = paper.get("note", "")
            if "ratio_median" in paper:
                paper_str = f"median {paper['ratio_median']} / wins {paper['wins']}/10 ({paper_str})"
            try:
                p = float(wilcoxon(e_vals, o_vals, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            summary_rows.append({
                "experiment": exp,
                "n_seeds": len(exp_rows),
                "median_ratio_oracle_over_empirical": median_ratio,
                "mean_ratio_oracle_over_empirical": mean_ratio,
                "wins_oracle_over_empirical": wins,
                "p_uncorrected_empirical_greater": p,
                "paper_note": paper_str,
            })
            print(
                f"{exp:<18s}  {len(exp_rows):>3d}  {median_ratio:>12.4f}  "
                f"{mean_ratio:>12.4f}  {wins:>4d}/10 (O>E)  {paper_str:<28s}"
            )
        else:
            # Oracle vs random
            o_vals = np.array([float(r["oracle_max_error"]) for r in exp_rows])
            r_vals = np.array([float(r["random_max_error"]) for r in exp_rows])
            ratios = o_vals / np.where(r_vals > 1e-12, r_vals, 1e-12)
            median_ratio = float(np.median(ratios))
            mean_ratio = float(np.mean(ratios))
            wins = int(np.sum(o_vals > r_vals))
            paper = PAPER_EXPECTATIONS.get(exp, {})
            paper_str = paper.get("note", "")
            try:
                p = float(wilcoxon(o_vals, r_vals, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            summary_rows.append({
                "experiment": exp,
                "n_seeds": len(exp_rows),
                "median_ratio_oracle_over_random": median_ratio,
                "mean_ratio_oracle_over_random": mean_ratio,
                "wins_oracle_over_random": wins,
                "p_uncorrected_oracle_greater": p,
                "paper_note": paper_str,
            })
            print(
                f"{exp:<18s}  {len(exp_rows):>3d}  {median_ratio:>12.4f}  "
                f"{mean_ratio:>12.4f}  {wins:>4d}/10        {paper_str:<28s}"
            )

    print("-" * 110)

    fieldnames = [
        "experiment", "n_seeds",
        "median_ratio_oracle_over_random", "mean_ratio_oracle_over_random",
        "wins_oracle_over_random", "p_uncorrected_oracle_greater",
        "median_ratio_oracle_over_empirical", "mean_ratio_oracle_over_empirical",
        "wins_oracle_over_empirical", "p_uncorrected_empirical_greater",
        "paper_note",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Ablation summary
    if os.path.exists(CSV_ABLATION):
        print()
        print("=" * 80)
        print("S0 ablation on real-CKKS LR (B1)")
        print("=" * 80)
        abl_by_cfg: dict[str, list[float]] = defaultdict(list)
        with open(CSV_ABLATION) as fh:
            for r in csv.DictReader(fh):
                abl_by_cfg[r["config"]].append(float(r["max_error"]))
        full_vals = np.array(abl_by_cfg.get("FULL", []))
        print(
            f"{'cfg':>5s}  {'median':>12s}  {'mean':>12s}  "
            f"{'R(FULL/lesion)':>16s}  {'wins(F>L)':>10s}  {'p':>9s}  {'verdict':<14s}"
        )
        print("-" * 80)
        for cfg in ["FULL", "DIV", "-N", "-D", "-ND", "-S", "-MM", "-DS", "-NT"]:
            vals = np.array(abl_by_cfg.get(cfg, []))
            if len(vals) == 0:
                continue
            ratio = float(np.median(full_vals / np.where(vals > 1e-12, vals, 1e-12))) if cfg != "FULL" else 1.0
            wins = int(np.sum(full_vals > vals)) if cfg != "FULL" else 0
            try:
                p = float(wilcoxon(full_vals, vals, alternative="greater").pvalue) if cfg != "FULL" else 1.0
            except ValueError:
                p = 1.0
            if cfg == "FULL":
                verdict = "REFERENCE"
            elif ratio >= 1.10 and wins >= 7 and p < 0.05:
                verdict = "LOAD_BEARING"
            elif 0.95 <= ratio <= 1.05 and 3 <= wins <= 7:
                verdict = "INERT"
            else:
                verdict = "INCONCLUSIVE"
            print(
                f"{cfg:>5s}  {float(np.median(vals)):>12.4e}  "
                f"{float(np.mean(vals)):>12.4e}  "
                f"{ratio:>16.3f}  {wins:>3d}/10  {p:>9.3e}  {verdict:<14s}"
            )

    print()
    print(f"Wrote {len(summary_rows)} rows to {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
