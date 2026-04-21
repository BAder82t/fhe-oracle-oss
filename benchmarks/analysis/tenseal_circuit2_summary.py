# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Analyse Circuit 2 real-CKKS validation.

Reads benchmarks/results/tenseal_circuit2_validation.csv and emits
per-setting summary comparing to the mock Circuit 2 baseline from
paper Table 1 (1.27×, 8/10 wins on mock).

Output: benchmarks/results/tenseal_circuit2_summary.csv
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "tenseal_circuit2_validation.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "tenseal_circuit2_summary.csv"
)

MOCK_BASELINE = {
    "ratio_mean": 1.27,
    "wins": 8,
    "note": "Table 1 mock Circuit 2 (d=6, B=500)",
}


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing {CSV_IN}; run benchmarks/tenseal_circuit2_validation.py")
        return 1

    rows: list[dict] = []
    with open(CSV_IN) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    by_setting: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_setting[r["setting"]].append(r)

    summary_rows = []
    print("=" * 100)
    print("Circuit 2 real-CKKS validation (depth-4 polynomial, d=6, B=60)")
    print(f"Mock baseline (Table 1): {MOCK_BASELINE['ratio_mean']}x mean, "
          f"{MOCK_BASELINE['wins']}/10 wins")
    print("=" * 100)
    print(
        f"{'setting':<16s}  {'n':>3s}  {'δ̄ oracle':>12s}  {'δ̄ random':>12s}  "
        f"{'R median':>10s}  {'R mean':>10s}  {'wins':>6s}  {'p (1-sided)':>12s}"
    )
    print("-" * 100)

    for setting, srows in sorted(by_setting.items()):
        o_vals = np.array([float(r["oracle_max_error"]) for r in srows])
        r_vals = np.array([float(r["random_max_error"]) for r in srows])
        mask = r_vals > 1e-18
        ratios = np.where(mask, o_vals / np.where(mask, r_vals, 1.0), np.nan)
        valid = ratios[~np.isnan(ratios)]
        median_ratio = float(np.median(valid)) if valid.size else float("nan")
        mean_ratio = float(np.mean(valid)) if valid.size else float("nan")
        wins = int(np.sum(o_vals > r_vals))
        mean_oracle = float(np.mean(o_vals))
        mean_random = float(np.mean(r_vals))
        std_oracle = float(np.std(o_vals, ddof=1)) if len(o_vals) > 1 else 0.0
        std_random = float(np.std(r_vals, ddof=1)) if len(r_vals) > 1 else 0.0

        if HAVE_SCIPY:
            try:
                p = float(wilcoxon(o_vals, r_vals, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
        else:
            p = float("nan")

        print(
            f"{setting:<16s}  {len(srows):>3d}  "
            f"{mean_oracle:>12.4e}  {mean_random:>12.4e}  "
            f"{median_ratio:>10.3f}  {mean_ratio:>10.3f}  "
            f"{wins:>3d}/10  {p:>12.3e}"
        )

        summary_rows.append({
            "setting": setting,
            "n_seeds": len(srows),
            "mean_oracle_max_error": mean_oracle,
            "std_oracle_max_error": std_oracle,
            "mean_random_max_error": mean_random,
            "std_random_max_error": std_random,
            "median_ratio": median_ratio,
            "mean_ratio": mean_ratio,
            "wins_oracle_over_random": wins,
            "p_uncorrected_oracle_greater": p,
            "mock_baseline_ratio": MOCK_BASELINE["ratio_mean"],
            "mock_baseline_wins": MOCK_BASELINE["wins"],
        })

    print("-" * 100)

    fieldnames = [
        "setting", "n_seeds",
        "mean_oracle_max_error", "std_oracle_max_error",
        "mean_random_max_error", "std_random_max_error",
        "median_ratio", "mean_ratio",
        "wins_oracle_over_random", "p_uncorrected_oracle_greater",
        "mock_baseline_ratio", "mock_baseline_wins",
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
