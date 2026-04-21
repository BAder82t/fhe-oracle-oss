# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse union verdict sweep (A3).

Reads benchmarks/results/union_verdict_sweep.csv; emits
benchmarks/results/union_verdict_summary.csv.

For WDBC:
- Per config: median, mean, FAIL rate, max-error ratio vs
  wdbc_oracle baseline.
- Paired wins count: how often does empirical exceed oracle?
- Hybrid-union FAIL rate vs each leg.

For circuits 1-3 (regression check):
- Verify median max_error matches A1 baseline from hybrid_summary.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "union_verdict_sweep.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "union_verdict_summary.csv"
)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided Wilcoxon H1: a > b."""
    try:
        if np.all(a == b):
            return 1.0
        stat, p = wilcoxon(a, b, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing: {CSV_IN}. Run benchmarks/union_verdict_sweep.py first.")
        return 1

    rows: list[dict] = []
    with open(CSV_IN) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    # --- WDBC analysis ---
    wdbc_rows = [r for r in rows if r["circuit"] == "wdbc_lr"]
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in wdbc_rows:
        by_cfg[r["config"]].append(r)

    # Build seed-indexed arrays
    def seed_arr(cfg: str, field: str) -> np.ndarray:
        data = {int(r["seed"]): float(r[field]) for r in by_cfg.get(cfg, [])}
        seeds = sorted(data.keys())
        return np.array([data[s] for s in seeds]), seeds

    oracle_errs, oracle_seeds = seed_arr("wdbc_oracle", "max_error")
    emp_errs, emp_seeds = seed_arr("wdbc_empirical", "max_error")
    hyb_errs, hyb_seeds = seed_arr("wdbc_hybrid", "max_error")
    hyb_eq_errs, _ = seed_arr("wdbc_hybrid_eq", "max_error")

    summary_rows = []

    print("=" * 100)
    print("WDBC — median / mean / FAIL rate / ratio vs oracle / wins")
    print("=" * 100)
    print(
        f"{'config':<18s}  {'median_err':>12s}  {'mean_err':>12s}  "
        f"{'FAIL/10':>8s}  {'R vs oracle':>12s}  {'wins/oracle':>12s}"
    )
    print("-" * 100)

    configs = ["wdbc_oracle", "wdbc_empirical", "wdbc_hybrid", "wdbc_hybrid_eq"]
    for cfg in configs:
        errs, seeds = seed_arr(cfg, "max_error")
        fails = sum(1 for r in by_cfg[cfg] if r["verdict"] == "FAIL")
        if cfg == "wdbc_oracle":
            ratio_str = "-"
            wins_str = "-"
        else:
            denom = np.where(oracle_errs > 1e-12, oracle_errs, 1e-12)
            ratios = errs / denom
            ratio = float(np.median(ratios))
            wins = int(np.sum(errs > oracle_errs))
            p = paired_wilcoxon(errs, oracle_errs)
            ratio_str = f"{ratio:.3f}"
            wins_str = f"{wins}/10  (p={p:.3e})"
        print(
            f"{cfg:<18s}  {float(np.median(errs)):>12.4e}  "
            f"{float(np.mean(errs)):>12.4e}  "
            f"{fails:>3d}/10  "
            f"{ratio_str:>12s}  {wins_str:>12s}"
        )
        summary_rows.append({
            "config": cfg,
            "circuit": "wdbc_lr",
            "median_err": float(np.median(errs)),
            "mean_err": float(np.mean(errs)),
            "fail_rate": fails / 10,
            "ratio_vs_oracle": ratio_str,
            "wins_vs_oracle": wins_str,
        })
    print("-" * 100)

    # Source analysis for hybrid configs
    print()
    print("Hybrid source tracking — which leg found the max_error?")
    print("-" * 60)
    for cfg in ["wdbc_hybrid", "wdbc_hybrid_eq"]:
        source_counts = defaultdict(int)
        for r in by_cfg[cfg]:
            source_counts[r["source"]] += 1
        print(
            f"{cfg:<18s}  oracle-source: {source_counts['oracle']}/10  "
            f"empirical-source: {source_counts['empirical']}/10"
        )

    # Regression check on circuits 1-3
    print()
    print("=" * 80)
    print("Regression check — oracle_only_a1 on prior mock circuits")
    print("Expected match (from A1 hybrid_summary.csv at ρ=0.3):")
    print("  circuit1_lr    ~5.02e-01 median")
    print("  circuit2_poly  ~1.96e-02 median")
    print("  circuit3_cheb  ~1.00e-01 median")
    print("-" * 80)
    reg_rows = [r for r in rows if r["config"] == "oracle_only_a1"]
    by_circuit = defaultdict(list)
    for r in reg_rows:
        by_circuit[r["circuit"]].append(float(r["max_error"]))
    for circ in sorted(by_circuit):
        errs_arr = np.array(by_circuit[circ])
        print(
            f"  {circ:<16s}  median={float(np.median(errs_arr)):.4e}  "
            f"mean={float(np.mean(errs_arr)):.4e}  n={len(errs_arr)}"
        )
        summary_rows.append({
            "config": "oracle_only_a1",
            "circuit": circ,
            "median_err": float(np.median(errs_arr)),
            "mean_err": float(np.mean(errs_arr)),
            "fail_rate": 0.0,
            "ratio_vs_oracle": "-",
            "wins_vs_oracle": "-",
        })

    # Emit CSV
    fieldnames = [
        "config", "circuit", "median_err", "mean_err", "fail_rate",
        "ratio_vs_oracle", "wins_vs_oracle",
    ]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print()
    print(f"Wrote {len(summary_rows)} rows to {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
