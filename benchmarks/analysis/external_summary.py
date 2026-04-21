# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analyse external benchmarks sweep (B3).

Reads benchmarks/results/external_sweep.csv; emits
benchmarks/results/external_summary.csv.

Per (config, circuit):
  - median_max_error, mean_max_error
  - ratio: empirical / oracle (per-seed paired median)
  - ratio: oracle / random
  - wins: empirical vs oracle, oracle vs random
  - Wilcoxon one-sided + Holm correction across the pair comparisons
  - Hybrid source tracking

Side-by-side comparison to WDBC (paper §6.7, A3 result):
  empirical/oracle on WDBC = 2.29x (8/10 wins, p=0.005).
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


CSV_IN = os.path.join(
    os.path.dirname(__file__), "..", "results", "external_sweep.csv"
)
CSV_OUT = os.path.join(
    os.path.dirname(__file__), "..", "results", "external_summary.csv"
)


def _seed_errs(rows: list[dict], config: str, circuit: str) -> np.ndarray:
    cells = [
        r for r in rows
        if r["config"] == config and r["circuit"] == circuit
    ]
    by_seed = {int(r["seed"]): float(r["max_error"]) for r in cells}
    seeds = sorted(by_seed.keys())
    return np.array([by_seed[s] for s in seeds], dtype=np.float64)


def _paired_ratio(num: np.ndarray, den: np.ndarray) -> float:
    if num.size == 0 or den.size == 0:
        return float("nan")
    n = min(num.size, den.size)
    eps = 1e-12
    denom = np.where(den[:n] > eps, den[:n], eps)
    return float(np.median(num[:n] / denom))


def _wilcoxon_greater(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided paired Wilcoxon H1: a > b."""
    try:
        n = min(a.size, b.size)
        if n == 0 or np.all(a[:n] == b[:n]):
            return 1.0
        _stat, p = wilcoxon(a[:n], b[:n], alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def _holm(pvalues: list[float]) -> list[float]:
    """Holm step-down correction on a list of p-values. Returns adjusted."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main() -> int:
    if not os.path.exists(CSV_IN):
        print(f"Missing: {CSV_IN}. Run benchmarks/external_sweep.py first.")
        return 1

    with open(CSV_IN) as fh:
        rows = list(csv.DictReader(fh))

    circuits = sorted({r["circuit"] for r in rows})
    configs = ["oracle_only", "random_only", "empirical_only", "hybrid_union"]

    summary_rows: list[dict] = []

    print("=" * 110)
    print("External benchmarks — per-config medians and paired ratios")
    print("=" * 110)
    header = (
        f"{'circuit':<12s} {'config':<16s} "
        f"{'median_err':>13s} {'mean_err':>13s} "
        f"{'FAIL/10':>8s}"
    )
    print(header)
    print("-" * 110)

    per_circuit: dict[str, dict[str, np.ndarray]] = {}
    for circuit in circuits:
        per_circuit[circuit] = {}
        for cfg in configs:
            errs = _seed_errs(rows, cfg, circuit)
            per_circuit[circuit][cfg] = errs
            fails = sum(
                1 for r in rows
                if r["config"] == cfg and r["circuit"] == circuit
                and r["verdict"] == "FAIL"
            )
            print(
                f"{circuit:<12s} {cfg:<16s} "
                f"{float(np.median(errs)):>13.4e} "
                f"{float(np.mean(errs)):>13.4e} "
                f"{fails:>3d}/10"
            )
            summary_rows.append({
                "config": cfg,
                "circuit": circuit,
                "median_err": float(np.median(errs)),
                "mean_err": float(np.mean(errs)),
                "fail_rate": fails / 10,
                "ratio_vs_oracle": "-",
                "ratio_vs_random": "-",
                "wins_vs_oracle": "-",
                "wins_vs_random": "-",
                "p_greater_than_oracle": "-",
                "p_greater_than_random": "-",
            })
        print("-" * 110)

    # Paired comparisons: empirical vs oracle, oracle vs random, etc.
    print()
    print("Paired comparisons (Wilcoxon one-sided H1: a > b, Holm-adjusted)")
    print("=" * 110)
    print(
        f"{'circuit':<12s} {'comparison':<32s} "
        f"{'ratio':>10s} {'wins/10':>10s} "
        f"{'p_raw':>10s} {'p_holm':>10s}"
    )
    print("-" * 110)

    # Collect p-values per circuit to Holm-adjust together.
    for circuit in circuits:
        oracle = per_circuit[circuit]["oracle_only"]
        empirical = per_circuit[circuit]["empirical_only"]
        random_ = per_circuit[circuit]["random_only"]
        hybrid = per_circuit[circuit]["hybrid_union"]

        comparisons = [
            ("empirical > oracle",  empirical, oracle),
            ("oracle > random",     oracle,    random_),
            ("hybrid > oracle",     hybrid,    oracle),
            ("hybrid > empirical",  hybrid,    empirical),
        ]
        p_raws = [_wilcoxon_greater(a, b) for _, a, b in comparisons]
        p_adj = _holm(p_raws)

        for (label, a, b), p_raw, p_h in zip(comparisons, p_raws, p_adj):
            n = min(a.size, b.size)
            ratio = _paired_ratio(a, b)
            wins = int(np.sum(a[:n] > b[:n]))
            print(
                f"{circuit:<12s} {label:<32s} "
                f"{ratio:>10.3f} {wins:>4d}/{n:<5d} "
                f"{p_raw:>10.3e} {p_h:>10.3e}"
            )
            # Update summary for relevant rows
            if label == "empirical > oracle":
                for row in summary_rows:
                    if (
                        row["circuit"] == circuit
                        and row["config"] == "empirical_only"
                    ):
                        row["ratio_vs_oracle"] = f"{ratio:.3f}"
                        row["wins_vs_oracle"] = f"{wins}/{n}"
                        row["p_greater_than_oracle"] = f"{p_h:.3e}"
            if label == "oracle > random":
                for row in summary_rows:
                    if (
                        row["circuit"] == circuit
                        and row["config"] == "oracle_only"
                    ):
                        row["ratio_vs_random"] = f"{ratio:.3f}"
                        row["wins_vs_random"] = f"{wins}/{n}"
                        row["p_greater_than_random"] = f"{p_h:.3e}"
        print("-" * 110)

    # Hybrid source tracking
    print()
    print("Hybrid source tracking — which leg found the max_error?")
    print("-" * 60)
    for circuit in circuits:
        cells = [
            r for r in rows
            if r["config"] == "hybrid_union" and r["circuit"] == circuit
        ]
        oracle_ct = sum(1 for r in cells if r["source"] == "oracle")
        emp_ct = sum(1 for r in cells if r["source"] == "empirical")
        print(
            f"  {circuit:<12s}  oracle-source: {oracle_ct}/10  "
            f"empirical-source: {emp_ct}/10"
        )

    # Side-by-side with WDBC A3 result (for MNIST, if present).
    print()
    print("Comparison to WDBC (A3) baseline")
    print("=" * 70)
    print(f"  {'metric':<28s} {'WDBC (A3)':>15s} {'MNIST (B3)':>15s}")
    print("  " + "-" * 60)

    def _row(metric: str, wdbc_val: str, mnist_val: str) -> None:
        print(f"  {metric:<28s} {wdbc_val:>15s} {mnist_val:>15s}")

    if "mnist_d64" in circuits and "wdbc_lr" in circuits:
        w_oracle = per_circuit["wdbc_lr"]["oracle_only"]
        w_emp = per_circuit["wdbc_lr"]["empirical_only"]
        w_random = per_circuit["wdbc_lr"]["random_only"]
        m_oracle = per_circuit["mnist_d64"]["oracle_only"]
        m_emp = per_circuit["mnist_d64"]["empirical_only"]
        m_random = per_circuit["mnist_d64"]["random_only"]

        w_eo = _paired_ratio(w_emp, w_oracle)
        m_eo = _paired_ratio(m_emp, m_oracle)
        w_or = _paired_ratio(w_oracle, w_random)
        m_or = _paired_ratio(m_oracle, m_random)
        w_wins = int(np.sum(w_emp > w_oracle))
        m_wins = int(np.sum(m_emp > m_oracle))

        _row("empirical/oracle ratio", f"{w_eo:.3f}", f"{m_eo:.3f}")
        _row("empirical vs oracle wins", f"{w_wins}/10", f"{m_wins}/10")
        _row("oracle/random ratio", f"{w_or:.3f}", f"{m_or:.3f}")
        _row(
            "oracle median_err",
            f"{float(np.median(w_oracle)):.3e}",
            f"{float(np.median(m_oracle)):.3e}",
        )
        _row(
            "empirical median_err",
            f"{float(np.median(w_emp)):.3e}",
            f"{float(np.median(m_emp)):.3e}",
        )

    # Write CSV
    fieldnames = [
        "config", "circuit",
        "median_err", "mean_err", "fail_rate",
        "ratio_vs_oracle", "ratio_vs_random",
        "wins_vs_oracle", "wins_vs_random",
        "p_greater_than_oracle", "p_greater_than_random",
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
