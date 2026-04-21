# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2 — Summary statistics and ANOVA on the Taylor-vs-CKKS ablation.

Reads ``benchmarks/results/taylor_ckks_ablation.csv`` and produces
``benchmarks/results/taylor_ckks_summary.csv`` with one row per
``(arm, scale_bits)`` cell. Also runs a one-way ANOVA of ``oracle_max_error``
across ``scale_bits`` within each polynomial arm, reporting F, p, and
η² (variance explained). η² > 0.05 at Cheb-15 means CKKS parameters
carry measurable signal when polynomial approximation error is
reduced to near the CKKS noise floor.

The summary also reports, per cell:

- median ``oracle_max_error``, ``poly_error``, ``ckks_error``;
- oracle / random ratio and wins-out-of-N;
- **dominant component**: whether ``poly_error`` or ``ckks_error`` is
  larger at the oracle's worst input, averaged across seeds. This is
  the single most informative number for the paper — it answers
  "is the oracle finding polynomial bugs or FHE noise bugs?"
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy import stats


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "benchmarks", "results", "taylor_ckks_ablation.csv")
OUT = os.path.join(ROOT, "benchmarks", "results", "taylor_ckks_summary.csv")


def _group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["arm"], int(r["scale_bits"]))
        groups[key].append(r)
    return groups


def _median(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _summarise_cell(rows: list[dict]) -> dict:
    ora = np.array([float(r["oracle_max_error"]) for r in rows])
    rnd = np.array([float(r["random_max_error"]) for r in rows])
    poly = np.array([float(r["oracle_poly_error"]) for r in rows])
    ckks = np.array([float(r["oracle_ckks_error"]) for r in rows])
    wins = np.array([int(r["oracle_wins"]) for r in rows])
    ratios = np.where(rnd > 0, ora / rnd, np.inf)
    poly_dom = int(np.sum(poly > ckks))
    ckks_dom = int(np.sum(ckks >= poly))
    row0 = rows[0]
    return {
        "arm": row0["arm"],
        "scale_bits": int(row0["scale_bits"]),
        "poly_degree": int(row0["poly_degree"]),
        "poly_fit_error": float(row0["poly_fit_error"]),
        "N": int(row0["N"]),
        "chain": row0["chain"],
        "n_seeds": len(rows),
        "median_oracle_max_error": _median(ora),
        "median_random_max_error": _median(rnd),
        "median_oracle_over_random": _median(ratios),
        "oracle_wins": int(wins.sum()),
        "median_poly_error_at_worst": _median(poly),
        "median_ckks_error_at_worst": _median(ckks),
        "seeds_poly_dominant": poly_dom,
        "seeds_ckks_dominant": ckks_dom,
        "dominant_component": (
            "POLY" if poly_dom > ckks_dom else ("CKKS" if ckks_dom > poly_dom else "TIE")
        ),
    }


def _anova_scale_within_arm(rows: list[dict]) -> dict[str, dict]:
    """One-way ANOVA of oracle_max_error across scale_bits per arm."""
    arms: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)

    out: dict[str, dict] = {}
    for arm, arm_rows in arms.items():
        by_scale: dict[int, list[float]] = defaultdict(list)
        for r in arm_rows:
            by_scale[int(r["scale_bits"])].append(float(r["oracle_max_error"]))
        groups = list(by_scale.values())
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            out[arm] = {
                "F": float("nan"),
                "p": float("nan"),
                "eta_squared": float("nan"),
                "group_sizes": [len(g) for g in groups],
            }
            continue
        f, p = stats.f_oneway(*groups)
        # eta^2 = SS_between / SS_total
        flat = np.concatenate(groups)
        grand_mean = flat.mean()
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
        )
        ss_total = float(np.sum((flat - grand_mean) ** 2))
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        out[arm] = {
            "F": float(f),
            "p": float(p),
            "eta_squared": float(eta_sq),
            "group_sizes": [len(g) for g in groups],
        }
    return out


def main() -> int:
    if not os.path.exists(SRC):
        print(f"Missing source CSV: {SRC}")
        return 1
    with open(SRC) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"No rows in {SRC}")
        return 1

    groups = _group_rows(rows)
    summary = [_summarise_cell(rs) for rs in groups.values()]
    summary.sort(key=lambda r: (r["poly_degree"], r["scale_bits"]))

    print("Per-cell summary")
    print("=" * 100)
    header = (
        f"{'arm':<8s} {'scale':>6s}  {'N':>6s}  "
        f"{'ora_med':>10s}  {'rnd_med':>10s}  {'R':>6s}  "
        f"{'wins':>5s}  {'poly_err':>10s}  {'ckks_err':>10s}  "
        f"{'dominant':>9s}"
    )
    print(header)
    for r in summary:
        print(
            f"{r['arm']:<8s} 2^{r['scale_bits']:<4d} "
            f"{r['N']:>6d}  "
            f"{r['median_oracle_max_error']:>10.3e}  "
            f"{r['median_random_max_error']:>10.3e}  "
            f"{r['median_oracle_over_random']:>6.2f}  "
            f"{r['oracle_wins']:>2d}/{r['n_seeds']:<2d}  "
            f"{r['median_poly_error_at_worst']:>10.3e}  "
            f"{r['median_ckks_error_at_worst']:>10.3e}  "
            f"{r['dominant_component']:>9s}"
        )

    print("\nOne-way ANOVA: oracle_max_error ~ scale_bits, per polynomial arm")
    print("=" * 100)
    anovas = _anova_scale_within_arm(rows)
    for arm, res in anovas.items():
        print(
            f"  {arm:<8s}  F={res['F']:.3f}  p={res['p']:.3e}  "
            f"eta^2={res['eta_squared']:.4f}  groups={res['group_sizes']}"
        )

    fieldnames = list(summary[0].keys())
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary CSV: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
