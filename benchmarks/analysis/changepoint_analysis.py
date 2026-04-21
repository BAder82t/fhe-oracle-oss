# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C5 change-point analysis on per-evaluation shaping-component trajectories.

Reads the per-(circuit, seed) evaluation logs produced by
``benchmarks/component_logging_runs.py`` and tests whether any
shaping component (``noise_term``, ``depth_term``) carries
phase-dependent signal that the S0 run-averaged analysis could have
smoothed away.

Three analyses are run per trajectory:

1. Rolling Pearson correlation (window W=50) between the shaping
   component and ``divergence``. Persistently high correlation across
   the whole run means the shaping component is collinear with
   divergence and contributes no independent signal — exactly what
   S0 predicts.
2. Pettitt change-point test on each component trajectory. O(n²) but
   fine for n=500.
3. Cross-seed consistency: if ≥ 7/10 seeds flag a change-point at
   p < 0.05 AND the IQR of change-point indices is < 50 evaluations
   AND the minimum rolling correlation dips below 0.5, the trajectory
   is declared PHASE_SIGNAL. Otherwise NO_SIGNAL.

Output: ``benchmarks/results/changepoint_summary.csv``
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Iterable

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT, "benchmarks", "results", "component_logs")
SUMMARY_CSV = os.path.join(
    ROOT, "benchmarks", "results", "changepoint_summary.csv"
)

CIRCUITS = ["circuit1_lr", "circuit2_poly", "circuit3_cheb"]
COMPONENTS = ["noise_term", "depth_term"]
SEEDS = list(range(10))
WINDOW = 50
P_THRESHOLD = 0.05
MIN_SEEDS_FOR_CP = 7
IQR_THRESHOLD = 50
MIN_CORR_THRESHOLD = 0.5


def load_log(circuit: str, seed: int) -> dict[str, np.ndarray]:
    path = os.path.join(LOG_DIR, f"{circuit}_seed{seed}.csv")
    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return {
        key: np.array([float(r[key]) for r in rows])
        for key in rows[0].keys()
    }


def rolling_correlation(
    x: np.ndarray, y: np.ndarray, window: int = WINDOW
) -> np.ndarray:
    """Rolling Pearson r between x and y."""
    n = len(x)
    if n < window:
        return np.array([np.nan])
    r = np.full(n - window + 1, np.nan, dtype=np.float64)
    for i in range(n - window + 1):
        xi = x[i : i + window]
        yi = y[i : i + window]
        if np.std(xi) < 1e-12 or np.std(yi) < 1e-12:
            r[i] = 0.0
        else:
            r[i] = np.corrcoef(xi, yi)[0, 1]
    return r


def pettitt_test(x: np.ndarray) -> tuple[int, float]:
    """Pettitt non-parametric change-point test.

    Returns ``(cp_index, p_value)``. H0: no change point. O(n log n)
    implementation via rank-based Mann-Whitney statistic.
    """
    n = len(x)
    if n < 2:
        return 0, 1.0
    # Rank-based U_t = 2 * sum_{i<=t} rank(x_i) - t*(n+1)
    ranks = _rankdata(x)
    cumranks = np.cumsum(ranks)
    t_idx = np.arange(1, n + 1)
    U = 2.0 * cumranks - t_idx * (n + 1)
    K = float(np.max(np.abs(U)))
    cp = int(np.argmax(np.abs(U)))
    # Pettitt (1979) approximation
    p = 2.0 * np.exp(-6.0 * K * K / (n ** 3 + n ** 2))
    return cp, float(min(1.0, p))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank tie-breaking (scipy.stats.rankdata 'average')."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def analyse_cell(
    circuit: str, seed: int
) -> dict[str, dict[str, float]]:
    """Return per-component {'cp_index','p','mean_roll_r','min_roll_r',...}"""
    log = load_log(circuit, seed)
    div = log["divergence"]
    out: dict[str, dict[str, float]] = {}
    for comp in COMPONENTS:
        series = log[comp]
        cp, p = pettitt_test(series)
        roll_r = rolling_correlation(series, div)
        valid = roll_r[~np.isnan(roll_r)]
        if valid.size == 0:
            mean_r = np.nan
            min_r = np.nan
        else:
            mean_r = float(np.mean(valid))
            min_r = float(np.min(valid))
        if cp > 0 and cp < series.size - 1:
            mean_before = float(np.mean(series[: cp + 1]))
            mean_after = float(np.mean(series[cp + 1 :]))
            ratio = (
                mean_after / mean_before
                if abs(mean_before) > 1e-12
                else float("inf")
            )
        else:
            mean_before = float(np.mean(series))
            mean_after = float(np.mean(series))
            ratio = 1.0
        out[comp] = {
            "cp_index": cp,
            "p_value": p,
            "mean_roll_r_with_div": mean_r,
            "min_roll_r_with_div": min_r,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "ratio_after_over_before": ratio,
        }
    return out


def iqr(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    q75, q25 = np.percentile(arr, [75, 25])
    return float(q75 - q25)


def cross_seed_verdict(
    per_seed: list[dict[str, dict]], component: str
) -> dict[str, float | str | int]:
    seeds_with_cp = [
        s for s in per_seed if s[component]["p_value"] < P_THRESHOLD
    ]
    cp_indices = [s[component]["cp_index"] for s in seeds_with_cp]
    all_ps = np.array([s[component]["p_value"] for s in per_seed])
    all_mean_rs = np.array(
        [s[component]["mean_roll_r_with_div"] for s in per_seed]
    )
    all_min_rs = np.array(
        [s[component]["min_roll_r_with_div"] for s in per_seed]
    )
    median_cp = float(np.median(cp_indices)) if cp_indices else float("nan")
    iqr_cp = iqr(cp_indices) if len(cp_indices) >= 2 else float("nan")
    median_p = float(np.median(all_ps))
    mean_roll_r = float(np.nanmean(all_mean_rs))
    min_roll_r = float(np.nanmin(all_min_rs))
    verdict = "NO_SIGNAL"
    if (
        len(seeds_with_cp) >= MIN_SEEDS_FOR_CP
        and iqr_cp == iqr_cp  # not NaN
        and iqr_cp < IQR_THRESHOLD
        and min_roll_r < MIN_CORR_THRESHOLD
    ):
        verdict = "PHASE_SIGNAL"
    return {
        "n_seeds_with_cp": len(seeds_with_cp),
        "median_cp_index": median_cp,
        "iqr_cp_index": iqr_cp,
        "median_p": median_p,
        "mean_rolling_corr_with_div": mean_roll_r,
        "min_rolling_corr_with_div": min_roll_r,
        "verdict": verdict,
    }


def main() -> int:
    rows: list[dict] = []
    print(f"C5 change-point analysis")
    print(f"  Logs directory: {LOG_DIR}")
    print(f"  Circuits: {CIRCUITS}")
    print(f"  Components: {COMPONENTS}")
    print(f"  Seeds: {SEEDS}  (window={WINDOW})")
    print("=" * 80)

    for circuit in CIRCUITS:
        per_seed = [analyse_cell(circuit, s) for s in SEEDS]
        for comp in COMPONENTS:
            agg = cross_seed_verdict(per_seed, comp)
            row = {"circuit": circuit, "component": comp, **agg}
            rows.append(row)
            print(
                f"{circuit:<16s} {comp:<12s}  "
                f"cp_seeds={agg['n_seeds_with_cp']}/10  "
                f"med_cp={agg['median_cp_index']:.0f}  "
                f"iqr_cp={agg['iqr_cp_index']:.1f}  "
                f"med_p={agg['median_p']:.2e}  "
                f"mean_r={agg['mean_rolling_corr_with_div']:.3f}  "
                f"min_r={agg['min_rolling_corr_with_div']:.3f}  "
                f"=> {agg['verdict']}"
            )

    with open(SUMMARY_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"Summary: {SUMMARY_CSV}")

    verdicts = {r["verdict"] for r in rows}
    if "PHASE_SIGNAL" in verdicts:
        print("RESULT: at least one (circuit, component) shows PHASE_SIGNAL")
        print("  -> C5 adaptive-weights implementation gets a second life.")
    else:
        print("RESULT: NO_SIGNAL on all (circuit, component) pairs")
        print("  -> C5 adaptive-weights implementation permanently killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
