# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate highdim_summary.csv from highdim_sweep.csv with d=256/512.

Run after benchmarks/highdim_d512_extension.py to refresh the
canonical summary used by the paper. Computes per-(config, circuit, d)
median + mean error, ratio_vs_random, and wins counts.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np


def main() -> int:
    here = os.path.abspath(os.path.dirname(__file__))
    sweep_path = os.path.join(here, "..", "results", "highdim_sweep.csv")
    summary_path = os.path.join(here, "..", "results", "highdim_summary.csv")

    if not os.path.exists(sweep_path):
        print(f"missing {sweep_path}")
        return 1

    rows: list[dict] = []
    with open(sweep_path, "r") as fh:
        for row in csv.DictReader(fh):
            row["d"] = int(row["d"])
            row["seed"] = int(row["seed"])
            row["max_error"] = float(row["max_error"])
            rows.append(row)

    # Group by (config, circuit, d). Also collect per-seed random baselines
    # for win counts and ratio computation.
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["config"], r["circuit"], r["d"])].append(r)

    # Random-baseline lookup keyed by (circuit, d, seed).
    random_by_seed = {
        (r["circuit"], r["d"], r["seed"]): r["max_error"]
        for r in rows if r["config"] == "uniform_random"
    }
    full_cma_by_seed = {
        (r["circuit"], r["d"], r["seed"]): r["max_error"]
        for r in rows if r["config"] == "full_cma"
    }

    out_rows: list[dict] = []
    for (config, circuit, d), bucket in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])
    ):
        errs = np.array([r["max_error"] for r in bucket])
        seeds = [r["seed"] for r in bucket]
        median_err = float(np.median(errs))
        mean_err = float(np.mean(errs))

        if config == "uniform_random":
            ratio_vs_random = 1.0
            wins_vs_random = 0
            ratio_vs_full = "-"
            wins_vs_full = "-"
        else:
            # Per-seed wins vs random and vs full_cma.
            wins_r = 0
            paired_ratios = []
            for r in bucket:
                key = (r["circuit"], r["d"], r["seed"])
                if key in random_by_seed:
                    if r["max_error"] > random_by_seed[key]:
                        wins_r += 1
            random_med = float(np.median([
                random_by_seed[(circuit, d, s)]
                for s in seeds
                if (circuit, d, s) in random_by_seed
            ])) if any((circuit, d, s) in random_by_seed for s in seeds) else 1.0
            ratio_vs_random = median_err / max(random_med, 1e-30)

            if config == "full_cma":
                ratio_vs_full = "1.000"
                wins_vs_full = "-"
            else:
                wins_f = 0
                for r in bucket:
                    key = (r["circuit"], r["d"], r["seed"])
                    if key in full_cma_by_seed:
                        if r["max_error"] > full_cma_by_seed[key]:
                            wins_f += 1
                full_cma_med = float(np.median([
                    full_cma_by_seed[(circuit, d, s)]
                    for s in seeds
                    if (circuit, d, s) in full_cma_by_seed
                ])) if any((circuit, d, s) in full_cma_by_seed for s in seeds) else 1.0
                ratio_vs_full = f"{(median_err / max(full_cma_med, 1e-30)):.3f}"
                n_paired = sum(
                    1 for s in seeds if (circuit, d, s) in full_cma_by_seed
                )
                wins_vs_full = f"{wins_f}/{n_paired}"
            wins_vs_random = wins_r

        out_rows.append({
            "config": config,
            "circuit": circuit,
            "d": d,
            "median_err": median_err,
            "mean_err": mean_err,
            "ratio_vs_random": ratio_vs_random,
            "wins_vs_random": wins_vs_random,
            "ratio_vs_full_cma": ratio_vs_full,
            "wins_vs_full_cma": wins_vs_full,
        })

    fieldnames = [
        "config", "circuit", "d", "median_err", "mean_err",
        "ratio_vs_random", "wins_vs_random",
        "ratio_vs_full_cma", "wins_vs_full_cma",
    ]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print(f"Wrote {len(out_rows)} rows to {summary_path}")

    # Headline scaling table for d in {8, 16, 32, 64, 128, 256, 512}
    print("\n--- Headline scaling table (LR) ---")
    print(f"{'d':>4} {'random':>12} {'full_cma':>12} {'ratio':>8} {'full_warm':>12} {'ratio':>8}")
    for d in sorted(set(r["d"] for r in out_rows)):
        rand_v = next((r["median_err"] for r in out_rows
                       if r["circuit"] == "lr_mock" and r["d"] == d
                       and r["config"] == "uniform_random"), None)
        fc = next((r for r in out_rows
                   if r["circuit"] == "lr_mock" and r["d"] == d
                   and r["config"] == "full_cma"), None)
        fw = next((r for r in out_rows
                   if r["circuit"] == "lr_mock" and r["d"] == d
                   and r["config"] == "full_warm"), None)
        if rand_v is None or fc is None or fw is None:
            continue
        print(
            f"{d:>4} {rand_v:>12.4e} {fc['median_err']:>12.4e} "
            f"{fc['ratio_vs_random']:>8.3f} "
            f"{fw['median_err']:>12.4e} {fw['ratio_vs_random']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
