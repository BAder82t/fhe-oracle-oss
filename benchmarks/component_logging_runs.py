# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-run the S0 FULL configuration with per-evaluation component logging.

Reuses the 3 mock circuits from `benchmarks/ablation_heuristics.py`
(circuit1_lr d=8, circuit2_poly d=6, circuit3_cheb d=10) and runs the
FULL config (weights=(1.0, 0.5, 0.3), seeds=("mm","ds","nt")) at B=500
across seeds 0..9. Each run logs divergence/noise/depth per evaluation
into `benchmarks/results/component_logs/{circuit}_seed{seed}.csv`.

A summary CSV (`component_logs_summary.csv`) records the final
``max_error`` per run so it can be cross-validated against the
corresponding FULL rows of `benchmarks/results/ablation_heuristics.csv`.

Wall-clock: ~15s on an M-series Mac (mocks only, no TenSEAL).
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle.diagnostics import ComponentLog, InstrumentedFitness  # noqa: E402

from ablation_heuristics import make_circuit1, make_circuit2, make_circuit3  # noqa: E402


def run_instrumented(
    circuit: dict,
    seed: int,
    n_trials: int,
    w_div: float = 1.0,
    w_noise: float = 0.5,
    w_depth: float = 0.3,
) -> tuple[ComponentLog, float]:
    """Run one FULL-config cell with component logging.

    Returns (log, max_error). The oracle is constructed the same way
    as `ablation_heuristics.run_one_cell` with config="FULL" so that
    the final `max_error` is directly comparable to the S0 baseline.
    """
    log = ComponentLog()
    fit = InstrumentedFitness(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        dim=circuit["d"],
        w_div=w_div,
        w_noise=w_noise,
        w_depth=w_depth,
        log=log,
    )
    oracle = FHEOracle(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        input_dim=circuit["d"],
        input_bounds=circuit["bounds"],
        fitness=fit,
        seed=seed,
        use_heuristic_seeds=True,
        heuristic_which=("mm", "ds", "nt"),
        heuristic_k=10,
    )
    result = oracle.run(n_trials=n_trials, threshold=0.0)
    return log, float(result.max_error)


def main(n_trials: int = 500, seeds: list[int] | None = None) -> int:
    if seeds is None:
        seeds = list(range(10))
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results", "component_logs")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(base_dir, "results", "component_logs_summary.csv")

    total = len(circuits) * len(seeds)
    print(f"C5 component-logging runs")
    print(f"  Circuits: {len(circuits)}  ({[c['name'] for c in circuits]})")
    print(f"  Seeds:    {len(seeds)}  ({seeds})")
    print(f"  Budget:   B={n_trials}")
    print(f"  Output:   {out_dir}")
    print("=" * 70)

    t_start = time.perf_counter()
    rows: list[dict] = []
    count = 0
    for circuit in circuits:
        for seed in seeds:
            count += 1
            t0 = time.perf_counter()
            log, max_error = run_instrumented(circuit, seed, n_trials)
            wall = time.perf_counter() - t0
            csv_path = os.path.join(
                out_dir, f"{circuit['name']}_seed{seed}.csv"
            )
            log.to_csv(csv_path)
            rows.append(
                {
                    "circuit": circuit["name"],
                    "seed": seed,
                    "n_evaluations": len(log.evaluations),
                    "max_error": max_error,
                    "wall_clock_s": wall,
                    "log_path": os.path.relpath(csv_path, base_dir),
                }
            )
            print(
                f"[{count:3d}/{total}] "
                f"{circuit['name']:<16s} seed={seed}  "
                f"evals={len(log.evaluations):4d}  "
                f"max_err={max_error:.4e}  t={wall:.2f}s"
            )

    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    n = 500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    raise SystemExit(main(n_trials=n))
