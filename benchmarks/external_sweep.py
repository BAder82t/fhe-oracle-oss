# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""External benchmarks sweep (B3).

Adds MNIST (d=64) alongside WDBC (d=30) to reviewer-ready external
circuits. Compares four strategies per circuit:

  oracle_only    : FHEOracle with rho=0.3, separable=True (B2 stack)
  random_only    : uniform random over the input box
  empirical_only : EmpiricalSearch on training data + jitter
  hybrid_union   : run_hybrid (oracle + empirical, union verdict)

10 seeds per cell, B=500 per leg.

Output: benchmarks/results/external_sweep.csv
Columns: config, circuit, d, seed, max_error, verdict, source,
         oracle_max_error, empirical_max_error, wall_clock_s
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle import FHEOracle
from fhe_oracle.empirical import EmpiricalSearch
from fhe_oracle.hybrid import run_hybrid, _default_divergence_fn
from wdbc_mock import build_wdbc_circuit
from mnist_mock import build_mnist_circuit


THRESHOLD = 0.01
JITTER = 0.1
SEEDS = list(range(10))
BUDGET = 500


def _bounds_for(d: int, lo: float = -3.0, hi: float = 3.0):
    return [(lo, hi)] * d


def _random_baseline_result(
    plain, fhe, bounds: list[tuple[float, float]], budget: int, seed: int
) -> tuple[float, str, float]:
    """Uniform random baseline — returns (max_error, verdict, wall_s)."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best = 0.0
    t0 = time.perf_counter()
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            p = plain(x)
            f = fhe(x)
        except Exception:
            continue
        err = abs(float(p) - float(f))
        if err > best:
            best = err
    verdict = "FAIL" if best >= THRESHOLD else "PASS"
    return best, verdict, time.perf_counter() - t0


def run_circuit_cells(
    circuit_name: str,
    plain,
    fhe,
    data: np.ndarray,
    d: int,
) -> list[dict]:
    bounds = _bounds_for(d)
    div_fn = _default_divergence_fn(plain, fhe)
    rows: list[dict] = []

    for seed in SEEDS:
        # --- oracle_only (B2 stack: separable=True, random_floor=0.3) ---
        oracle = FHEOracle(
            plaintext_fn=plain, fhe_fn=fhe,
            input_dim=d, input_bounds=bounds,
            seed=seed,
            separable=True,
            random_floor=0.3,
        )
        t0 = time.perf_counter()
        o_res = oracle.run(n_trials=BUDGET, threshold=THRESHOLD)
        t_oracle = time.perf_counter() - t0
        rows.append({
            "config": "oracle_only",
            "circuit": circuit_name,
            "d": d,
            "seed": seed,
            "max_error": o_res.max_error,
            "verdict": o_res.verdict,
            "source": "oracle",
            "oracle_max_error": o_res.max_error,
            "empirical_max_error": 0.0,
            "wall_clock_s": t_oracle,
        })

        # --- random_only ---
        r_err, r_verdict, t_rand = _random_baseline_result(
            plain, fhe, bounds, BUDGET, seed
        )
        rows.append({
            "config": "random_only",
            "circuit": circuit_name,
            "d": d,
            "seed": seed,
            "max_error": r_err,
            "verdict": r_verdict,
            "source": "random",
            "oracle_max_error": 0.0,
            "empirical_max_error": 0.0,
            "wall_clock_s": t_rand,
        })

        # --- empirical_only ---
        emp = EmpiricalSearch(
            divergence_fn=div_fn,
            data=data,
            threshold=THRESHOLD,
            budget=BUDGET,
            jitter_std=JITTER,
            seed=seed + 100,
        )
        t0 = time.perf_counter()
        e_res = emp.run()
        t_emp = time.perf_counter() - t0
        rows.append({
            "config": "empirical_only",
            "circuit": circuit_name,
            "d": d,
            "seed": seed,
            "max_error": e_res.max_error,
            "verdict": e_res.verdict,
            "source": "empirical",
            "oracle_max_error": 0.0,
            "empirical_max_error": e_res.max_error,
            "wall_clock_s": t_emp,
        })

        # --- hybrid_union ---
        t0 = time.perf_counter()
        h_res = run_hybrid(
            plaintext_fn=plain, fhe_fn=fhe,
            input_dim=d, input_bounds=bounds,
            threshold=THRESHOLD,
            oracle_budget=BUDGET, oracle_seed=seed,
            random_floor=0.3,
            data=data, empirical_budget=BUDGET,
            jitter_std=JITTER, empirical_seed=seed + 100,
        )
        t_hyb = time.perf_counter() - t0
        rows.append({
            "config": "hybrid_union",
            "circuit": circuit_name,
            "d": d,
            "seed": seed,
            "max_error": h_res.max_error,
            "verdict": h_res.union_verdict,
            "source": h_res.source,
            "oracle_max_error": h_res.oracle_result.max_error,
            "empirical_max_error": h_res.empirical_result.max_error,
            "wall_clock_s": t_hyb,
        })

    return rows


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "external_sweep.csv")

    print("External benchmarks sweep (B3)")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Budget    : {BUDGET} per leg")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Jitter    : {JITTER}")
    print("=" * 70)

    t_start = time.perf_counter()
    all_rows: list[dict] = []

    print("[1/2] WDBC (d=30)...")
    plain_w, fhe_w, data_w, _, _, d_w = build_wdbc_circuit(random_state=42)
    wdbc_rows = run_circuit_cells("wdbc_lr", plain_w, fhe_w, data_w, d_w)
    all_rows.extend(wdbc_rows)
    print(f"       {len(wdbc_rows)} cells")

    print("[2/2] MNIST load_digits (d=64)...")
    plain_m, fhe_m, data_m, _, _, d_m = build_mnist_circuit(random_state=42)
    mnist_rows = run_circuit_cells("mnist_d64", plain_m, fhe_m, data_m, d_m)
    all_rows.extend(mnist_rows)
    print(f"       {len(mnist_rows)} cells")

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    fieldnames = [
        "config", "circuit", "d", "seed",
        "max_error", "verdict", "source",
        "oracle_max_error", "empirical_max_error", "wall_clock_s",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
