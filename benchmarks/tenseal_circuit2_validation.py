# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Real-CKKS validation for Circuit 2 (depth-4 polynomial, d=6).

Ports the last mock-only internal circuit (Limitation 1 of the paper)
to TenSEAL. Runs the standard matched + asymmetric protocol at B=60
for 10 seeds. Plus a bonus warm-start run at ρ=0.3.

Circuit 2 is special: plaintext and FHE compute the EXACT same
polynomial. δ therefore measures pure CKKS noise — no
Taylor/Chebyshev approximation confound.

Outputs: benchmarks/results/tenseal_circuit2_validation.csv
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
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL, TenSEALContext
from tenseal_circuits import build_tenseal_circuit2


B_TENSEAL = 60
SEEDS = list(range(10))

ADVERSARIAL_BOUNDS = [(-2.0, 2.0)] * 6
OPERATIONAL_BOUNDS = [(-0.5, 0.5)] * 6


def run_random_baseline(plaintext_fn, fhe_fn, bounds, budget, seed):
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best_err = -np.inf
    for _ in range(budget):
        x = rng.uniform(lows, highs).tolist()
        p = plaintext_fn(x)
        f = fhe_fn(x)
        err = abs(float(p) - float(f))
        if err > best_err:
            best_err = err
    return float(best_err)


def run_oracle(circuit, seed, budget, random_floor=0.0, threshold=0.0):
    oracle = FHEOracle(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        input_dim=circuit["d"],
        input_bounds=circuit["bounds"],
        seed=seed,
        random_floor=random_floor,
    )
    return oracle.run(n_trials=budget, threshold=threshold)


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not available. Skipping Circuit 2 real-CKKS validation.")
        return 0

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tenseal_circuit2_validation.csv")

    print("TenSEAL real-CKKS Circuit 2 validation (Limitation-1 port)")
    print(f"  Budget B    = {B_TENSEAL}")
    print(f"  Seeds       = {SEEDS}")
    print(f"  d           = 6")
    print(f"  Adversarial = {ADVERSARIAL_BOUNDS[0]}")
    print(f"  Operational = {OPERATIONAL_BOUNDS[0]}")
    print("=" * 80)

    ctx = TenSEALContext()
    circuit = build_tenseal_circuit2(ctx)

    rows = []
    t_start = time.perf_counter()

    # --- Setting 1: matched (both oracle and random on [-2,2]^6) -----
    print("\n[1/3] matched  (both oracle and random on [-2,2]^6)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(circuit, seed, B_TENSEAL)
        rnd = run_random_baseline(
            circuit["plain"], circuit["fhe"],
            ADVERSARIAL_BOUNDS, B_TENSEAL, seed,
        )
        ratio = ores.max_error / rnd if rnd > 0 else float("inf")
        wins = int(ores.max_error > rnd)
        wall = time.perf_counter() - t0
        print(
            f"   seed={seed} oracle={ores.max_error:.4e} "
            f"rand={rnd:.4e} R={ratio:.2f}x  t={wall:.1f}s"
        )
        rows.append({
            "setting": "matched", "seed": seed,
            "oracle_max_error": ores.max_error,
            "random_max_error": rnd,
            "ratio": ratio, "oracle_wins": wins,
            "wall_clock_s": wall,
        })

    # --- Setting 2: asymmetric (random on [-0.5,0.5]^6) ---------------
    print("\n[2/3] asymmetric  (random on operational [-0.5,0.5]^6)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(circuit, seed, B_TENSEAL)
        rnd = run_random_baseline(
            circuit["plain"], circuit["fhe"],
            OPERATIONAL_BOUNDS, B_TENSEAL, seed,
        )
        ratio = ores.max_error / rnd if rnd > 0 else float("inf")
        wins = int(ores.max_error > rnd)
        wall = time.perf_counter() - t0
        print(
            f"   seed={seed} oracle={ores.max_error:.4e} "
            f"rand={rnd:.4e} R={ratio:.2f}x  t={wall:.1f}s"
        )
        rows.append({
            "setting": "asymmetric", "seed": seed,
            "oracle_max_error": ores.max_error,
            "random_max_error": rnd,
            "ratio": ratio, "oracle_wins": wins,
            "wall_clock_s": wall,
        })

    # --- Setting 3: A1 warm-start ρ=0.3 on matched domain -------------
    print("\n[3/3] matched_warm  (oracle with random_floor ρ=0.3)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(circuit, seed, B_TENSEAL, random_floor=0.3)
        rnd = run_random_baseline(
            circuit["plain"], circuit["fhe"],
            ADVERSARIAL_BOUNDS, B_TENSEAL, seed,
        )
        ratio = ores.max_error / rnd if rnd > 0 else float("inf")
        wins = int(ores.max_error > rnd)
        wall = time.perf_counter() - t0
        print(
            f"   seed={seed} oracle={ores.max_error:.4e} "
            f"rand={rnd:.4e} R={ratio:.2f}x  t={wall:.1f}s"
        )
        rows.append({
            "setting": "matched_warm", "seed": seed,
            "oracle_max_error": ores.max_error,
            "random_max_error": rnd,
            "ratio": ratio, "oracle_wins": wins,
            "wall_clock_s": wall,
        })

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    fieldnames = [
        "setting", "seed",
        "oracle_max_error", "random_max_error",
        "ratio", "oracle_wins", "wall_clock_s",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
