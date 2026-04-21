# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""TenSEAL real-CKKS validation of A-spine results (B1).

Six experiments × 10 seeds × B=60:
  1. lr_matched      — LR d=8 oracle vs random
     (paper: 2.28× ratio, 7/10 wins)
  2. cheb_matched    — Chebyshev d=10 oracle vs random
     (paper: 0.68× ratio, 0/10 wins)
  3. cheb_warm       — Chebyshev d=10 + A1 ρ=0.3 warm-start
     (mock: 8/10 wins → real CKKS: ?)
  4. wdbc_matched    — WDBC d=30 oracle uniform-box vs random
     (paper: 1.14× median, 5/10 wins)
  5. wdbc_asymmetric — WDBC d=30 oracle vs empirical training
     distribution (paper: 0.25× ratio, 1/10 wins)
  6. wdbc_hybrid     — run_hybrid union verdict on WDBC

Outputs: benchmarks/results/tenseal_validation.csv
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
from fhe_oracle.empirical import EmpiricalSearch
from fhe_oracle.hybrid import _default_divergence_fn, run_hybrid
from tenseal_circuits import (
    build_tenseal_chebyshev_d10,
    build_tenseal_lr_d8,
    build_tenseal_wdbc,
)


B_TENSEAL = 60
SEEDS = list(range(10))


def run_random_baseline(plaintext_fn, fhe_fn, bounds, budget, seed):
    """Uniform-random sampling baseline at matched budget."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best_err = -np.inf
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        p = plaintext_fn(x.tolist())
        f = fhe_fn(x.tolist())
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
        f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
        n = min(p_arr.size, f_arr.size)
        err = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
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


def run_empirical_on_wdbc(circuit, seed, budget, jitter=0.1, threshold=0.0):
    div_fn = _default_divergence_fn(circuit["plain"], circuit["fhe"])
    emp = EmpiricalSearch(
        divergence_fn=div_fn,
        data=circuit["data"],
        threshold=threshold,
        budget=budget,
        jitter_std=jitter,
        seed=seed + 100,
    )
    return emp.run()


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not available. Skipping B1 validation.")
        return 0

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tenseal_validation.csv")

    print("TenSEAL real-CKKS validation (B1)")
    print(f"  Budget B = {B_TENSEAL}")
    print(f"  Seeds    = {SEEDS}")
    print("=" * 80)
    rows = []
    t_start = time.perf_counter()

    # Build contexts once per circuit — avoids noise carry-over.
    ctx_lr = TenSEALContext()
    lr = build_tenseal_lr_d8(ctx_lr)
    ctx_ch = TenSEALContext()
    ch = build_tenseal_chebyshev_d10(ctx_ch)
    ctx_wdbc = TenSEALContext()
    wdbc = build_tenseal_wdbc(ctx_wdbc)

    # --- Experiment 1: LR matched ---
    print(f"\n[1/6] LR d=8 matched  (paper: 2.28× mean, 7/10 wins)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(lr, seed, B_TENSEAL)
        rnd_err = run_random_baseline(lr["plain"], lr["fhe"], lr["bounds"], B_TENSEAL, seed)
        ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
        print(f"   seed={seed} oracle={ores.max_error:.4e} rand={rnd_err:.4e} R={ratio:.2f}x  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "lr_matched", "circuit": "lr_d8_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": rnd_err,
            "ratio_oracle_over_random": ratio, "oracle_wins_random": int(ores.max_error > rnd_err),
            "empirical_max_error": 0.0, "wall_clock_s": time.perf_counter() - t0,
        })

    # --- Experiment 2: Chebyshev matched (paper 0.68x) ---
    print(f"\n[2/6] Chebyshev d=10 matched (paper: 0.68×, 0/10 wins)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(ch, seed, B_TENSEAL)
        rnd_err = run_random_baseline(ch["plain"], ch["fhe"], ch["bounds"], B_TENSEAL, seed)
        ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
        print(f"   seed={seed} oracle={ores.max_error:.4e} rand={rnd_err:.4e} R={ratio:.2f}x  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "cheb_matched", "circuit": "cheb_d10_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": rnd_err,
            "ratio_oracle_over_random": ratio, "oracle_wins_random": int(ores.max_error > rnd_err),
            "empirical_max_error": 0.0, "wall_clock_s": time.perf_counter() - t0,
        })

    # --- Experiment 3: Chebyshev + A1 warm-start ---
    print(f"\n[3/6] Chebyshev d=10 + A1 warm-start ρ=0.3 (mock: 8/10 wins)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(ch, seed, B_TENSEAL, random_floor=0.3)
        rnd_err = run_random_baseline(ch["plain"], ch["fhe"], ch["bounds"], B_TENSEAL, seed)
        ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
        print(f"   seed={seed} oracle={ores.max_error:.4e} rand={rnd_err:.4e} R={ratio:.2f}x  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "cheb_warm", "circuit": "cheb_d10_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": rnd_err,
            "ratio_oracle_over_random": ratio, "oracle_wins_random": int(ores.max_error > rnd_err),
            "empirical_max_error": 0.0, "wall_clock_s": time.perf_counter() - t0,
        })

    # --- Experiment 4: WDBC matched ---
    print(f"\n[4/6] WDBC d=30 matched (paper: 1.14× median, 5/10 wins)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(wdbc, seed, B_TENSEAL)
        rnd_err = run_random_baseline(wdbc["plain"], wdbc["fhe"], wdbc["bounds"], B_TENSEAL, seed)
        ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
        print(f"   seed={seed} oracle={ores.max_error:.4e} rand={rnd_err:.4e} R={ratio:.2f}x  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "wdbc_matched", "circuit": "wdbc_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": rnd_err,
            "ratio_oracle_over_random": ratio, "oracle_wins_random": int(ores.max_error > rnd_err),
            "empirical_max_error": 0.0, "wall_clock_s": time.perf_counter() - t0,
        })

    # --- Experiment 5: WDBC asymmetric ---
    print(f"\n[5/6] WDBC d=30 asymmetric (paper: 0.25×, 1/10 wins)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(wdbc, seed, B_TENSEAL)
        eres = run_empirical_on_wdbc(wdbc, seed, B_TENSEAL)
        ratio = ores.max_error / eres.max_error if eres.max_error > 0 else float("inf")
        print(f"   seed={seed} oracle={ores.max_error:.4e} emp={eres.max_error:.4e} R={ratio:.3f}x  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "wdbc_asymmetric", "circuit": "wdbc_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": 0.0,
            "ratio_oracle_over_random": 0.0, "oracle_wins_random": int(ores.max_error > eres.max_error),
            "empirical_max_error": eres.max_error, "wall_clock_s": time.perf_counter() - t0,
        })

    # --- Experiment 6: WDBC hybrid union verdict ---
    print(f"\n[6/6] WDBC d=30 hybrid union (new)")
    for seed in SEEDS:
        t0 = time.perf_counter()
        hres = run_hybrid(
            plaintext_fn=wdbc["plain"], fhe_fn=wdbc["fhe"],
            input_dim=wdbc["d"], input_bounds=wdbc["bounds"],
            threshold=0.0,
            oracle_budget=B_TENSEAL, oracle_seed=seed, random_floor=0.3,
            data=wdbc["data"], empirical_budget=B_TENSEAL,
            jitter_std=0.1, empirical_seed=seed + 100,
        )
        print(f"   seed={seed} oracle={hres.oracle_result.max_error:.4e} emp={hres.empirical_result.max_error:.4e} source={hres.source}  t={time.perf_counter()-t0:.1f}s")
        rows.append({
            "experiment": "wdbc_hybrid", "circuit": "wdbc_tenseal", "seed": seed,
            "oracle_max_error": hres.oracle_result.max_error,
            "random_max_error": 0.0,
            "ratio_oracle_over_random": 0.0,
            "oracle_wins_random": int(hres.source == "oracle"),
            "empirical_max_error": hres.empirical_result.max_error,
            "wall_clock_s": time.perf_counter() - t0,
        })

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    fieldnames = [
        "experiment", "circuit", "seed",
        "oracle_max_error", "random_max_error",
        "ratio_oracle_over_random", "oracle_wins_random",
        "empirical_max_error", "wall_clock_s",
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
