# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Extension of B2 high-dimensional scaling sweep to d in {256, 512}.

Reuses the circuit builders, oracle runner, and random baseline from
``highdim_sweep.py``. Appends new rows to
``benchmarks/results/highdim_sweep.csv`` so the existing 5-point curve
becomes a 7-point curve.

Budget rule extends the existing pattern (B=500 at d<=32, B=1000 at
d=64, B=2000 at d=128): B=4000 at d=256, B=8000 at d=512. This keeps
the per-eval/dim ratio in the ~15 range used by the existing data and
makes the new cells directly comparable with the existing 5-point
curve.

Seeds 1..10 (skip seed=0 per pycma quirk noted in the proposal brief).

Configurations: full_cma, full_warm only (the two configurations whose
ratio appears in the headline scaling table). sep_* variants are
omitted since they have already been shown to be dominated by full
covariance at d=128.

Output:
  - benchmarks/results/highdim_sweep_d512_extension.csv  (new rows
    only, for inspection)
  - rows are also appended to benchmarks/results/highdim_sweep.csv
    (the canonical sweep CSV) so the existing summary script picks
    them up automatically.
"""

from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from highdim_sweep import (
    CIRCUIT_BUILDERS,
    run_oracle,
    run_random_baseline,
)


NEW_DIMENSIONS = [256, 512]
NEW_SEEDS = list(range(1, 6))   # 5 seeds at higher d to manage wall-clock
NEW_CONFIGS = {
    "full_cma":  {"separable": False, "random_floor": 0.0},
    "full_warm": {"separable": False, "random_floor": 0.3},
}


def budget_for_dim_extended(d: int) -> int:
    # B for d=256 is half-step from existing B=2000@d=128; B for d=512
    # is capped at 4000 because full-covariance CMA-ES does O(d^3) eigen
    # decomposition per generation, which dominates wall-clock at d=512.
    # Empirically B=4000 keeps a single seed under ~30s; B=8000 was
    # >5min/seed in prior runs.
    if d == 256:
        return 3000
    if d == 512:
        return 4000
    raise ValueError(f"unexpected d={d}")


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    canonical_path = os.path.join(out_dir, "highdim_sweep.csv")
    extension_path = os.path.join(out_dir, "highdim_sweep_d512_extension.csv")

    print("High-dim scaling extension (B2): d in", NEW_DIMENSIONS)
    print(f"  Configs : {list(NEW_CONFIGS.keys())} + uniform_random baseline")
    print(f"  Circuits: {list(CIRCUIT_BUILDERS.keys())}")
    print(f"  Seeds   : {NEW_SEEDS}")
    print(f"  Append  : {canonical_path}")
    print(f"  New CSV : {extension_path}")
    print("=" * 75)

    rows: list[dict] = []
    t_start = time.perf_counter()

    for circ_name, builder in CIRCUIT_BUILDERS.items():
        for d in NEW_DIMENSIONS:
            budget = budget_for_dim_extended(d)
            print(f"  Building {circ_name} at d={d}, B={budget} ...")
            circuit = builder(d)

            for seed in NEW_SEEDS:
                err, wall, n_evals = run_random_baseline(circuit, budget, seed)
                rows.append({
                    "config": "uniform_random",
                    "circuit": circ_name,
                    "d": d,
                    "budget": budget,
                    "seed": seed,
                    "max_error": err,
                    "wall_clock_s": wall,
                    "n_trials": n_evals,
                })
                print(
                    f"    [random]   d={d} seed={seed} "
                    f"err={err:.4e} t={wall:.2f}s"
                )

            for cfg_name, cfg in NEW_CONFIGS.items():
                for seed in NEW_SEEDS:
                    err, wall, n_trials = run_oracle(
                        cfg_name, cfg, circuit, budget, seed
                    )
                    rows.append({
                        "config": cfg_name,
                        "circuit": circ_name,
                        "d": d,
                        "budget": budget,
                        "seed": seed,
                        "max_error": err,
                        "wall_clock_s": wall,
                        "n_trials": n_trials,
                    })
                    print(
                        f"    [{cfg_name}] d={d} seed={seed} "
                        f"err={err:.4e} t={wall:.2f}s"
                    )

    elapsed = time.perf_counter() - t_start
    print("=" * 75)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    fieldnames = [
        "config", "circuit", "d", "budget", "seed",
        "max_error", "wall_clock_s", "n_trials",
    ]

    # Standalone CSV (new rows only).
    with open(extension_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {extension_path}")

    # Append to canonical sweep CSV (preserving header).
    write_header = not os.path.exists(canonical_path)
    with open(canonical_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Appended {len(rows)} rows to {canonical_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
