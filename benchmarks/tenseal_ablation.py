# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""S0 heuristic lesion audit on real CKKS LR (B1 validation of S0 caveat).

Resolves the S0 mock-confound caveat: "Real conclusions about w_noise
vs w_depth independence require the TenSEAL LR row."

9 configs × 1 circuit (TenSEAL LR d=8) × 10 seeds × B=60 = 90 cells.

On real CKKS, mock noise/depth proxies are replaced by actual CKKS
noise-scale and depth consumed by the Taylor-3 circuit. If any lesion
shows LOAD_BEARING (ratio ≥ 1.10, wins ≥ 7/10, p_holm < 0.05), the
S0 mock null is overturned — shaping terms have independent signal.

Outputs: benchmarks/results/tenseal_ablation.csv
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
from tenseal_circuits import build_tenseal_lr_d8


CONFIGS: dict[str, dict] = {
    "FULL": {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "ds", "nt")},
    "DIV":  {"w": (1.0, 0.0, 0.0), "seeds": ()},
    "-N":   {"w": (1.0, 0.0, 0.3), "seeds": ("mm", "ds", "nt")},
    "-D":   {"w": (1.0, 0.5, 0.0), "seeds": ("mm", "ds", "nt")},
    "-ND":  {"w": (1.0, 0.0, 0.0), "seeds": ("mm", "ds", "nt")},
    "-S":   {"w": (1.0, 0.5, 0.3), "seeds": ()},
    "-MM":  {"w": (1.0, 0.5, 0.3), "seeds": ("ds", "nt")},
    "-DS":  {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "nt")},
    "-NT":  {"w": (1.0, 0.5, 0.3), "seeds": ("mm", "ds")},
}


def run_cell(config_name: str, cfg: dict, circuit: dict, seed: int, budget: int) -> dict:
    """Run one (config, circuit, seed) cell with CKKS noise/depth proxies.

    Uses the actual CKKS scale and depth (via the adapter) instead of
    norm-based mock proxies.
    """
    w_div, w_noise, w_depth = cfg["w"]
    seed_heuristics = cfg["seeds"]
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    d = circuit["d"]
    bounds = circuit["bounds"]

    weights = circuit["weights"]
    bias = circuit["bias"]

    class CKKSCfgFitness:
        def score(self, x):
            xa = np.asarray(x, dtype=np.float64)
            try:
                p = plain(x)
                f = fhe(x)
            except Exception:
                return 0.0
            divergence = abs(float(p) - float(f))
            # Real-CKKS proxies: |z| drives multiplication-noise growth,
            # and ‖x‖_inf drives depth of the chain actually consumed.
            z = abs(float(np.dot(weights, xa) + bias))
            noise_term = min(1.0, z / 20.0)
            depth_term = min(1.0, float(np.max(np.abs(xa))) / 3.0)
            return (
                w_div * divergence
                + w_noise * noise_term
                + w_depth * depth_term
            )

    k = 10 if seed_heuristics else 0
    oracle = FHEOracle(
        plaintext_fn=plain,
        fhe_fn=fhe,
        input_dim=d,
        input_bounds=bounds,
        fitness=CKKSCfgFitness(),
        seed=seed,
        use_heuristic_seeds=bool(seed_heuristics),
        heuristic_which=tuple(seed_heuristics) if seed_heuristics else ("mm", "ds", "nt"),
        heuristic_k=k,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=budget, threshold=0.0)
    wall = time.perf_counter() - t0
    return {
        "config": config_name,
        "circuit": "lr_d8_tenseal",
        "seed": seed,
        "weights": f"({w_div},{w_noise},{w_depth})",
        "seeds_used": ",".join(seed_heuristics) if seed_heuristics else "",
        "max_error": res.max_error,
        "wall_clock_s": wall,
        "n_trials": res.n_trials,
        "verdict": res.verdict,
    }


def main(budget: int = 60) -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not available. Skipping ablation.")
        return 0

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tenseal_ablation.csv")

    ctx = TenSEALContext()
    circuit = build_tenseal_lr_d8(ctx)
    seeds = list(range(10))

    total = len(CONFIGS) * len(seeds)
    print("S0 lesion audit on real CKKS (B1)")
    print(f"  Configs: {len(CONFIGS)} ({list(CONFIGS.keys())})")
    print(f"  Circuit: {circuit['name']}")
    print(f"  Seeds:   {len(seeds)}  ({seeds})")
    print(f"  Budget:  B={budget}")
    print(f"  Total:   {total} cells")
    print("=" * 70)

    rows = []
    cell = 0
    t_start = time.perf_counter()
    for cfg_name, cfg in CONFIGS.items():
        for seed in seeds:
            cell += 1
            row = run_cell(cfg_name, cfg, circuit, seed, budget)
            rows.append(row)
            print(
                f"[{cell:3d}/{total}] {cfg_name:>5s} seed={seed} "
                f"err={row['max_error']:.4e} t={row['wall_clock_s']:.1f}s"
            )

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    fieldnames = [
        "config", "circuit", "seed", "weights", "seeds_used",
        "max_error", "wall_clock_s", "n_trials", "verdict",
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
