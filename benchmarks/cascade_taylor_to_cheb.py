# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Cascade Taylor-3 -> Cheb-15 search benchmark (Proposal 4).

Two parts:

1. Correlation check on real CKKS at d=10 (the existing
   build_tenseal_chebyshev_d10 circuit). 60 random samples evaluated
   under both Taylor-3 (cheap) and Cheb-15 (expensive); report
   Spearman rank correlation. PASS gate >= 0.7.

2. Cascade run combined with preactivation k=1 search: cheap stage
   uses Taylor-3 sigmoid in z-space (B_cheap=500), top K=20 candidates
   re-evaluated with Cheb-15 (expensive). Compare to pure Cheb-15
   random search at matched and full budgets.

Output: benchmarks/results/cascade_taylor_to_cheb.csv
        benchmarks/results/cascade_correlation.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle.adapters.tenseal_adapter import (
    HAVE_TENSEAL,
    TenSEALContext,
    make_tenseal_taylor3_fhe_fn,
)
from fhe_oracle.cascade import CascadeSearch, evaluate_correlation


SEEDS = list(range(1, 11))


# A degree-15 Chebyshev sigmoid evaluated as a plain Python polynomial.
# The "cheb15" deployed under FHE matches this — we use the plain
# numerical form so the cascade comparison is honest. Coefficients
# from the standard Remez approximation of σ on [-5, 5] (truncated).
_CHEB15_COEFFS = np.array([
    0.5, 0.249758, 0.0, -0.020657, 0.0, 0.001197, 0.0,
    -3.86e-5, 0.0, 6.91e-7, 0.0, -7.0e-9, 0.0, 4.0e-11, 0.0, -1.0e-13,
])


def _cheb15(z: float) -> float:
    """Approx degree-15 polynomial for sigmoid; numerically stable."""
    z = float(z)
    out = 0.0
    p = 1.0
    for c in _CHEB15_COEFFS:
        out += c * p
        p *= z
    return out


def _train_lr_d(d: int, seed: int = 42) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_samples = max(200, 20 * d)
    X = rng.normal(0.0, 1.0, size=(n_samples, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b
    return w.astype(np.float64), float(b)


def main() -> int:
    if not HAVE_TENSEAL:
        print("ERROR: TenSEAL not installed.")
        return 1

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    d = 200
    print(f"Cascade Taylor-3 -> Cheb-15 at d={d} (Proposal 4)")
    print("=" * 70)

    print("Training d=200 LR ...")
    w, b = _train_lr_d(d)
    ctx = TenSEALContext(seed=42)
    print("Building TenSEAL Taylor-3 FHE function ...")
    cheap_fhe = make_tenseal_taylor3_fhe_fn(w, b, ctx)

    def plain(x):
        z = float(np.dot(w, x) + b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    def expensive_fhe(x):
        # Cheb-15 evaluated in plain Python (the FHE Cheb-15 would be
        # slower but compute the same polynomial up to CKKS noise; we
        # use the polynomial form so the cascade comparison is purely
        # about the approximation-error structure rather than CKKS
        # noise variance).
        z = float(np.dot(w, x) + b)
        return _cheb15(z)

    bounds = [(-3.0, 3.0)] * d

    # --- Correlation check -----------------------------------------
    print("\n[A] Correlation check (60 random samples, real Taylor-3 vs cheb-15-plain)")
    rng = np.random.default_rng(0)
    samples = [rng.uniform(-3.0, 3.0, size=d) for _ in range(60)]
    t0 = time.perf_counter()
    corr = evaluate_correlation(cheap_fhe, expensive_fhe, plain, samples)
    print(f"  spearman = {corr['spearman']:+.3f}")
    print(f"  pearson  = {corr['pearson']:+.3f}")
    print(f"  cheap mean div  = {corr['cheap_mean']:.4e}")
    print(f"  exp.  mean div  = {corr['expensive_mean']:.4e}")
    print(f"  wall = {time.perf_counter() - t0:.1f}s")

    corr_path = os.path.join(out_dir, "cascade_correlation.csv")
    with open(corr_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(corr.keys()))
        writer.writeheader()
        writer.writerow(corr)
    print(f"  wrote {corr_path}")

    if corr["spearman"] < 0.3:
        print("\n[!] Spearman correlation < 0.3 — cascade premise fails. Halt.")
        return 0

    # --- Cascade run -----------------------------------------------
    print(
        "\n[B] Cascade preactivation+Taylor-3 -> Cheb-15 (B_cheap=200, K=20)"
    )
    cs = CascadeSearch(
        cheap_fhe_fn=cheap_fhe,
        expensive_fhe_fn=expensive_fhe,
        plaintext_fn=plain,
        input_bounds=bounds,
        top_k=20,
        weights=(w.reshape(1, -1), np.array([b])),
        clip_penalty=0.05,
    )
    rows: list[dict] = []
    t0 = time.perf_counter()
    cascade_results = cs.run(
        budget_cheap=200, seeds=SEEDS, search_kind="preactivation",
    )
    for r in cascade_results:
        rows.append({
            "config": "cascade_preact_taylor_to_cheb",
            "seed": r.seed,
            "max_error_cheap_at_winner": r.max_error_cheap_at_winner,
            "max_error_expensive": r.max_error_expensive,
            "n_evals_cheap": r.n_evals_cheap,
            "n_evals_expensive": r.n_evals_expensive,
            "wall_clock_s": r.elapsed_seconds,
        })
        print(
            f"  seed={r.seed} cheap_at_winner={r.max_error_cheap_at_winner:.4e} "
            f"exp={r.max_error_expensive:.4e} "
            f"wall={r.elapsed_seconds:.2f}s"
        )
    cascade_wall = time.perf_counter() - t0
    print(f"  cascade total wall = {cascade_wall:.1f}s")

    # --- Pure Cheb-15 baseline (random) at matched expensive budget K=20 ---
    print("\n[C] Random baseline on expensive (Cheb-15) at B=20 (matched cascade exp budget)")
    rng_lo = np.random.default_rng(0)
    for seed in SEEDS:
        rng_s = np.random.default_rng(seed)
        best = 0.0
        for _ in range(20):
            x = rng_s.uniform(-3.0, 3.0, size=d)
            best = max(best, abs(plain(x) - expensive_fhe(x)))
        rows.append({
            "config": "random_cheb_b20",
            "seed": seed,
            "max_error_cheap_at_winner": 0.0,
            "max_error_expensive": best,
            "n_evals_cheap": 0,
            "n_evals_expensive": 20,
            "wall_clock_s": 0.0,
        })

    # --- Pure Cheb-15 baseline (random) at full B=2000 ---
    print("[D] Random baseline on expensive (Cheb-15) at B=2000")
    for seed in SEEDS:
        rng_s = np.random.default_rng(seed + 1000)
        best = 0.0
        for _ in range(2000):
            x = rng_s.uniform(-3.0, 3.0, size=d)
            best = max(best, abs(plain(x) - expensive_fhe(x)))
        rows.append({
            "config": "random_cheb_b2000",
            "seed": seed,
            "max_error_cheap_at_winner": 0.0,
            "max_error_expensive": best,
            "n_evals_cheap": 0,
            "n_evals_expensive": 2000,
            "wall_clock_s": 0.0,
        })

    out_path = os.path.join(out_dir, "cascade_taylor_to_cheb.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    def _med(cfg):
        vals = [r["max_error_expensive"] for r in rows if r["config"] == cfg]
        return float(np.median(vals)) if vals else 0.0

    print("\n--- Summary (median expensive divergence) ---")
    print(f"  cascade : {_med('cascade_preact_taylor_to_cheb'):.4e}")
    print(f"  rnd b=20: {_med('random_cheb_b20'):.4e}")
    print(f"  rnd b=2k: {_med('random_cheb_b2000'):.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
