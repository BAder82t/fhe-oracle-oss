# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2 — Taylor-vs-CKKS parameter ablation.

Tests whether CKKS parameters (scale) produce measurable signal
once polynomial-approximation error is reduced to near the CKKS
residual noise floor. Three polynomial arms × two scales × five
seeds × B = 40:

- Taylor-3 (paper's current polynomial) — fit error ~1e-1.
- Chebyshev-7 — fit error ~1e-3, transition regime.
- Chebyshev-15 — fit error ~1e-6, comparable to CKKS residual noise.

For each ``(poly_degree, scale_bits, seed)`` cell, runs the oracle
and a matched-budget random baseline, then evaluates the error
decomposition at the oracle's worst-case input:

- ``poly_error``  = |σ(z_worst) − poly_plaintext(z_worst)|
- ``ckks_error``  = |poly_plaintext(z_worst) − poly_ckks(z_worst)|
- ``total_error`` = |σ(z_worst) − poly_ckks(z_worst)| (≡ oracle_max_error)

Chain depth is fixed per polynomial degree at the minimum viable
probed recipe (see ``chebyshev_polynomials._DEGREE_PARAMS``).
Varying chain depth at fixed degree would require building a
second-dimension sweep and is deferred — the scale sweep alone
answers the primary question ("do CKKS parameters matter when the
polynomial floor is removed?").

Output: ``benchmarks/results/taylor_ckks_ablation.csv``
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
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL  # noqa: E402

from chebyshev_polynomials import (  # noqa: E402
    build_tenseal_context,
    eval_poly_plaintext,
    fit_cheb_sigmoid,
    make_tenseal_poly_lr_fhe_fn,
    taylor3_approx,
)
from tenseal_circuits import _fit_lr_synthetic  # noqa: E402


SEEDS = list(range(5))
BUDGET = 40
SCALE_BITS = [30, 40]
POLY_ARMS: list[tuple[str, int]] = [
    ("taylor3", 3),
    ("cheb7", 7),
    ("cheb15", 15),
]


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))


def run_random_baseline(plaintext_fn, fhe_fn, bounds, budget, seed):
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best_err = -np.inf
    best_x = None
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        p = plaintext_fn(x.tolist())
        f = fhe_fn(x.tolist())
        err = abs(float(p) - float(f))
        if err > best_err:
            best_err = err
            best_x = x.copy()
    return float(best_err), best_x


def run_oracle(plaintext_fn, fhe_fn, bounds, dim, seed, budget):
    oracle = FHEOracle(
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
        input_dim=dim,
        input_bounds=bounds,
        seed=seed,
    )
    return oracle.run(n_trials=budget, threshold=0.0)


def decompose_error(
    x: np.ndarray,
    w: np.ndarray,
    b: float,
    approx,
    fhe_fn,
) -> dict[str, float]:
    """Return the three error components at a given input."""
    z = float(np.dot(w, x) + b)
    sigma_true = _sigmoid(z)
    poly_plain = eval_poly_plaintext(z, approx)
    poly_ckks = float(fhe_fn(list(x)))
    return {
        "z_at_worst": z,
        "sigma_true": sigma_true,
        "poly_plaintext": poly_plain,
        "poly_ckks": poly_ckks,
        "poly_error": abs(sigma_true - poly_plain),
        "ckks_error": abs(poly_plain - poly_ckks),
        "total_error": abs(sigma_true - poly_ckks),
    }


def run_arm(
    arm_name: str,
    degree: int,
    scale_bits: int,
    w: np.ndarray,
    b: float,
    bounds: list[tuple[float, float]],
    dim: int,
) -> list[dict]:
    """Run a full (polynomial, scale) cell across all seeds."""
    if degree == 3 and arm_name == "taylor3":
        approx = taylor3_approx()
    else:
        approx = fit_cheb_sigmoid(degree)

    ctx = build_tenseal_context(degree=degree, scale_bits=scale_bits)
    fhe_fn = make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx)

    def plaintext_fn(x):
        z = float(np.dot(w, np.asarray(x)) + b)
        return _sigmoid(z)

    rows: list[dict] = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(plaintext_fn, fhe_fn, bounds, dim, seed, BUDGET)
        t_oracle = time.perf_counter() - t0

        t1 = time.perf_counter()
        rnd_err, rnd_x = run_random_baseline(
            plaintext_fn, fhe_fn, bounds, BUDGET, seed
        )
        t_random = time.perf_counter() - t1

        decomp_oracle = decompose_error(
            np.asarray(ores.worst_input), w, b, approx, fhe_fn
        )
        decomp_random = decompose_error(rnd_x, w, b, approx, fhe_fn)

        ratio = (
            ores.max_error / rnd_err
            if rnd_err > 0
            else float("inf")
        )
        rows.append(
            {
                "arm": arm_name,
                "poly_degree": degree,
                "scale_bits": scale_bits,
                "poly_fit_error": approx.fit_error,
                "N": ctx.N,
                "chain": "-".join(str(b_) for b_ in ctx.chain),
                "seed": seed,
                "oracle_max_error": ores.max_error,
                "random_max_error": rnd_err,
                "ratio_oracle_over_random": ratio,
                "oracle_wins": int(ores.max_error > rnd_err),
                "z_at_oracle_worst": decomp_oracle["z_at_worst"],
                "oracle_poly_error": decomp_oracle["poly_error"],
                "oracle_ckks_error": decomp_oracle["ckks_error"],
                "oracle_total_error": decomp_oracle["total_error"],
                "random_poly_error": decomp_random["poly_error"],
                "random_ckks_error": decomp_random["ckks_error"],
                "random_total_error": decomp_random["total_error"],
                "oracle_wall_clock_s": t_oracle,
                "random_wall_clock_s": t_random,
            }
        )
        print(
            f"   {arm_name:<8s} scale=2^{scale_bits:<2d} seed={seed}  "
            f"ora={ores.max_error:.3e} rnd={rnd_err:.3e} R={ratio:.2f}x  "
            f"poly_err={decomp_oracle['poly_error']:.3e} "
            f"ckks_err={decomp_oracle['ckks_error']:.3e}  "
            f"t_o={t_oracle:.1f}s"
        )
    return rows


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not installed. Skipping C2 ablation.")
        return 0

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "taylor_ckks_ablation.csv")

    w, b = _fit_lr_synthetic(d=8, seed=42)
    bounds = [(-3.0, 3.0)] * 8
    dim = 8

    total_cells = len(POLY_ARMS) * len(SCALE_BITS)
    print(f"C2 Taylor-vs-CKKS parameter ablation")
    print(f"  Polynomials: {[a[0] for a in POLY_ARMS]}")
    print(f"  Scales:      {[f'2^{s}' for s in SCALE_BITS]}")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Budget:      {BUDGET}")
    print(f"  Circuit:     LR d=8 (synthetic, paper seed=42)")
    print(f"  Output:      {out_path}")
    print("=" * 80)

    rows: list[dict] = []
    t_start = time.perf_counter()
    cell = 0
    for arm_name, degree in POLY_ARMS:
        for scale_bits in SCALE_BITS:
            cell += 1
            print(
                f"\n[{cell}/{total_cells}] {arm_name} (degree={degree}) @ "
                f"scale=2^{scale_bits}"
            )
            rows.extend(
                run_arm(arm_name, degree, scale_bits, w, b, bounds, dim)
            )

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
