# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2 follow-up — Cheb-15 tight-domain confirmation.

Runs Cheb-15 at bounds=[-1, 1]^8 (instead of the paper's [-3, 3]^8)
to keep |z| inside the fit interval [-5, 5]. With ‖w‖_1 ≈ 3.5 on the
synthetic LR weights, |z| stays under ~4.

Goal: confirm whether CKKS dominance persists inside the fit domain
(addresses the C2 "extrapolation artefact" caveat).

5 seeds, scale=2^30 (the scale where CKKS dominates most), B=40.
Output: benchmarks/results/cheb15_tight_domain.csv
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
)
from tenseal_circuits import _fit_lr_synthetic  # noqa: E402


SEEDS = [0, 1, 2, 3, 4]
BUDGET = 40
SCALE_BITS = 30
DEGREE = 15


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


def decompose_error(x, w, b, approx, fhe_fn):
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


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not installed. Skipping.")
        return 0

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cheb15_tight_domain.csv")

    w, b = _fit_lr_synthetic(d=8, seed=42)
    print(f"  ‖w‖_1 = {float(np.sum(np.abs(w))):.3f}, |b| = {abs(float(b)):.3f}")

    bounds = [(-1.0, 1.0)] * 8
    dim = 8

    approx = fit_cheb_sigmoid(DEGREE)
    ctx = build_tenseal_context(degree=DEGREE, scale_bits=SCALE_BITS)
    fhe_fn = make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx)

    def plaintext_fn(x):
        z = float(np.dot(w, np.asarray(x)) + b)
        return _sigmoid(z)

    print(f"Cheb-15 tight domain: bounds=[-1,1]^8, scale=2^{SCALE_BITS}")
    print(f"  expected |z|_max ≈ ‖w‖_1 + |b| ≈ {float(np.sum(np.abs(w))) + abs(float(b)):.2f}")
    print("=" * 80)

    rows = []
    t_start = time.perf_counter()
    for seed in SEEDS:
        t0 = time.perf_counter()
        ores = run_oracle(plaintext_fn, fhe_fn, bounds, dim, seed, BUDGET)
        t_oracle = time.perf_counter() - t0

        rnd_err, rnd_x = run_random_baseline(
            plaintext_fn, fhe_fn, bounds, BUDGET, seed
        )

        decomp_oracle = decompose_error(
            np.asarray(ores.worst_input), w, b, approx, fhe_fn
        )
        decomp_random = decompose_error(rnd_x, w, b, approx, fhe_fn)

        ratio = (
            ores.max_error / rnd_err
            if rnd_err > 0
            else float("inf")
        )
        dominant = (
            "POLY" if decomp_oracle["poly_error"] > decomp_oracle["ckks_error"]
            else "CKKS"
        )
        rows.append(
            {
                "arm": "cheb15",
                "poly_degree": DEGREE,
                "scale_bits": SCALE_BITS,
                "bounds_lo": -1.0,
                "bounds_hi": 1.0,
                "poly_fit_error": approx.fit_error,
                "N": ctx.N,
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
                "dominant": dominant,
                "oracle_wall_clock_s": t_oracle,
            }
        )
        print(
            f"   seed={seed} z*={decomp_oracle['z_at_worst']:+.2f}  "
            f"ora={ores.max_error:.3e} rnd={rnd_err:.3e} R={ratio:.2f}x  "
            f"poly_err={decomp_oracle['poly_error']:.3e} "
            f"ckks_err={decomp_oracle['ckks_error']:.3e}  "
            f"dom={dominant} t={t_oracle:.1f}s"
        )
    print("=" * 80)
    print(f"Wall-clock: {time.perf_counter() - t_start:.1f}s")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")

    poly_dom = sum(1 for r in rows if r["dominant"] == "POLY")
    ckks_dom = sum(1 for r in rows if r["dominant"] == "CKKS")
    print(f"\nDominant component breakdown: POLY={poly_dom}, CKKS={ckks_dom}")
    print("Median |z| at oracle worst:",
          float(np.median([abs(r["z_at_oracle_worst"]) for r in rows])))
    print("Verdict:",
          "CKKS dominates inside fit domain" if ckks_dom > poly_dom
          else "POLY dominates inside fit domain (extrapolation artefact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
