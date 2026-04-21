# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2 n=10 extension — append seeds 5-9 to taylor_ckks_ablation.csv.

Runs Cheb-15 only (the gate-relevant arm) at scales {2^30, 2^40} for
seeds 5-9, mirroring exactly the protocol of taylor_ckks_ablation.py.
Appends the new rows to the existing CSV. Re-running is safe: the
script de-duplicates rows by (arm, scale_bits, seed).
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


NEW_SEEDS = [5, 6, 7, 8, 9]
BUDGET = 40
SCALE_BITS = [30, 40]
DEGREE = 15
ARM = "cheb15"


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


def run_arm_seeds(degree, scale_bits, w, b, bounds, dim, seeds):
    approx = fit_cheb_sigmoid(degree)
    ctx = build_tenseal_context(degree=degree, scale_bits=scale_bits)
    fhe_fn = make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx)

    def plaintext_fn(x):
        z = float(np.dot(w, np.asarray(x)) + b)
        return _sigmoid(z)

    rows = []
    for seed in seeds:
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
                "arm": ARM,
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
            f"   {ARM} scale=2^{scale_bits} seed={seed}  "
            f"ora={ores.max_error:.3e} rnd={rnd_err:.3e} R={ratio:.2f}x  "
            f"poly_err={decomp_oracle['poly_error']:.3e} "
            f"ckks_err={decomp_oracle['ckks_error']:.3e}  "
            f"t_o={t_oracle:.1f}s"
        )
    return rows


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL not installed. Skipping.")
        return 0

    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results", "taylor_ckks_ablation.csv")

    w, b = _fit_lr_synthetic(d=8, seed=42)
    bounds = [(-3.0, 3.0)] * 8
    dim = 8

    print("C2 n=10 extension: Cheb-15, seeds 5-9, scales {2^30, 2^40}")
    print("=" * 80)

    new_rows = []
    t_start = time.perf_counter()
    for scale_bits in SCALE_BITS:
        print(f"\nCheb-15 @ scale=2^{scale_bits}")
        new_rows.extend(
            run_arm_seeds(DEGREE, scale_bits, w, b, bounds, dim, NEW_SEEDS)
        )
    print("=" * 80)
    print(f"Wall-clock: {time.perf_counter() - t_start:.1f}s")

    # Read existing rows; de-dup on (arm, scale_bits, seed).
    existing = []
    with open(out_path) as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for r in reader:
            existing.append(r)

    new_keys = {(r["arm"], int(r["scale_bits"]), int(r["seed"])) for r in new_rows}
    filtered = [
        r for r in existing
        if (r["arm"], int(r["scale_bits"]), int(r["seed"])) not in new_keys
    ]

    combined = filtered + [{k: r[k] for k in fieldnames} for r in new_rows]
    combined.sort(
        key=lambda r: (
            int(r["poly_degree"]),
            int(r["scale_bits"]),
            int(r["seed"]),
        )
    )

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in combined:
            writer.writerow(r)
    print(f"Wrote {len(combined)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
