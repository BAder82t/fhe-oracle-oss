# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cheb-15 cross-circuit replication (resolves paper Limitation 10).

The C2 ablation (§6.15) showed η²=0.254 for scale-bits at Cheb-15 on
LR d=8. Limitation 10 asks whether that POLY→CKKS flip generalizes
beyond the synthetic LR circuit.

Primary target: WDBC d=30 (same sigmoid/LR structure, different
weights, different dimension, real dataset). TenSEAL Cheb-15 at
N=32768 with a 16-level chain is known to work at d=8 (see
taylor_ckks_ablation.csv). The only new cost is encrypting 30 slots
instead of 8, which does not change the chain budget.

Arms:
  - cheb15 × scale_bits ∈ {30, 40} × seeds 0..9
  - n=10, B=40 (matches the original C2 protocol so ANOVA is comparable)

For each cell we report:
  oracle_max_error, random_max_error, ratio, wins,
  poly_error, ckks_error, dominant component.

Produces:
  benchmarks/results/cheb15_cross_circuit.csv
  benchmarks/results/cheb15_cross_circuit_summary.csv
     (per-arm aggregates + ANOVA η² for scale_bits effect)
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL  # noqa: E402

from chebyshev_polynomials import (  # noqa: E402
    build_tenseal_context,
    eval_poly_plaintext,
    fit_cheb_sigmoid,
    make_tenseal_poly_lr_fhe_fn,
)


SEEDS = list(range(10))
BUDGET = 40
SCALE_BITS = [30, 40]
DEGREE = 15
RESULTS = os.path.join(THIS_DIR, "results")


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))))


def _fit_wdbc_model():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    return w, b, X


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
        plaintext_fn=plaintext_fn, fhe_fn=fhe_fn,
        input_dim=dim, input_bounds=bounds, seed=seed,
    )
    return oracle.run(n_trials=budget, threshold=0.0)


def decompose_error(x, w, b, approx, fhe_fn):
    z = float(np.dot(w, x) + b)
    sigma_true = _sigmoid(z)
    poly_plain = eval_poly_plaintext(z, approx)
    poly_ckks = float(fhe_fn(list(x)))
    return {
        "z": z,
        "poly_error": abs(sigma_true - poly_plain),
        "ckks_error": abs(poly_plain - poly_ckks),
        "total_error": abs(sigma_true - poly_ckks),
    }


def run_wdbc_cheb15() -> list[dict]:
    print("=" * 80)
    print(f"Cheb-15 cross-circuit on WDBC d=30")
    print("=" * 80)
    w, b, _ = _fit_wdbc_model()
    d = int(w.shape[0])
    # Tight domain: the paper's bounds were [-3, 3] for the LR circuit.
    # WDBC features are standardised so [-3, 3] is ~4σ — reasonable.
    # If |W·x + b| exceeds the fit interval [-5, 5], Cheb-15 extrapolation
    # artefacts dominate. We use [-1, 1] to keep |z| ≈ ‖w‖_1 + |b| in-range.
    w_l1 = float(np.sum(np.abs(w)))
    print(f"  ‖w‖_1 = {w_l1:.3f}, |b| = {abs(b):.3f}")
    # For WDBC, ‖w‖_1 ≈ 13, so even [-0.3, 0.3] yields |z| ≈ 4. Use [-0.3, 0.3].
    bounds_lo, bounds_hi = -0.3, 0.3
    bounds = [(bounds_lo, bounds_hi)] * d
    print(f"  bounds=[{bounds_lo}, {bounds_hi}]^{d}  -> expected |z|_max ≈ {w_l1 * bounds_hi + abs(b):.2f}")

    def plaintext_fn(x):
        z = float(np.dot(w, np.asarray(x)) + b)
        return _sigmoid(z)

    approx = fit_cheb_sigmoid(DEGREE)
    rows: list[dict] = []
    for scale_bits in SCALE_BITS:
        print(f"\n  [scale=2^{scale_bits}]")
        ctx = build_tenseal_context(degree=DEGREE, scale_bits=scale_bits)
        fhe_fn = make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx)
        for seed in SEEDS:
            t0 = time.perf_counter()
            try:
                ores = run_oracle(plaintext_fn, fhe_fn, bounds, d, seed, BUDGET)
            except Exception as exc:  # pragma: no cover
                print(f"    seed={seed} ORACLE FAILED: {exc}")
                continue
            t_oracle = time.perf_counter() - t0
            rnd_err, rnd_x = run_random_baseline(
                plaintext_fn, fhe_fn, bounds, BUDGET, seed
            )
            dec_oracle = decompose_error(
                np.asarray(ores.worst_input), w, b, approx, fhe_fn
            )
            dec_random = decompose_error(rnd_x, w, b, approx, fhe_fn)
            ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
            dominant = (
                "POLY" if dec_oracle["poly_error"] > dec_oracle["ckks_error"]
                else "CKKS"
            )
            print(
                f"    seed={seed:2d} z*={dec_oracle['z']:+.2f}  "
                f"ora={ores.max_error:.3e} rnd={rnd_err:.3e} R={ratio:.2f}x  "
                f"poly={dec_oracle['poly_error']:.3e} "
                f"ckks={dec_oracle['ckks_error']:.3e}  "
                f"dom={dominant} t={t_oracle:.1f}s"
            )
            rows.append({
                "circuit": "wdbc_d30",
                "arm": "cheb15",
                "scale_bits": scale_bits,
                "N": ctx.N,
                "seed": seed,
                "oracle_max_error": ores.max_error,
                "random_max_error": rnd_err,
                "ratio": ratio,
                "oracle_wins": int(ores.max_error > rnd_err),
                "z_at_oracle_worst": dec_oracle["z"],
                "poly_error": dec_oracle["poly_error"],
                "ckks_error": dec_oracle["ckks_error"],
                "total_error": dec_oracle["total_error"],
                "random_poly_error": dec_random["poly_error"],
                "random_ckks_error": dec_random["ckks_error"],
                "dominant": dominant,
                "wall_clock_s": t_oracle,
            })
    return rows


def compute_eta_squared(
    rows: list[dict], group_key: str, value_key: str,
) -> tuple[float, float, float]:
    """One-way ANOVA η² for group_key on value_key. Returns (eta2, F, p)."""
    from scipy.stats import f_oneway

    groups: dict[str, list[float]] = {}
    for r in rows:
        g = str(r[group_key])
        v = float(r[value_key])
        groups.setdefault(g, []).append(v)
    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")
    arrays = list(groups.values())
    overall = np.concatenate(arrays)
    grand_mean = float(np.mean(overall))
    ss_total = float(np.sum((overall - grand_mean) ** 2))
    ss_between = float(sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in arrays
    ))
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    res = f_oneway(*arrays)
    return eta2, float(res.statistic), float(res.pvalue)


def build_summary(rows: list[dict]) -> None:
    out = os.path.join(RESULTS, "cheb15_cross_circuit_summary.csv")
    summary: list[dict] = []

    # Per-arm aggregates.
    for scale in SCALE_BITS:
        sub = [r for r in rows if r["scale_bits"] == scale]
        if not sub:
            continue
        ratios = [r["ratio"] for r in sub if np.isfinite(r["ratio"])]
        poly_errs = [r["poly_error"] for r in sub]
        ckks_errs = [r["ckks_error"] for r in sub]
        oracle_errs = [r["oracle_max_error"] for r in sub]
        wins = sum(1 for r in sub if r["oracle_wins"])
        poly_dom = sum(1 for r in sub if r["dominant"] == "POLY")
        ckks_dom = sum(1 for r in sub if r["dominant"] == "CKKS")
        summary.append({
            "circuit": "wdbc_d30",
            "arm": "cheb15",
            "scale_bits": scale,
            "n": len(sub),
            "mean_oracle_max_error": float(np.mean(oracle_errs)),
            "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "wins": wins,
            "mean_poly_error": float(np.mean(poly_errs)),
            "mean_ckks_error": float(np.mean(ckks_errs)),
            "poly_dominant_count": poly_dom,
            "ckks_dominant_count": ckks_dom,
        })

    # ANOVA across scale_bits on the oracle_max_error.
    eta2, F, p = compute_eta_squared(rows, "scale_bits", "oracle_max_error")
    summary.append({
        "circuit": "wdbc_d30",
        "arm": "ANOVA",
        "scale_bits": "all",
        "n": len(rows),
        "mean_oracle_max_error": float(np.mean([r["oracle_max_error"] for r in rows])),
        "mean_ratio": float("nan"),
        "wins": -1,
        "mean_poly_error": eta2,  # reused slot for eta2
        "mean_ckks_error": F,      # reused slot for F-stat
        "poly_dominant_count": -1,
        "ckks_dominant_count": -1,
    })
    print(
        f"\n  ANOVA scale_bits on oracle_max_error:  "
        f"η²={eta2:.3f}  F={F:.3f}  p={p:.4f}"
    )
    if np.isfinite(eta2):
        if eta2 > 0.10:
            print("  => PASS: POLY→CKKS flip generalizes to WDBC (η² > 0.10)")
        elif eta2 > 0.05:
            print("  => PARTIAL: direction holds but weaker (0.05 < η² ≤ 0.10)")
        else:
            print("  => FAIL: POLY→CKKS flip does not generalize to WDBC (η² ≤ 0.05)")

    fieldnames = [
        "circuit", "arm", "scale_bits", "n",
        "mean_oracle_max_error", "mean_ratio", "wins",
        "mean_poly_error", "mean_ckks_error",
        "poly_dominant_count", "ckks_dominant_count",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"  wrote summary → {out}")


def main() -> int:
    if not HAVE_TENSEAL:
        print("TenSEAL unavailable. Skipping.")
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.perf_counter()
    rows = run_wdbc_cheb15()
    elapsed = time.perf_counter() - t0
    print(f"\nWall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    out = os.path.join(RESULTS, "cheb15_cross_circuit.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {len(rows)} rows to {out}")
        build_summary(rows)
    else:
        print("No rows produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
