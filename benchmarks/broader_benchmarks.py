# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Broader external benchmarks: Concrete-ML compiler-generated circuits.

Addresses Limitation 2 of the v5 paper: all prior external benchmarks
are hand-built TenSEAL circuits. Here we run the oracle against
*compiler-generated* circuits where precision parameters (8-bit
quantization, lookup-table sizes, programmable bootstrapping noise
budget) are chosen by Concrete-ML, not by the paper's author.

Setup
-----
- plaintext_fn : sklearn.linear_model.LogisticRegression.predict_proba
- fhe_fn       : concrete.ml.sklearn.LogisticRegression.predict_proba(fhe="execute")
- divergence   : |p_sk(class=1) - p_cml(class=1)|

Note on TFHE semantics
----------------------
Concrete-ML's TFHE compilation is deterministic at the chosen
quantization level: empirically `predict_proba(fhe='execute')` is
bit-identical to `predict_proba(fhe='disable')`. The divergence
between sklearn and Concrete-ML is therefore dominated by 8-bit
input/output quantization and PBS lookup-table truncation — i.e.
the precision regime a TFHE compiler must reason about. We run
the oracle in pure-divergence mode (no CKKS shaping).

Protocol
--------
- Bounds: [-3, 3]^d (matches WDBC / MNIST standardised feature scale)
- Budget: B = 60 per leg (matched real-FHE protocol)
- Seeds: 10
- Strategies: oracle (separable=True, random_floor=0.3) vs uniform random
- Threshold for verdict: 0.01

Output
------
- benchmarks/results/broader_benchmarks.csv (per-cell rows)
- benchmarks/results/broader_benchmarks_summary.csv (per-circuit summary)
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhe_oracle import FHEOracle


THRESHOLD = 0.01
SEEDS = list(range(10))
BUDGET = 60
N_BITS = 8


def _bounds_for(d: int, lo: float = -3.0, hi: float = 3.0):
    return [(lo, hi)] * d


def _build_concrete_circuit(
    dataset: str, random_state: int = 42
) -> tuple[Callable, Callable, np.ndarray, int]:
    """Train sklearn LR (plaintext ref) + Concrete-ML LR (FHE under test).

    Returns (plaintext_fn, fhe_fn, X_train_scaled, dim).
    """
    try:
        from concrete.ml.sklearn import LogisticRegression as CMLLR
    except ImportError as exc:
        raise RuntimeError(
            "concrete-ml is required for this benchmark. Install on Python "
            "<3.13 with: pip install concrete-ml"
        ) from exc

    from sklearn.linear_model import LogisticRegression as SKLR
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if dataset == "wdbc":
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
    elif dataset == "mnist":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        # binary: digit 0 vs rest
        y = (y == 0).astype(int)
    elif dataset.startswith("synth"):
        # synthetic high-dim binary classification (Phase 5; addresses
        # Limitation 2 sub-claim (c): d >= 200).
        from sklearn.datasets import make_classification
        d_target = int(dataset[len("synth"):])
        n_inf = max(10, d_target // 10)
        X, y = make_classification(
            n_samples=max(400, 4 * d_target),
            n_features=d_target,
            n_informative=n_inf,
            n_redundant=0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    X = X.astype(np.float32)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, _, ytr, _ = train_test_split(
        Xs, y, test_size=0.2, random_state=random_state, stratify=y
    )

    sk = SKLR(max_iter=1000, random_state=random_state).fit(Xtr, ytr)
    cml = CMLLR(n_bits=N_BITS).fit(Xtr, ytr)
    cml.compile(Xtr)

    d = Xtr.shape[1]

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return float(sk.predict_proba(xa)[0, 1])

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return float(cml.predict_proba(xa, fhe="execute")[0, 1])

    return plaintext_fn, fhe_fn, Xtr, d


def _random_baseline(
    plain: Callable,
    fhe: Callable,
    bounds: list[tuple[float, float]],
    budget: int,
    seed: int,
) -> tuple[float, str, float]:
    """Uniform random baseline. Returns (max_error, verdict, wall_s)."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best = 0.0
    t0 = time.perf_counter()
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            err = abs(plain(x) - fhe(x))
        except Exception:
            continue
        if err > best:
            best = err
    verdict = "FAIL" if best >= THRESHOLD else "PASS"
    return best, verdict, time.perf_counter() - t0


def _oracle_run(
    plain: Callable,
    fhe: Callable,
    bounds: list[tuple[float, float]],
    budget: int,
    seed: int,
) -> tuple[float, str, float]:
    """Oracle run with the B2 stack. Returns (max_error, verdict, wall_s)."""
    oracle = FHEOracle(
        plaintext_fn=plain,
        fhe_fn=fhe,
        input_dim=len(bounds),
        input_bounds=bounds,
        seed=seed,
        separable=True,
        random_floor=0.3,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=budget, threshold=THRESHOLD)
    return res.max_error, res.verdict, time.perf_counter() - t0


def run_circuit(circuit_name: str, dataset: str) -> list[dict]:
    """Run oracle vs random over SEEDS for one circuit. Returns rows."""
    print(f"  building circuit ({dataset})...", flush=True)
    plain, fhe, _Xtr, d = _build_concrete_circuit(dataset)
    bounds = _bounds_for(d)
    print(f"    d={d}, n_bits={N_BITS}, bounds=[-3,3]^d", flush=True)
    rows: list[dict] = []

    for seed in SEEDS:
        o_err, o_v, o_t = _oracle_run(plain, fhe, bounds, BUDGET, seed)
        r_err, r_v, r_t = _random_baseline(plain, fhe, bounds, BUDGET, seed)
        ratio = o_err / r_err if r_err > 0 else float("inf")
        print(
            f"    seed={seed:2d}  oracle={o_err:.4e} ({o_t:.1f}s) "
            f"random={r_err:.4e} ({r_t:.1f}s)  ratio={ratio:.3f}",
            flush=True,
        )
        rows.append({
            "circuit": circuit_name,
            "d": d,
            "seed": seed,
            "n_bits": N_BITS,
            "oracle_max_error": o_err,
            "random_max_error": r_err,
            "ratio": ratio,
            "oracle_verdict": o_v,
            "random_verdict": r_v,
            "oracle_wall_s": o_t,
            "random_wall_s": r_t,
        })
    return rows


def _summarise(rows: list[dict]) -> list[dict]:
    """Per-circuit summary: median ratio, wins, p-value, mean wall-clock."""
    from collections import defaultdict

    by_circuit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_circuit[r["circuit"]].append(r)

    out: list[dict] = []
    for name, items in by_circuit.items():
        oracle_errs = np.array([r["oracle_max_error"] for r in items])
        random_errs = np.array([r["random_max_error"] for r in items])
        ratios = np.array([r["ratio"] for r in items])
        wins = int((oracle_errs > random_errs).sum())
        ties = int((oracle_errs == random_errs).sum())
        # Sign test p-value (two-sided): under H0 oracle == random,
        # P(>= wins) follows Binom(n, 0.5). We report one-sided since
        # the alternative is oracle > random.
        n = len(items)
        from math import comb
        p_one_sided = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n

        out.append({
            "circuit": name,
            "d": items[0]["d"],
            "n_bits": items[0]["n_bits"],
            "n_seeds": n,
            "median_ratio": float(np.median(ratios[np.isfinite(ratios)]))
            if np.any(np.isfinite(ratios)) else float("nan"),
            "mean_oracle_max_error": float(oracle_errs.mean()),
            "mean_random_max_error": float(random_errs.mean()),
            "wins": wins,
            "ties": ties,
            "p_value_sign_test_one_sided": p_one_sided,
            "mean_oracle_wall_s": float(np.mean([r["oracle_wall_s"] for r in items])),
            "mean_random_wall_s": float(np.mean([r["random_wall_s"] for r in items])),
        })
    return out


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "broader_benchmarks.csv")
    summary_path = os.path.join(out_dir, "broader_benchmarks_summary.csv")

    print("Broader external benchmarks (Limitation 2)")
    print(f"  Compiler  : Concrete-ML (TFHE)")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Budget    : {BUDGET} per leg")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  n_bits    : {N_BITS}")
    print("=" * 70)

    t_start = time.perf_counter()
    all_rows: list[dict] = []

    print("[1/3] Concrete-ML WDBC LR (d=30)...")
    all_rows.extend(run_circuit("concrete_wdbc_lr", "wdbc"))

    print("[2/3] Concrete-ML MNIST(load_digits) LR (d=64)...")
    all_rows.extend(run_circuit("concrete_mnist_lr", "mnist"))

    print("[3/3] Concrete-ML synthetic LR (d=200) — high-dim leg...")
    all_rows.extend(run_circuit("concrete_synth_d200_lr", "synth200"))

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    # Per-cell rows
    fieldnames = list(all_rows[0].keys())
    with open(rows_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"  wrote {len(all_rows)} rows -> {rows_path}")

    # Summary
    summary = _summarise(all_rows)
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"  wrote {len(summary)} summary rows -> {summary_path}")

    print("\nSummary:")
    for s in summary:
        print(
            f"  {s['circuit']:25s}  d={s['d']:3d}  "
            f"median_ratio={s['median_ratio']:.3f}  "
            f"wins={s['wins']}/{s['n_seeds']}  "
            f"p={s['p_value_sign_test_one_sided']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
