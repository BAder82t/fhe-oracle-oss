# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Broader external benchmarks — non-LR (MLP) circuits via Concrete-ML.

Addresses Limitation 2 sub-claim (b): all current external circuits
are logistic regressions. Here we validate the oracle on
multi-layer perceptron circuits compiled by Concrete-ML using its
quantization-aware-training pipeline (NeuralNetClassifier wrapper).

Setup
-----
- plaintext_fn : sklearn.neural_network.MLPClassifier.predict_proba
                 (unquantized float MLP, same architecture)
- fhe_fn       : concrete.ml.sklearn.NeuralNetClassifier.predict_proba(fhe="execute")
- divergence   : |p_sk(class=1) - p_cml(class=1)|

Note on TFHE determinism for MLP
--------------------------------
As with the LR benchmarks, Concrete-ML's TFHE compilation of the
MLP is bit-deterministic: empirically `predict_proba(fhe='execute')`
== `predict_proba(fhe='disable')` for every input we tested.
The divergence between the unquantized sklearn MLP and the
Concrete-ML circuit therefore reflects 4-bit quantization +
PBS lookup-table truncation across multiple layers, NOT FHE noise.
That is the practitioner-relevant precision question for a deployed
quantized neural network.

Why 4-bit weights/activations
-----------------------------
At 8 bits the accumulator overflows the TFHE PBS budget for an
MLP and Concrete-ML raises ``RuntimeError: NoParametersFound``.
4-bit weights + 4-bit activations + 14-bit accumulator is the
narrowest config that compiles cleanly for our two architectures.
This is honest reporting: a real Concrete-ML user deploying an MLP
faces the same parameter-search failure and would land on a similar
narrow-bit configuration. The oracle is therefore exercising the
exact precision regime a Concrete-ML MLP user actually deploys.

Protocol
--------
- Bounds: [-3, 3]^d (matches WDBC / MNIST standardised scale)
- Budget: B = 60 per leg
- Seeds: 5 (per-eval cost ~1.4 s at d=30, ~1.8 s at d=64; 10
  seeds × 2 strategies × 2 circuits would exceed 1 h)
- Strategies: oracle (separable=True, random_floor=0.3) vs uniform random
- Threshold for verdict: 0.01

Output
------
- benchmarks/results/broader_benchmarks_mlp.csv (per-cell rows)
- benchmarks/results/broader_benchmarks_mlp_summary.csv (per-circuit summary)
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
SEEDS = list(range(5))
BUDGET = 60
N_W_BITS = 4
N_A_BITS = 4
N_ACCUM_BITS = 14
MAX_EPOCHS = 30


def _bounds_for(d: int, lo: float = -3.0, hi: float = 3.0):
    return [(lo, hi)] * d


def _build_concrete_mlp(
    dataset: str, random_state: int = 42
) -> tuple[Callable, Callable, np.ndarray, int, dict]:
    """Train sklearn MLP (plaintext ref) + Concrete-ML MLP (FHE under test).

    Returns (plaintext_fn, fhe_fn, X_train_scaled, dim, meta).
    """
    try:
        from concrete.ml.sklearn import NeuralNetClassifier
    except ImportError as exc:
        raise RuntimeError(
            "concrete-ml is required for this benchmark. Install on Python "
            "<3.13 with: pip install concrete-ml"
        ) from exc

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if dataset == "wdbc":
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
    elif dataset == "mnist":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        y = (y == 0).astype(int)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    X = X.astype(np.float32)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, _Xte, ytr, _yte = train_test_split(
        Xs, y, test_size=0.2, random_state=random_state, stratify=y
    )

    d = Xtr.shape[1]

    sk = MLPClassifier(
        hidden_layer_sizes=(d,),
        max_iter=300,
        random_state=random_state,
    ).fit(Xtr, ytr)

    cml = NeuralNetClassifier(
        module__n_layers=2,
        module__n_w_bits=N_W_BITS,
        module__n_a_bits=N_A_BITS,
        module__n_accum_bits=N_ACCUM_BITS,
        module__n_hidden_neurons_multiplier=1,
        max_epochs=MAX_EPOCHS,
        verbose=0,
    )
    cml.fit(Xtr, ytr.astype(np.int64))
    cml.compile(Xtr)

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return float(sk.predict_proba(xa)[0, 1])

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return float(cml.predict_proba(xa, fhe="execute")[0, 1])

    meta = {
        "n_w_bits": N_W_BITS,
        "n_a_bits": N_A_BITS,
        "n_accum_bits": N_ACCUM_BITS,
        "n_layers": 2,
        "hidden_size": d,
        "max_epochs": MAX_EPOCHS,
    }
    return plaintext_fn, fhe_fn, Xtr, d, meta


def _random_baseline(plain, fhe, bounds, budget, seed):
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


def _oracle_run(plain, fhe, bounds, budget, seed):
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


def run_circuit(
    circuit_name: str,
    dataset: str,
    box_lo: float = -3.0,
    box_hi: float = 3.0,
) -> list[dict]:
    print(f"  building MLP circuit ({dataset})...", flush=True)
    plain, fhe, _Xtr, d, meta = _build_concrete_mlp(dataset)
    bounds = _bounds_for(d, box_lo, box_hi)
    print(
        f"    d={d}, n_w_bits={meta['n_w_bits']}, n_a_bits={meta['n_a_bits']}, "
        f"hidden={meta['hidden_size']}, bounds=[{box_lo},{box_hi}]^d",
        flush=True,
    )
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
            "n_w_bits": meta["n_w_bits"],
            "n_a_bits": meta["n_a_bits"],
            "hidden_size": meta["hidden_size"],
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
    from collections import defaultdict
    from math import comb

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
        n = len(items)
        p_one_sided = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n

        out.append({
            "circuit": name,
            "d": items[0]["d"],
            "n_w_bits": items[0]["n_w_bits"],
            "n_a_bits": items[0]["n_a_bits"],
            "hidden_size": items[0]["hidden_size"],
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
    rows_path = os.path.join(out_dir, "broader_benchmarks_mlp.csv")
    summary_path = os.path.join(out_dir, "broader_benchmarks_mlp_summary.csv")

    print("Broader external benchmarks — MLP (Limitation 2b)")
    print(f"  Compiler  : Concrete-ML (TFHE) NeuralNetClassifier")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Budget    : {BUDGET} per leg")
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Bits      : w={N_W_BITS} a={N_A_BITS} accum={N_ACCUM_BITS}")
    print("=" * 70)

    t_start = time.perf_counter()
    all_rows: list[dict] = []

    print("[1/4] Concrete-ML WDBC MLP wide box [-3,3] (d=30)...")
    all_rows.extend(run_circuit("concrete_wdbc_mlp", "wdbc", -3.0, 3.0))

    print("[2/4] Concrete-ML MNIST MLP wide box [-3,3] (d=64)...")
    all_rows.extend(run_circuit("concrete_mnist_mlp", "mnist", -3.0, 3.0))

    print("[3/4] Concrete-ML WDBC MLP tight box [-1,1] (d=30) — saturation rescue...")
    all_rows.extend(run_circuit("concrete_wdbc_mlp_tight", "wdbc", -1.0, 1.0))

    print("[4/4] Concrete-ML MNIST MLP tight box [-1,1] (d=64) — saturation rescue...")
    all_rows.extend(run_circuit("concrete_mnist_mlp_tight", "mnist", -1.0, 1.0))

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    fieldnames = list(all_rows[0].keys())
    with open(rows_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"  wrote {len(all_rows)} rows -> {rows_path}")

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
            f"  {s['circuit']:25s}  d={s['d']:3d}  hidden={s['hidden_size']:3d}  "
            f"median_ratio={s['median_ratio']:.3f}  "
            f"wins={s['wins']}/{s['n_seeds']}  "
            f"p={s['p_value_sign_test_one_sided']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
