# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""v0.3.1 regression benchmark.

Reruns the four high-d circuits from the v0.3.0 Benchmark 4 with the
fixed SubspaceOracle (ball-radius z-bounds + multi-anchor + clip
penalty + random fallback) and reports before/after ratios.

Also reruns the AutoOracle plateau-cliff classification on a Chebyshev-
like simulated probe to confirm the relaxed thresholds catch the v0.3.0
miss without false positives elsewhere.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Callable

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from fhe_oracle import __version__
from fhe_oracle.autoconfig import Regime, classify_landscape
from fhe_oracle.subspace import SubspaceOracle

_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
_CSV_PATH = os.path.join(_RESULTS_DIR, "v031_regression.csv")

SEEDS = list(range(1, 6))


def _random_search(plain: Callable, fhe: Callable, bounds, n_trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    best = 0.0
    for _ in range(n_trials):
        x = rng.uniform(lo, hi)
        p = plain(x)
        f = fhe(x)
        if np.isscalar(p) and np.isscalar(f):
            e = abs(float(p) - float(f))
        else:
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            e = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
        if e > best:
            best = float(e)
    return best


def build_lr_mock_highdim(d: int, seed: int = 42) -> dict:
    """LR mock at arbitrary d with hot-zone amplification (matches
    benchmarks/v030_validation.py)."""
    rng = np.random.default_rng(seed)
    n_samples = 200
    X = rng.normal(0.0, 1.0, size=(n_samples, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    lr = 0.1
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w, xa) + b)
        return 1.0 / (1.0 + np.exp(-z))

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        plain = plaintext_fn(xa)
        h = int(abs(hash(tuple(round(float(v), 9) for v in xa))) % (2**31))
        local = np.random.default_rng(h)
        noise = float(local.normal(0.0, 1e-4))
        z_proxy = float(np.dot(xa, xa))
        if z_proxy > 4.0 and abs(plain - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 4.0)
            noise *= amp
        return plain + noise

    return {
        "name": f"lr_mock_d{d}",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


def build_directional_bug(d: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)

    def plain(x):
        return 0.0

    def fhe(x):
        z = float(w @ np.asarray(x))
        return abs(z) ** 3 / 48.0

    return {
        "name": f"directional_d{d}",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


def build_cnn_like(d: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((20, d)) * 0.1
    b1 = np.zeros(20)
    W2 = rng.standard_normal((1, 20)) * 0.1
    b2 = np.zeros(1)

    def plain(x):
        xa = np.asarray(x, dtype=np.float64)
        h = np.maximum(0.0, W1 @ xa + b1)
        return float((W2 @ h + b2)[0])

    def fhe(x):
        xa = np.asarray(x, dtype=np.float64)
        h = np.maximum(0.0, W1 @ xa + b1)
        h_q = np.round(h * 16.0) / 16.0
        return float((W2 @ h_q + b2)[0])

    return {
        "name": f"cnn_like_d{d}",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


def bench_subspace(circuit: dict, n_trials: int, k: int, n_proj: int,
                   n_anchors: int, seeds: list[int]) -> dict:
    sub_errs, rand_errs = [], []
    sub_times = []
    fallback_count = 0
    wins = 0
    for seed in seeds:
        oracle = SubspaceOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            bounds=circuit["bounds"],
            subspace_dim=k,
            n_projections=n_proj,
            n_anchors=n_anchors,
            clip_penalty=0.1,
        )
        t0 = time.perf_counter()
        res = oracle.run(n_trials=n_trials, seed=seed)
        sub_times.append(time.perf_counter() - t0)
        sub = float(res.max_error)
        if getattr(res, "fallback_taken", False):
            fallback_count += 1
        rnd = _random_search(
            circuit["plain"], circuit["fhe"], circuit["bounds"], n_trials, seed
        )
        sub_errs.append(sub)
        rand_errs.append(rnd)
        if sub > rnd:
            wins += 1
    mean_sub = float(np.mean(sub_errs))
    mean_rnd = float(np.mean(rand_errs))
    ratio = mean_sub / mean_rnd if mean_rnd > 0 else float("inf")
    return {
        "circuit": circuit["name"],
        "d": circuit["d"],
        "k": k,
        "n_projections": n_proj,
        "n_anchors": n_anchors,
        "budget": n_trials,
        "mean_subspace": mean_sub,
        "mean_random": mean_rnd,
        "ratio": ratio,
        "wins": wins,
        "n_seeds": len(seeds),
        "mean_wallclock_s": float(np.mean(sub_times)),
        "fallback_count": fallback_count,
    }


def _fmt(x, prec=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "nan"
        return f"{x:.{prec}e}"
    return str(x)


# v0.3.0 baseline ratios from research/release/v030-benchmark-report.md
V030_RATIOS = {
    "lr_mock_d200": 0.294,
    "lr_mock_d500": 0.128,
    "directional_d500": 0.133,
    "cnn_like_d200": 1.117,
}


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 70)
    print(f"v{__version__} subspace+plateau regression")
    print("=" * 70)

    rows: list[dict] = []

    print("\n[A] SubspaceOracle Benchmark 4 rerun ----------------------")
    circuits_specs = [
        (build_lr_mock_highdim(d=200), 50, 3),
        (build_lr_mock_highdim(d=500), 50, 3),
        (build_directional_bug(d=500), 50, 5),
        (build_cnn_like(d=200), 50, 5),
    ]
    for circuit, k, n_proj in circuits_specs:
        row = bench_subspace(
            circuit, n_trials=500, k=k, n_proj=n_proj,
            n_anchors=2, seeds=SEEDS,
        )
        v030 = V030_RATIOS.get(circuit["name"], float("nan"))
        row["v030_ratio"] = v030
        row["fixed"] = (
            "yes"
            if (row["ratio"] >= 1.0 and v030 < 1.0)
            else ("regress" if (row["ratio"] < v030 * 0.9) else "no-change")
        )
        rows.append(row)
        print(
            f"    {row['circuit']:<22} d={row['d']:<3} k={row['k']:<3} "
            f"P={row['n_projections']} A={row['n_anchors']} "
            f"sub={_fmt(row['mean_subspace'])} rnd={_fmt(row['mean_random'])} "
            f"ratio={row['ratio']:.3f} (v0.3.0={v030:.3f}) "
            f"wins={row['wins']}/{row['n_seeds']} "
            f"wall={row['mean_wallclock_s']:.1f}s "
            f"fb={row['fallback_count']}"
        )

    print("\n[B] AutoOracle plateau detection rerun --------------------")
    # Simulated Chebyshev TenSEAL probe stats from v0.3.0 report:
    # mean=0.098, std=0.041, max=0.363, median=0.098
    # 47/50 plateau samples ~0.098, 3/50 cliff samples 0.25-0.40.
    sim_state = {"i": 0}
    rng = np.random.RandomState(42)
    plateau = rng.normal(0.098, 0.012, 47)
    cliff = rng.uniform(0.25, 0.40, 3)
    pool = np.concatenate([plateau, cliff])
    rng.shuffle(pool)

    def fhe_chebyshev_like(x):
        i = sim_state["i"] % pool.size
        sim_state["i"] += 1
        return float(pool[i])

    res = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_chebyshev_like,
        bounds=[(-3.0, 3.0)] * 10,
        n_probes=50,
        seed=0,
    )
    print(f"    Chebyshev-like probe -> {res.regime.value}")
    print(f"    reason: {res.recommendation.get('reason', '')}")
    rows.append({
        "circuit": "cheb_simulated_probe",
        "d": 10,
        "regime": res.regime.value,
        "expected": Regime.PLATEAU_THEN_CLIFF.value,
        "match": res.regime == Regime.PLATEAU_THEN_CLIFF,
    })

    # Also confirm STANDARD landscapes still classify correctly.
    sim_state["i"] = 0
    res_std = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.001 * float(np.sum(np.asarray(x) ** 2)),
        bounds=[(-3.0, 3.0)] * 8,
        n_probes=50,
        seed=1,
    )
    print(f"    LR-like quadratic probe -> {res_std.regime.value} (expect STANDARD)")
    rows.append({
        "circuit": "lr_like_simulated_probe",
        "d": 8,
        "regime": res_std.regime.value,
        "expected": Regime.STANDARD.value,
        "match": res_std.regime == Regime.STANDARD,
    })

    # Save CSV
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV -> {_CSV_PATH}")

    # Gate: subspace beats random on >=3/4 circuits AND no regression
    # on cnn_like_d200.
    sub_rows = [r for r in rows if "ratio" in r and "circuit" in r and "cheb" not in r["circuit"] and "lr_like" not in r["circuit"]]
    wins = sum(1 for r in sub_rows if r["ratio"] >= 1.0)
    cnn_row = next((r for r in sub_rows if r["circuit"] == "cnn_like_d200"), None)
    cnn_ok = cnn_row is not None and cnn_row["ratio"] >= 1.0
    plateau_ok = res.regime == Regime.PLATEAU_THEN_CLIFF
    standard_ok = res_std.regime == Regime.STANDARD

    print("\n" + "=" * 70)
    print("Gate evaluation")
    print("=" * 70)
    print(f"  [A1] >=3/4 high-d circuits beat random (ratio >= 1.0): "
          f"{'PASS' if wins >= 3 else 'FAIL'} ({wins}/4)")
    print(f"  [A2] cnn_like_d200 no regression (ratio >= 1.0): "
          f"{'PASS' if cnn_ok else 'FAIL'}")
    print(f"  [B1] Chebyshev-like simulated probe -> PLATEAU: "
          f"{'PASS' if plateau_ok else 'FAIL'}")
    print(f"  [B2] STANDARD landscape stays STANDARD: "
          f"{'PASS' if standard_ok else 'FAIL'}")

    overall = (wins >= 3) and cnn_ok and plateau_ok and standard_ok
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"Overall verdict: {'PASS' if overall else 'FAIL'} "
          f"(elapsed {elapsed:.1f}s)")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
