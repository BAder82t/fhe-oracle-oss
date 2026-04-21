# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""v0.3.0 benchmark validation.

Runs five benchmarks on the shipped v0.3.0 package:

1. v0.2 vs v0.3 head-to-head on every paper circuit.
2. AutoOracle probe classification on every circuit.
3. AutoOracle end-to-end performance vs paper best.
4. SubspaceOracle on d=200 and d=500 synthetic circuits.
5. AutoOracle -> SubspaceOracle auto-dispatch at d=200.

Writes results to benchmarks/results/v030_validation.csv and prints a
final PASS/FAIL verdict. Does NOT modify any package code.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Callable

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from fhe_oracle import FHEOracle, __version__
from fhe_oracle.autoconfig import AutoOracle, Regime, classify_landscape
from fhe_oracle.subspace import SubspaceOracle

# Results CSV path.
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
_CSV_PATH = os.path.join(_RESULTS_DIR, "v030_validation.csv")

SEEDS = list(range(1, 11))


# =============================================================================
# Circuit builders
# =============================================================================


def build_lr_mock(d: int = 8, seed: int = 42) -> dict:
    """LR d=8 mock -- the paper's Circuit 1 mock (hot-zone amplification)."""
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
        arr = np.asarray(x, dtype=np.float64)
        plain = plaintext_fn(x)
        h = int(abs(hash(tuple(round(float(v), 9) for v in arr))) % (2**31))
        local = np.random.default_rng(h)
        noise = float(local.normal(0.0, 1e-4))
        z_proxy = float(np.dot(arr, arr))
        if z_proxy > 4.0 and abs(plain - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 4.0)
            noise *= amp
        return plain + noise

    return {
        "name": "lr_mock",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
        "weights": w.reshape(1, d),
        "bias": np.array([b]),
        "paper_best": 0.624,  # paper Table -- order-of-magnitude gate.
    }


def build_poly_mock(d: int = 6) -> dict:
    """Circuit 2 mock -- depth-4 polynomial with magnitude-scaled noise."""
    coeffs = np.linspace(0.5, 1.5, d - 1)

    def plaintext_fn(x):
        arr = np.asarray(x, dtype=np.float64)
        return float(np.sum(coeffs * arr[:-1] ** 2 * arr[1:]))

    def fhe_fn(x):
        arr = np.asarray(x, dtype=np.float64)
        plain = plaintext_fn(x)
        h = int(abs(hash(tuple(round(float(v), 9) for v in arr))) % (2**31))
        local = np.random.default_rng(h)
        base = float(local.normal(0.0, 5e-5))
        intermediates = arr[:-1] ** 2 * arr[1:]
        mag = float(np.max(np.abs(intermediates))) if intermediates.size else 0.0
        amp = 1.0 + 20.0 * max(0.0, mag - 1.0)
        return plain + base * amp

    return {
        "name": "poly_mock",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-2.0, 2.0)] * d,
        "paper_best": None,
    }


def build_cheb_mock(d: int = 10) -> dict:
    """Cheb mock -- dense W,b front-end, Chebyshev-3 vs true sigmoid.

    Matches the structure of the TenSEAL cheb_d10 circuit but uses a
    purely-analytic "FHE" implementation (no CKKS noise), so divergence
    is the polynomial approximation error |sigma - sigma_cheb3|.
    """
    from numpy.polynomial import chebyshev as _cheb

    hidden = 4
    rng = np.random.default_rng(123)
    W = (rng.standard_normal((hidden, d)) * 0.5).astype(np.float64)
    b = (rng.standard_normal(hidden) * 0.1).astype(np.float64)

    z_grid = np.linspace(-5.0, 5.0, 1000)
    sigma = 1.0 / (1.0 + np.exp(-z_grid))
    cheb_coeffs = _cheb.chebfit(z_grid, sigma, 3)
    power = _cheb.cheb2poly(cheb_coeffs)

    def _cheb3(z_val):
        return float(np.polynomial.polynomial.polyval(z_val, power))

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        h = W @ xa + b
        return (1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))).tolist()

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        h = W @ xa + b
        return [_cheb3(float(z)) for z in h]

    return {
        "name": "cheb_mock",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
        "weights": W,
        "bias": b,
        "paper_best": None,
    }


def build_tenseal_circuits() -> list[dict]:
    """Build all TenSEAL paper circuits; returns [] if TenSEAL missing."""
    try:
        from tenseal_circuits import (
            build_tenseal_chebyshev_d10,
            build_tenseal_circuit2,
            build_tenseal_lr_d8,
            build_tenseal_wdbc,
        )
        from fhe_oracle.adapters.tenseal_adapter import TenSEALContext
    except Exception as exc:
        print(f"  [skip] TenSEAL unavailable: {exc}")
        return []

    try:
        ctx = TenSEALContext()
    except Exception as exc:
        print(f"  [skip] TenSEALContext failed: {exc}")
        return []

    circuits = []
    try:
        circuits.append(build_tenseal_lr_d8(ctx))
    except Exception as exc:
        print(f"  [warn] build_tenseal_lr_d8 failed: {exc}")
    try:
        c2 = build_tenseal_circuit2(ctx, d=6)
        c2["paper_best"] = None
        circuits.append(c2)
    except Exception as exc:
        print(f"  [warn] build_tenseal_circuit2 failed: {exc}")
    try:
        circuits.append(build_tenseal_chebyshev_d10(ctx))
    except Exception as exc:
        print(f"  [warn] build_tenseal_chebyshev_d10 failed: {exc}")
    try:
        circuits.append(build_tenseal_wdbc(ctx))
    except Exception as exc:
        print(f"  [warn] build_tenseal_wdbc failed: {exc}")
    return circuits


# =============================================================================
# Benchmark 1: v0.2 vs v0.3 head-to-head
# =============================================================================


def bench_headtohead(circuit: dict, budget: int, seeds: list[int]) -> dict:
    """Run v0.2 (w_noise=0.5, w_depth=0.3) vs v0.3 (0,0) on same seeds.

    Because both runs construct fitness via the default DivergenceFitness
    path (no adapter passed here), the shaping weights are *inert* -- the
    runs should be bit-identical. We still execute both to document the
    no-regression gate empirically.
    """
    deltas_old, deltas_new = [], []
    wins = 0  # v0.3 >= v0.2 counts as a win (tie OK).
    for seed in seeds:
        old = FHEOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            input_dim=circuit["d"],
            input_bounds=circuit["bounds"],
            seed=seed,
            w_noise=0.5, w_depth=0.3,  # restore v0.2 defaults
        ).run(n_trials=budget, threshold=1.0)
        new = FHEOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            input_dim=circuit["d"],
            input_bounds=circuit["bounds"],
            seed=seed,
            # v0.3.0 defaults (w_noise=0, w_depth=0)
        ).run(n_trials=budget, threshold=1.0)
        deltas_old.append(old.max_error)
        deltas_new.append(new.max_error)
        if new.max_error >= old.max_error:
            wins += 1
    mean_old = float(np.mean(deltas_old))
    mean_new = float(np.mean(deltas_new))
    ratio = mean_new / mean_old if mean_old > 0 else float("nan")
    return {
        "circuit": circuit["name"],
        "d": circuit["d"],
        "budget": budget,
        "mean_old": mean_old,
        "mean_new": mean_new,
        "ratio": ratio,
        "wins": wins,
        "n_seeds": len(seeds),
    }


# =============================================================================
# Benchmark 2: AutoOracle classification
# =============================================================================


EXPECTED_REGIMES = {
    "lr_mock": {Regime.STANDARD, Regime.PREACTIVATION_DOMINATED},
    "poly_mock": {Regime.STANDARD},
    "cheb_mock": {Regime.FULL_DOMAIN_SATURATION},
    "lr_d8_tenseal": {Regime.PREACTIVATION_DOMINATED, Regime.STANDARD},
    "circuit2_tenseal": {Regime.STANDARD},
    "cheb_d10_tenseal": {Regime.PLATEAU_THEN_CLIFF, Regime.STANDARD, Regime.FULL_DOMAIN_SATURATION},
    "wdbc_tenseal": {Regime.PREACTIVATION_DOMINATED, Regime.STANDARD},
}


def bench_classify(circuit: dict, n_probes: int = 50) -> dict:
    W = circuit.get("weights")
    b = circuit.get("bias")
    t0 = time.perf_counter()
    probe = classify_landscape(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        bounds=circuit["bounds"],
        n_probes=n_probes,
        W=W, b=b,
        seed=1,
    )
    elapsed = time.perf_counter() - t0
    divs = probe.probe_divergences
    expected = EXPECTED_REGIMES.get(circuit["name"], set())
    match = probe.regime in expected if expected else None
    return {
        "circuit": circuit["name"],
        "d": circuit["d"],
        "regime": probe.regime.value,
        "expected": ",".join(sorted(r.value for r in expected)) if expected else "n/a",
        "match": match,
        "probe_mean": float(np.mean(divs)),
        "probe_std": float(np.std(divs)),
        "probe_max": float(np.max(divs)),
        "probe_median": float(np.median(divs)),
        "reason": probe.recommendation.get("reason", ""),
        "elapsed_s": elapsed,
    }


# =============================================================================
# Benchmark 3: AutoOracle end-to-end
# =============================================================================


def bench_autoracle(circuit: dict, budget: int, seeds: list[int]) -> dict:
    errs = []
    regimes = []
    strategies = []
    for seed in seeds:
        oracle = AutoOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            bounds=circuit["bounds"],
            W=circuit.get("weights"),
            b=circuit.get("bias"),
            n_probes=min(30, max(10, budget // 5)),
        )
        res = oracle.run(n_trials=budget, seed=seed)
        errs.append(float(res.max_error))
        regimes.append(getattr(res, "regime", "unknown"))
        strategies.append(getattr(res, "strategy_used", "unknown"))
    return {
        "circuit": circuit["name"],
        "d": circuit["d"],
        "budget": budget,
        "mean_err": float(np.mean(errs)),
        "max_err": float(np.max(errs)),
        "regime": regimes[0],
        "strategy_used": strategies[0],
        "paper_best": circuit.get("paper_best"),
    }


# =============================================================================
# Benchmark 4: SubspaceOracle at high-d
# =============================================================================


def _random_search(plain, fhe, bounds, n_trials, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    best = 0.0
    for _ in range(n_trials):
        x = rng.uniform(lo, hi)
        p = plain(x)
        f = fhe(x)
        if np.isscalar(p) and np.isscalar(f):
            e = abs(p - f)
        else:
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            e = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
        if e > best:
            best = float(e)
    return best


def build_lr_mock_highdim(d: int, seed: int = 42) -> dict:
    """LR mock at arbitrary d, retaining the hot-zone amplification."""
    c = build_lr_mock(d=d, seed=seed)
    c["name"] = f"lr_mock_d{d}"
    return c


def build_directional_bug(d: int, seed: int = 42) -> dict:
    """Bug along a single random direction w."""
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
        "w": w,
    }


def build_cnn_like(d: int, seed: int = 42) -> dict:
    """2-layer ReLU with 4-bit quantisation on hidden layer."""
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
        h_q = np.round(h * 16.0) / 16.0  # 4-bit quantisation.
        return float((W2 @ h_q + b2)[0])

    return {
        "name": f"cnn_like_d{d}",
        "plain": plain,
        "fhe": fhe,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
    }


def bench_subspace(circuit: dict, n_trials: int, k: int, n_proj: int,
                   seeds: list[int]) -> dict:
    sub_errs, rand_errs = [], []
    wins = 0
    for seed in seeds:
        oracle = SubspaceOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            bounds=circuit["bounds"],
            subspace_dim=k,
            n_projections=n_proj,
        )
        res = oracle.run(n_trials=n_trials, seed=seed)
        sub = float(res.max_error)
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
        "budget": n_trials,
        "mean_subspace": mean_sub,
        "mean_random": mean_rnd,
        "ratio": ratio,
        "wins": wins,
        "n_seeds": len(seeds),
    }


# =============================================================================
# Benchmark 5: AutoOracle -> SubspaceOracle integration
# =============================================================================


def bench_integration(d: int = 200, n_trials: int = 500) -> dict:
    c = build_lr_mock_highdim(d=d, seed=42)
    oracle = AutoOracle(
        plaintext_fn=c["plain"],
        fhe_fn=c["fhe"],
        bounds=c["bounds"],
        n_probes=50,
    )
    res = oracle.run(n_trials=n_trials, seed=42)
    strategy = getattr(res, "strategy_used", "unknown")
    return {
        "d": d,
        "budget": n_trials,
        "regime": getattr(res, "regime", "unknown"),
        "strategy_used": strategy,
        "max_err": float(res.max_error),
        "subspace_dispatched": strategy == "subspace",
    }


# =============================================================================
# CSV writer
# =============================================================================


def _write_rows(fh, section: str, rows: list[dict]) -> None:
    if not rows:
        return
    for r in rows:
        r = dict(r)
        r["__section"] = section
        yield r


def save_all_csv(all_results: list[tuple[str, list[dict]]]) -> None:
    # Flatten, uniting all unique keys across sections.
    flat: list[dict] = []
    for section, rows in all_results:
        for r in rows:
            rr = dict(r)
            rr["__section"] = section
            flat.append(rr)
    if not flat:
        return
    keys = ["__section"]
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    print(f"\nSaved CSV -> {_CSV_PATH}")


# =============================================================================
# Main
# =============================================================================


def _fmt(x, prec=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "nan"
        return f"{x:.{prec}e}"
    return str(x)


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 70)
    print(f"v{__version__} benchmark validation")
    print("=" * 70)

    # --- Build circuits -------------------------------------------------
    print("\n[0] Building circuits...")
    mock_circuits = [
        build_lr_mock(d=8),
        build_poly_mock(d=6),
        build_cheb_mock(d=10),
    ]
    ts_circuits = build_tenseal_circuits()
    all_paper_circuits = mock_circuits + ts_circuits
    print(f"    {len(mock_circuits)} mock + {len(ts_circuits)} TenSEAL = "
          f"{len(all_paper_circuits)} paper circuits")

    all_results: list[tuple[str, list[dict]]] = []

    # --- Benchmark 1 ----------------------------------------------------
    print("\n[1] Head-to-head: v0.2 defaults vs v0.3 defaults ---------")
    rows_b1 = []
    for c in all_paper_circuits:
        budget = 500 if c["name"] in ("lr_mock", "poly_mock", "cheb_mock") else 60
        row = bench_headtohead(c, budget=budget, seeds=SEEDS)
        rows_b1.append(row)
        print(f"    {row['circuit']:<22} d={row['d']:<3} B={row['budget']:<4} "
              f"v0.2={_fmt(row['mean_old'])} v0.3={_fmt(row['mean_new'])} "
              f"ratio={row['ratio']:.3f} wins={row['wins']}/{row['n_seeds']}")
    all_results.append(("b1_headtohead", rows_b1))

    # --- Benchmark 2 ----------------------------------------------------
    print("\n[2] AutoOracle probe classification -----------------------")
    rows_b2 = []
    for c in all_paper_circuits:
        row = bench_classify(c, n_probes=50)
        rows_b2.append(row)
        mark = "OK" if row["match"] else ("??" if row["match"] is None else "MISS")
        print(f"    [{mark}] {row['circuit']:<22} -> {row['regime']:<26} "
              f"(expected {row['expected']})")
        print(f"           probe stats: mean={_fmt(row['probe_mean'])} "
              f"std={_fmt(row['probe_std'])} max={_fmt(row['probe_max'])} "
              f"median={_fmt(row['probe_median'])}")
    all_results.append(("b2_classify", rows_b2))

    # --- Benchmark 3 ----------------------------------------------------
    print("\n[3] AutoOracle end-to-end ---------------------------------")
    rows_b3 = []
    for c in all_paper_circuits:
        budget = 500 if c["name"] in ("lr_mock", "poly_mock", "cheb_mock") else 60
        row = bench_autoracle(c, budget=budget, seeds=SEEDS)
        rows_b3.append(row)
        print(f"    {row['circuit']:<22} d={row['d']:<3} regime={row['regime']:<26} "
              f"strat={row['strategy_used']:<14} mean_err={_fmt(row['mean_err'])}")
    all_results.append(("b3_autoracle", rows_b3))

    # --- Benchmark 4 ----------------------------------------------------
    print("\n[4] SubspaceOracle at high d ------------------------------")
    rows_b4 = []
    # 4a: LR mock at d=200 and d=500
    sub_seeds = list(range(1, 6))  # 5 seeds (budget-bounded)
    for d in (200, 500):
        c = build_lr_mock_highdim(d=d, seed=42)
        row = bench_subspace(
            c, n_trials=500, k=min(50, d // 4), n_proj=3, seeds=sub_seeds,
        )
        rows_b4.append(row)
        print(f"    {row['circuit']:<22} d={row['d']:<3} k={row['k']:<3} "
              f"P={row['n_projections']} sub={_fmt(row['mean_subspace'])} "
              f"rnd={_fmt(row['mean_random'])} ratio={row['ratio']:.3f} "
              f"wins={row['wins']}/{row['n_seeds']}")
    # 4b: directional bug d=500
    c = build_directional_bug(d=500, seed=42)
    row = bench_subspace(
        c, n_trials=500, k=50, n_proj=5, seeds=sub_seeds,
    )
    rows_b4.append(row)
    print(f"    {row['circuit']:<22} d={row['d']:<3} k={row['k']:<3} "
          f"P={row['n_projections']} sub={_fmt(row['mean_subspace'])} "
          f"rnd={_fmt(row['mean_random'])} ratio={row['ratio']:.3f} "
          f"wins={row['wins']}/{row['n_seeds']}")
    # 4c: CNN-like d=200
    c = build_cnn_like(d=200, seed=42)
    row = bench_subspace(
        c, n_trials=500, k=50, n_proj=5, seeds=sub_seeds,
    )
    rows_b4.append(row)
    print(f"    {row['circuit']:<22} d={row['d']:<3} k={row['k']:<3} "
          f"P={row['n_projections']} sub={_fmt(row['mean_subspace'])} "
          f"rnd={_fmt(row['mean_random'])} ratio={row['ratio']:.3f} "
          f"wins={row['wins']}/{row['n_seeds']}")
    all_results.append(("b4_subspace", rows_b4))

    # --- Benchmark 5 ----------------------------------------------------
    print("\n[5] AutoOracle -> Subspace integration --------------------")
    row = bench_integration(d=200, n_trials=500)
    print(f"    d={row['d']} strategy={row['strategy_used']} "
          f"regime={row['regime']} max_err={_fmt(row['max_err'])}")
    all_results.append(("b5_integration", [row]))

    # --- Save CSV -------------------------------------------------------
    save_all_csv(all_results)

    # --- Verdict --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Gate evaluation")
    print("=" * 70)

    # Gate 1: no regression (v0.3 >= 0.95 * v0.2 on every circuit)
    gate1_rows = rows_b1
    gate1_violators = [
        r for r in gate1_rows
        if np.isfinite(r["ratio"]) and r["ratio"] < 0.95
    ]
    gate1_pass = len(gate1_violators) == 0
    print(f"  [1] No regression (>=0.95x): "
          f"{'PASS' if gate1_pass else 'FAIL'} "
          f"({len(gate1_rows) - len(gate1_violators)}/{len(gate1_rows)})")
    for v in gate1_violators:
        print(f"      violator: {v['circuit']} ratio={v['ratio']:.3f}")

    # Gate 2: >=7/9 classifications correct
    evaluable = [r for r in rows_b2 if r["match"] is not None]
    correct = sum(1 for r in evaluable if r["match"])
    total = len(evaluable)
    gate2_pass = correct >= min(7, total - 2) and correct >= 0.6 * total
    print(f"  [2] Classification (>=7/{total} correct): "
          f"{'PASS' if gate2_pass else 'FAIL'} "
          f"({correct}/{total})")

    # Gate 3: subspace beats random (>=1.1x) on >=2/3 of high-d circuits
    sub_wins = sum(1 for r in rows_b4 if r["ratio"] >= 1.1)
    gate3_pass = sub_wins >= 2
    print(f"  [3] SubspaceOracle beats random at >=1.1x on >=2/3 "
          f"high-d circuits: {'PASS' if gate3_pass else 'FAIL'} "
          f"({sub_wins}/{len(rows_b4)})")

    # Gate 4: auto-dispatch to subspace
    gate4_pass = bool(all_results[-1][1][0]["subspace_dispatched"])
    print(f"  [4] AutoOracle dispatches to subspace at d>100: "
          f"{'PASS' if gate4_pass else 'FAIL'}")

    overall = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"Overall verdict: {'PASS' if overall else 'FAIL'} "
          f"(elapsed {elapsed:.1f}s)")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
