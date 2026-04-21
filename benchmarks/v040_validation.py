# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""v0.4.0 benchmark validation.

Four benchmarks on the shipped v0.4.0 package:

A1. Diversity injection on Chebyshev TenSEAL (4 configs x 20 seeds).
A2. Adaptive budget (early-stop, auto-extend, strategy-switch + head-
    to-head adaptive vs vanilla on every paper circuit).
A3. Multi-output fitness (synthetic 10-class circuit with controlled
    rank inversions; head-to-head MAX_ABSOLUTE vs RANK_INVERSION vs
    COMBINED).
A4. AutoOracle with all v0.4.0 features forwarded.

Writes results to benchmarks/results/v040_validation.csv and prints a
final PASS/FAIL verdict.
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
from fhe_oracle.autoconfig import AutoOracle
from fhe_oracle.multi_output import MultiOutputFitness, MultiOutputMode

_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
_CSV_PATH = os.path.join(_RESULTS_DIR, "v040_validation.csv")

SEEDS_FULL = list(range(1, 21))   # A1
SEEDS_SHORT = list(range(1, 11))  # A2/A4 (paper circuits)


# ============================================================
# Mock circuits (also used in v030_validation.py)
# ============================================================


def build_lr_mock(d: int = 8, seed: int = 42) -> dict:
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
        "name": "lr_mock", "plain": plaintext_fn, "fhe": fhe_fn,
        "d": d, "bounds": [(-3.0, 3.0)] * d,
        "weights": w.reshape(1, d), "bias": np.array([b]),
    }


def build_poly_mock(d: int = 6) -> dict:
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

    return {"name": "poly_mock", "plain": plaintext_fn, "fhe": fhe_fn,
            "d": d, "bounds": [(-2.0, 2.0)] * d}


def build_cheb_mock(d: int = 10) -> dict:
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

    return {"name": "cheb_mock", "plain": plaintext_fn, "fhe": fhe_fn,
            "d": d, "bounds": [(-3.0, 3.0)] * d, "weights": W, "bias": b}


def build_tenseal_circuits() -> list[dict]:
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
    out = []
    for fn in (build_tenseal_lr_d8,
               lambda c: build_tenseal_circuit2(c, d=6),
               build_tenseal_chebyshev_d10,
               build_tenseal_wdbc):
        try:
            out.append(fn(ctx))
        except Exception as exc:
            print(f"  [warn] {fn.__name__ if hasattr(fn,'__name__') else fn}: {exc}")
    return out


# ============================================================
# A1 -- Diversity injection on Chebyshev
# ============================================================


def _fail_count(circuit: dict, seeds: list[int], budget: int,
                threshold: float, **oracle_kwargs: Any) -> tuple[int, list[float]]:
    """Run oracle on circuit across seeds; return (n_fails, max_errs).

    A 'fail' is max_error >= threshold (the verdict the user cares
    about for benchmark scoring).
    """
    fails = 0
    errs: list[float] = []
    for seed in seeds:
        oracle = FHEOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            input_dim=circuit["d"],
            input_bounds=circuit["bounds"],
            seed=seed,
            **oracle_kwargs,
        )
        result = oracle.run(n_trials=budget, threshold=threshold)
        errs.append(float(result.max_error))
        if result.max_error >= threshold:
            fails += 1
    return fails, errs


def benchmark_a1_chebyshev_diversity(circuit: dict) -> list[dict]:
    """Four configs on cheb_d10_tenseal: baseline, warm-start,
    diversity-only, warm-start+diversity. Threshold = 0.20 (calibrated
    so the cliff samples count as fails)."""
    print("\n[A1] Diversity injection on Chebyshev TenSEAL ----------")
    threshold = 0.20
    budget = 60
    configs = [
        ("baseline", {}),
        ("warm_start", {"random_floor": 0.3, "warm_start": True}),
        ("diversity_only", {
            "diversity_injection": True,
            "inject_every": 5, "inject_count": 3,
        }),
        ("warm_start_diversity", {
            "random_floor": 0.3, "warm_start": True,
            "diversity_injection": True,
            "inject_every": 5, "inject_count": 3,
        }),
    ]
    rows = []
    for name, kwargs in configs:
        fails, errs = _fail_count(
            circuit, SEEDS_FULL, budget, threshold, **kwargs
        )
        row = {
            "config": name,
            "circuit": circuit["name"],
            "budget": budget,
            "threshold": threshold,
            "fails": fails,
            "n_seeds": len(SEEDS_FULL),
            "mean_err": float(np.mean(errs)),
            "max_err": float(np.max(errs)),
        }
        rows.append(row)
        print(f"    {name:<22} fails={fails}/{len(SEEDS_FULL)} "
              f"mean_err={row['mean_err']:.4f} max_err={row['max_err']:.4f}")
    return rows


def benchmark_a1_lr_diversity_no_regression(circuit: dict) -> list[dict]:
    """Diversity should not regress on the LR mock circuit."""
    print("\n[A1b] Diversity injection on LR mock (no-regression) ---")
    threshold = 0.10
    budget = 500
    configs = [
        ("baseline", {}),
        ("diversity", {
            "diversity_injection": True,
            "inject_every": 5, "inject_count": 3,
        }),
    ]
    rows = []
    for name, kwargs in configs:
        fails, errs = _fail_count(
            circuit, SEEDS_SHORT, budget, threshold, **kwargs
        )
        row = {
            "config": name,
            "circuit": circuit["name"],
            "budget": budget,
            "threshold": threshold,
            "fails": fails,
            "n_seeds": len(SEEDS_SHORT),
            "mean_err": float(np.mean(errs)),
            "max_err": float(np.max(errs)),
        }
        rows.append(row)
        print(f"    {name:<22} fails={fails}/{len(SEEDS_SHORT)} "
              f"mean_err={row['mean_err']:.4f}")
    return rows


# ============================================================
# A2 -- Adaptive budget
# ============================================================


def benchmark_a2a_early_stop() -> dict:
    print("\n[A2a] Early stop on immediate-FAIL circuit ------------")
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 100.0,
        input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4,
        adaptive=True,
        seed=1,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=500, threshold=0.01)
    elapsed = time.perf_counter() - t0
    print(f"    trials_used={res.n_trials}/500 "
          f"reason={res.adaptive_stop_reason} "
          f"max_err={res.max_error:.2e} elapsed={elapsed*1000:.1f}ms")
    return {
        "test": "a2a_early_stop",
        "trials_used": res.n_trials,
        "trials_budget": 500,
        "stop_reason": res.adaptive_stop_reason,
        "max_err": float(res.max_error),
        "elapsed_ms": elapsed * 1000.0,
    }


def benchmark_a2b_auto_extend(circuit: dict) -> dict:
    print("\n[A2b] Auto-extend on small budget ---------------------")
    oracle = FHEOracle(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        input_dim=circuit["d"],
        input_bounds=circuit["bounds"],
        adaptive=True,
        seed=1,
    )
    res = oracle.run(n_trials=30, threshold=1e9)
    print(f"    trials_used={res.n_trials} ext={res.adaptive_extensions_used} "
          f"max_err={res.max_error:.4f}")
    return {
        "test": "a2b_auto_extend",
        "circuit": circuit["name"],
        "trials_used": res.n_trials,
        "trials_budget": 30,
        "extensions": res.adaptive_extensions_used,
        "max_err": float(res.max_error),
    }


def benchmark_a2c_strategy_switch(circuit: dict) -> dict:
    print("\n[A2c] Strategy switch on Chebyshev (no warm-start) -----")
    switch_count = 0
    for seed in SEEDS_SHORT:
        oracle = FHEOracle(
            plaintext_fn=circuit["plain"],
            fhe_fn=circuit["fhe"],
            input_dim=circuit["d"],
            input_bounds=circuit["bounds"],
            adaptive=True,
            seed=seed,
        )
        res = oracle.run(n_trials=60, threshold=1e9)
        if res.adaptive_stop_reason == "stall_switch_to_random":
            switch_count += 1
    print(f"    seeds with switch: {switch_count}/{len(SEEDS_SHORT)}")
    return {
        "test": "a2c_strategy_switch",
        "circuit": circuit["name"],
        "switch_seeds": switch_count,
        "n_seeds": len(SEEDS_SHORT),
    }


def benchmark_a2d_head_to_head(circuit: dict, budget: int) -> dict:
    """Head-to-head adaptive vs vanilla on a circuit. Threshold huge
    so EARLY_STOP cannot fire and the comparison measures search
    quality only."""
    threshold = 1e9
    vanilla_errs, adaptive_errs = [], []
    vanilla_trials, adaptive_trials = [], []
    for seed in SEEDS_SHORT:
        v = FHEOracle(
            plaintext_fn=circuit["plain"], fhe_fn=circuit["fhe"],
            input_dim=circuit["d"], input_bounds=circuit["bounds"],
            seed=seed,
        ).run(n_trials=budget, threshold=threshold)
        a = FHEOracle(
            plaintext_fn=circuit["plain"], fhe_fn=circuit["fhe"],
            input_dim=circuit["d"], input_bounds=circuit["bounds"],
            adaptive=True, seed=seed,
        ).run(n_trials=budget, threshold=threshold)
        vanilla_errs.append(v.max_error)
        adaptive_errs.append(a.max_error)
        vanilla_trials.append(v.n_trials)
        adaptive_trials.append(a.n_trials)
    mean_v = float(np.mean(vanilla_errs))
    mean_a = float(np.mean(adaptive_errs))
    ratio = mean_a / mean_v if mean_v > 0 else float("inf")
    return {
        "circuit": circuit["name"],
        "budget": budget,
        "vanilla_mean": mean_v,
        "adaptive_mean": mean_a,
        "ratio": ratio,
        "vanilla_trials_mean": float(np.mean(vanilla_trials)),
        "adaptive_trials_mean": float(np.mean(adaptive_trials)),
    }


# ============================================================
# A3 -- Multi-output fitness
# ============================================================


def build_synthetic_multiclass(d: int = 8, k: int = 10) -> dict:
    """Synthetic k-class classifier where the FHE quantisation
    can flip the argmax in narrow regions of input space."""
    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 1.0, size=(k, d)) * 0.5
    b = rng.normal(0.0, 0.1, size=k)

    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def plain(x):
        z = W @ np.asarray(x) + b
        return softmax(z)

    def fhe(x):
        z = W @ np.asarray(x) + b
        # Coarse quantisation flips argmax near decision boundaries.
        z_q = np.round(z * 1.0) / 1.0
        return softmax(z_q)

    return {
        "name": "synthetic_multiclass",
        "plain": plain, "fhe": fhe,
        "d": d, "bounds": [(-3.0, 3.0)] * d, "k": k,
    }


def benchmark_a3_multioutput(circuit: dict) -> list[dict]:
    print("\n[A3] Multi-output fitness on synthetic 10-class --------")
    rows = []
    modes = [
        ("max_absolute", MultiOutputMode.MAX_ABSOLUTE),
        ("rank_inversion", MultiOutputMode.RANK_INVERSION),
        ("combined", MultiOutputMode.COMBINED),
    ]
    seeds = list(range(1, 11))
    for label, mode in modes:
        flips = 0
        max_abs_errs = []
        margins = []
        for seed in seeds:
            fitness = MultiOutputFitness(
                plaintext_fn=circuit["plain"],
                fhe_fn=circuit["fhe"],
                mode=mode,
                rank_weight=1.0,
            )
            oracle = FHEOracle(
                plaintext_fn=circuit["plain"],
                fhe_fn=circuit["fhe"],
                input_dim=circuit["d"],
                input_bounds=circuit["bounds"],
                fitness=fitness,
                seed=seed,
            )
            res = oracle.run(n_trials=200, threshold=1e9)
            x_star = np.asarray(res.worst_input)
            report = fitness.detailed_report(x_star)
            if report.get("decision_flipped"):
                flips += 1
            max_abs_errs.append(report["max_absolute_error"])
            if report.get("fhe_top2_margin") is not None:
                margins.append(report["fhe_top2_margin"])
        row = {
            "mode": label,
            "circuit": circuit["name"],
            "flips": flips,
            "n_seeds": len(seeds),
            "mean_max_abs_err": float(np.mean(max_abs_errs)),
            "mean_top2_margin": float(np.mean(margins)) if margins else None,
        }
        rows.append(row)
        print(f"    {label:<16} flips={flips}/{len(seeds)} "
              f"mean_abs_err={row['mean_max_abs_err']:.4f} "
              f"mean_margin={row['mean_top2_margin']!r}")
    return rows


# ============================================================
# A4 -- AutoOracle with all features
# ============================================================


def benchmark_a4_autoracle(circuit: dict, budget: int) -> dict:
    oracle = AutoOracle(
        plaintext_fn=circuit["plain"],
        fhe_fn=circuit["fhe"],
        bounds=circuit["bounds"],
        W=circuit.get("weights"),
        b=circuit.get("bias"),
        n_probes=min(50, max(10, budget // 5)),
        diversity_injection=True,
        inject_every=5,
        inject_count=3,
        adaptive=True,
    )
    res = oracle.run(n_trials=budget, seed=42)
    return {
        "circuit": circuit["name"],
        "d": circuit["d"],
        "budget": budget,
        "regime": getattr(res, "regime", "unknown"),
        "strategy_used": getattr(res, "strategy_used", "unknown"),
        "max_err": float(res.max_error),
        "n_trials": res.n_trials,
        "adaptive_stop_reason": getattr(res, "adaptive_stop_reason", None),
        "diversity_injections": getattr(res, "diversity_injections", 0),
    }


# ============================================================
# CSV writer / main
# ============================================================


def save_csv(all_rows: list[tuple[str, list[dict]]]) -> None:
    flat: list[dict] = []
    for section, rows in all_rows:
        for r in rows:
            rr = dict(r)
            rr["__section"] = section
            flat.append(rr)
    if not flat:
        return
    keys: list[str] = ["__section"]
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(_CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"\nSaved CSV -> {_CSV_PATH}")


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 70)
    print(f"v{__version__} validation")
    print("=" * 70)

    print("\n[0] Building circuits ...")
    mock = {
        "lr_mock": build_lr_mock(d=8),
        "poly_mock": build_poly_mock(d=6),
        "cheb_mock": build_cheb_mock(d=10),
    }
    ts = build_tenseal_circuits()
    ts_by_name = {c["name"]: c for c in ts}
    multi = build_synthetic_multiclass(d=8, k=10)
    print(f"    mock={list(mock)} tenseal={list(ts_by_name)}")

    all_rows: list[tuple[str, list[dict]]] = []

    # --- A1 -----------------------------------------------------
    if "cheb_d10_tenseal" in ts_by_name:
        rows_a1 = benchmark_a1_chebyshev_diversity(ts_by_name["cheb_d10_tenseal"])
        all_rows.append(("a1_chebyshev", rows_a1))
    else:
        print("\n[A1] Skipped: cheb_d10_tenseal unavailable")
        rows_a1 = []
    rows_a1b = benchmark_a1_lr_diversity_no_regression(mock["lr_mock"])
    all_rows.append(("a1_lr_no_regression", rows_a1b))

    # --- A2 -----------------------------------------------------
    rows_a2: list[dict] = [benchmark_a2a_early_stop()]
    if "lr_d8_tenseal" in ts_by_name:
        rows_a2.append(benchmark_a2b_auto_extend(ts_by_name["lr_d8_tenseal"]))
    if "cheb_d10_tenseal" in ts_by_name:
        rows_a2.append(benchmark_a2c_strategy_switch(ts_by_name["cheb_d10_tenseal"]))
    all_rows.append(("a2_adaptive_units", rows_a2))

    print("\n[A2d] Adaptive vs vanilla head-to-head ----------------")
    rows_a2d = []
    for c, budget in [
        (mock["lr_mock"], 500),
        (mock["poly_mock"], 500),
        (mock["cheb_mock"], 500),
    ]:
        row = benchmark_a2d_head_to_head(c, budget)
        rows_a2d.append(row)
        print(f"    {row['circuit']:<22} d={c['d']:<3} B={row['budget']:<4} "
              f"vanilla={row['vanilla_mean']:.3e} adaptive={row['adaptive_mean']:.3e} "
              f"ratio={row['ratio']:.3f}")
    for name in ("lr_d8_tenseal", "circuit2_tenseal", "cheb_d10_tenseal", "wdbc_tenseal"):
        if name in ts_by_name:
            row = benchmark_a2d_head_to_head(ts_by_name[name], budget=60)
            rows_a2d.append(row)
            print(f"    {row['circuit']:<22} B={row['budget']:<4} "
                  f"vanilla={row['vanilla_mean']:.3e} adaptive={row['adaptive_mean']:.3e} "
                  f"ratio={row['ratio']:.3f}")
    all_rows.append(("a2d_head_to_head", rows_a2d))

    # --- A3 -----------------------------------------------------
    rows_a3 = benchmark_a3_multioutput(multi)
    all_rows.append(("a3_multi_output", rows_a3))

    # --- A4 -----------------------------------------------------
    print("\n[A4] AutoOracle with all v0.4 features ----------------")
    rows_a4 = []
    for c, budget in [
        (mock["lr_mock"], 500),
        (mock["poly_mock"], 500),
        (mock["cheb_mock"], 500),
    ]:
        row = benchmark_a4_autoracle(c, budget)
        rows_a4.append(row)
        print(f"    {row['circuit']:<22} d={row['d']:<3} regime={row['regime']:<26} "
              f"strat={row['strategy_used']:<14} err={row['max_err']:.3e} "
              f"trials={row['n_trials']}")
    for name in ("lr_d8_tenseal", "circuit2_tenseal", "cheb_d10_tenseal", "wdbc_tenseal"):
        if name in ts_by_name:
            row = benchmark_a4_autoracle(ts_by_name[name], budget=60)
            rows_a4.append(row)
            print(f"    {row['circuit']:<22} d={row['d']:<3} regime={row['regime']:<26} "
                  f"strat={row['strategy_used']:<14} err={row['max_err']:.3e} "
                  f"trials={row['n_trials']}")
    all_rows.append(("a4_autoracle", rows_a4))

    save_csv(all_rows)

    # --- Verdict -----------------------------------------------
    print("\n" + "=" * 70)
    print("Gate evaluation")
    print("=" * 70)

    # A1 gates
    a1_lookup = {r["config"]: r for r in rows_a1}
    if a1_lookup:
        ws = a1_lookup.get("warm_start", {}).get("fails", -1)
        wsd = a1_lookup.get("warm_start_diversity", {}).get("fails", -1)
        do = a1_lookup.get("diversity_only", {}).get("fails", -1)
        gate_a1_combined = wsd >= 15
        gate_a1_diversity_only = do >= 8
        print(f"  [A1] warm_start_diversity >=15/20: "
              f"{'PASS' if gate_a1_combined else 'FAIL'} ({wsd}/20)")
        print(f"  [A1] diversity_only       >=8/20:  "
              f"{'PASS' if gate_a1_diversity_only else 'FAIL'} ({do}/20)")
    else:
        gate_a1_combined = gate_a1_diversity_only = False
        print("  [A1] SKIPPED (TenSEAL unavailable)")

    # A1b no-regression on LR
    base_lr = next((r for r in rows_a1b if r["config"] == "baseline"), None)
    div_lr = next((r for r in rows_a1b if r["config"] == "diversity"), None)
    if base_lr and div_lr:
        gate_a1b = div_lr["fails"] >= max(0, base_lr["fails"] - 1)
        print(f"  [A1b] LR diversity no regression: "
              f"{'PASS' if gate_a1b else 'FAIL'} "
              f"({div_lr['fails']} vs baseline {base_lr['fails']})")
    else:
        gate_a1b = False

    # A2d gates: ratio >= 0.95 on every circuit
    gate_a2d = all(r["ratio"] >= 0.95 for r in rows_a2d)
    print(f"  [A2d] adaptive ratio >=0.95 on all: "
          f"{'PASS' if gate_a2d else 'FAIL'}")

    # A3: rank_inversion finds >= max_absolute flips
    a3_lookup = {r["mode"]: r for r in rows_a3}
    rank_flips = a3_lookup.get("rank_inversion", {}).get("flips", -1)
    abs_flips = a3_lookup.get("max_absolute", {}).get("flips", -1)
    gate_a3 = rank_flips >= abs_flips
    print(f"  [A3] rank_inversion finds >= max_absolute flips: "
          f"{'PASS' if gate_a3 else 'FAIL'} "
          f"({rank_flips} vs {abs_flips})")

    overall = (
        (gate_a1_combined or not rows_a1)
        and (gate_a1_diversity_only or not rows_a1)
        and gate_a1b
        and gate_a2d
        and gate_a3
    )
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"Overall verdict: {'PASS' if overall else 'FAIL'} "
          f"(elapsed {elapsed:.1f}s)")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
