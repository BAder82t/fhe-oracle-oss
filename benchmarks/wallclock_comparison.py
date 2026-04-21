# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Wall-clock comparison: oracle vs random at varying eval costs.

Addresses Limitation 5 — eval count vs wall-clock.

Three regimes:
    1. Real FHE (TenSEAL): LR d=8, B=60, 3 seeds.
    2. Mock pure-Python: LR d=8 mock, B=500, 5 seeds.
    3. Parameterized cost sweep: mock + time.sleep(cost_per_eval),
       cost_per_eval ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1} s, B=500, 5 seeds.

Outputs:
    - benchmarks/results/wallclock_comparison.csv
    - printed break-even analysis

For each row we measure:
    - eval_cost_ms : per-evaluation cost in ms (measured or prescribed)
    - oracle_wallclock_s : total CMA-ES hybrid wall-clock
    - random_wallclock_s : total random-search wall-clock
    - total_evals   : B
    - overhead_pct  : (oracle_wallclock − total_evals * eval_cost_s)
                     / oracle_wallclock   (fraction NOT in eval)
    - wallclock_ratio : oracle_wallclock / random_wallclock

The "break-even" is the eval cost at which overhead_pct < 5%.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle import FHEOracle


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "wallclock_comparison.csv"
)


# ---------------------------------------------------------------------------
# Mock LR circuit — pure Python, negligible intrinsic cost
# ---------------------------------------------------------------------------

def _fit_lr_mock(d: int = 8, seed: int = 42) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(0.0, 1.0, size=(n, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    lr = 0.1
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        w -= lr * (X.T @ (p - y) / n)
        b -= lr * float(np.mean(p - y))
    return w, b


def build_mock_lr(d: int, sleep_s: float) -> dict:
    """Mock LR with Taylor-3 sigmoid. Optional sleep to inject eval cost."""
    w, b = _fit_lr_mock(d=d, seed=42)

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w, xa) + b)
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + np.exp(-z))

    def fhe_fn(x):
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w, xa) + b)
        return 0.5 + z * 0.25 - (z ** 3) * (1.0 / 48.0)

    return {
        "name": f"mock_lr_d{d}_sleep{sleep_s}",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
        "sleep_s": sleep_s,
    }


# ---------------------------------------------------------------------------
# Random-search baseline that mirrors oracle's random_floor in cost
# ---------------------------------------------------------------------------

def random_search_wallclock(
    plain_fn: Callable,
    fhe_fn: Callable,
    bounds: list[tuple[float, float]],
    budget: int,
    seed: int,
) -> tuple[float, float]:
    """Return (wallclock_s, max_error)."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best = 0.0
    t0 = time.perf_counter()
    for _ in range(budget):
        x = rng.uniform(lows, highs).tolist()
        p = plain_fn(x)
        f = fhe_fn(x)
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
        f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
        n = min(p_arr.size, f_arr.size)
        err = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
        if err > best:
            best = err
    return time.perf_counter() - t0, best


def oracle_wallclock(
    plain_fn: Callable,
    fhe_fn: Callable,
    bounds: list[tuple[float, float]],
    budget: int,
    seed: int,
) -> tuple[float, float, int]:
    """Return (wallclock_s, max_error, total_evals)."""
    oracle = FHEOracle(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_fn,
        input_dim=len(bounds),
        input_bounds=bounds,
        seed=seed,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=budget, threshold=0.0)
    wall = time.perf_counter() - t0
    return wall, float(res.max_error), int(res.n_trials)


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def run_regime(
    label: str,
    circuit: dict,
    budget: int,
    seeds: list[int],
    eval_cost_s: float | None = None,
) -> list[dict]:
    """Run oracle + random at matched budget over seeds. Return per-seed rows."""
    rows = []
    for s in seeds:
        o_wall, o_err, o_evals = oracle_wallclock(
            circuit["plain"], circuit["fhe"], circuit["bounds"], budget, s
        )
        r_wall, r_err = random_search_wallclock(
            circuit["plain"], circuit["fhe"], circuit["bounds"], budget, s
        )
        # Empirical per-eval cost uses random baseline (zero CMA-ES overhead).
        empirical_eval_cost_s = r_wall / budget if budget > 0 else 0.0
        # Overhead = extra oracle time above B evaluations at the empirical rate.
        overhead = max(o_wall - o_evals * empirical_eval_cost_s, 0.0)
        overhead_pct = overhead / o_wall if o_wall > 0 else 0.0
        rows.append({
            "regime": label,
            "circuit": circuit["name"],
            "seed": s,
            "budget": budget,
            "eval_cost_s": empirical_eval_cost_s,
            "eval_cost_ms": empirical_eval_cost_s * 1000.0,
            "oracle_wallclock_s": o_wall,
            "random_wallclock_s": r_wall,
            "oracle_total_evals": o_evals,
            "oracle_max_error": o_err,
            "random_max_error": r_err,
            "cma_overhead_s": overhead,
            "overhead_pct": overhead_pct,
            "wallclock_ratio": o_wall / r_wall if r_wall > 0 else float("inf"),
        })
    return rows


def _summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def med(k):
        return statistics.median(r[k] for r in rows)
    return {
        "regime": rows[0]["regime"],
        "circuit": rows[0]["circuit"],
        "n_seeds": len(rows),
        "budget": rows[0]["budget"],
        "eval_cost_ms": rows[0]["eval_cost_ms"],
        "median_oracle_wallclock_s": med("oracle_wallclock_s"),
        "median_random_wallclock_s": med("random_wallclock_s"),
        "median_overhead_pct": med("overhead_pct"),
        "median_wallclock_ratio": med("wallclock_ratio"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    all_rows: list[dict] = []

    # --- Regime 1: real TenSEAL FHE ---
    try:
        from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL
        if HAVE_TENSEAL:
            from tenseal_circuits import build_tenseal_lr_d8
            from fhe_oracle.adapters.tenseal_adapter import TenSEALContext
            ctx = TenSEALContext(seed=42)
            circ = build_tenseal_lr_d8(ctx)
            print(f"[regime=real_fhe_lr_d8] B=60 seeds=[0,1,2] …")
            rows = run_regime("real_fhe_lr_d8", circ, budget=60, seeds=[0, 1, 2])
            all_rows.extend(rows)
            print(f"  median overhead_pct = {_summarise(rows)['median_overhead_pct']:.3%}")
        else:
            print("[regime=real_fhe_lr_d8] TenSEAL not available — skipping")
    except Exception as exc:
        print(f"[regime=real_fhe_lr_d8] skipped: {exc}")

    # --- Regime 2: mock pure-Python ---
    circ = build_mock_lr(d=8, sleep_s=0.0)
    print(f"[regime=mock_lr_d8] B=500 seeds=[0..4] …")
    rows = run_regime("mock_lr_d8", circ, budget=500, seeds=list(range(5)))
    all_rows.extend(rows)
    print(f"  median overhead_pct = {_summarise(rows)['median_overhead_pct']:.3%}")

    # --- Regime 3: parametrised sleep sweep ---
    for cost_s in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        circ = build_mock_lr(d=8, sleep_s=cost_s)
        label = f"sweep_sleep_{cost_s:g}"
        # Predeclared eval_cost_s = measured overhead + sleep; use measurement
        budget = 500 if cost_s < 1e-2 else (200 if cost_s < 1e-1 else 60)
        seeds = list(range(3 if cost_s >= 1e-2 else 5))
        print(f"[regime={label}] B={budget} seeds={seeds} …")
        rows = run_regime(label, circ, budget=budget, seeds=seeds)
        all_rows.extend(rows)
        summ = _summarise(rows)
        print(
            f"  eval_cost={summ['eval_cost_ms']:.4f} ms  "
            f"overhead_pct={summ['median_overhead_pct']:.3%}  "
            f"wc_ratio={summ['median_wallclock_ratio']:.2f}"
        )

    # --- Emit CSV ---
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nWrote {len(all_rows)} rows → {RESULTS_PATH}")

    # --- Break-even analysis ---
    print("\n=== Break-even analysis (sweep, grouped by regime) ===")
    sweep_by_regime: dict[str, list[dict]] = {}
    for r in all_rows:
        if r["regime"].startswith("sweep_sleep_"):
            sweep_by_regime.setdefault(r["regime"], []).append(r)
    first_under_5 = True
    for regime in sorted(sweep_by_regime, key=lambda k: float(k.split("_")[-1])):
        rows = sweep_by_regime[regime]
        med_over = statistics.median(r["overhead_pct"] for r in rows)
        med_ratio = statistics.median(r["wallclock_ratio"] for r in rows)
        med_cost = statistics.median(r["eval_cost_ms"] for r in rows)
        marker = ""
        if med_over <= 0.05 and first_under_5:
            marker = "  <-- first ≤5% overhead"
            first_under_5 = False
        print(
            f"  regime={regime:22s}  "
            f"eval_cost={med_cost:10.4f} ms  "
            f"median_overhead={med_over:7.3%}  "
            f"wc_ratio={med_ratio:6.2f}{marker}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
