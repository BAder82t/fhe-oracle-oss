# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Real-CKKS preactivation search at d=200 and d=784 (Proposal 1c).

Builds a d-dim LR with Taylor-3 sigmoid under TenSEAL CKKS (N=16384,
4-level chain). Compares:

1. ``random_matched``  uniform random in d at B=50 (matched budget)
2. ``random_full``     uniform random in d at B=50d (50x larger budget)
3. ``preact_k1``       preactivation k=1 search at B=50

For LR + Taylor-3 the divergence δ(x) = |σ(z) - σ_T3(z)| depends on x
only through z = W·x + b. Preactivation search at B=50 in 1-D should
beat random at B=50d in d-D.

Per-eval cost on a laptop: ~50-200 ms at d=200 (3 ct-ct mults), so
B=50 is ~5-10s/seed. At d=784 the dot product needs more rotations
(O(log d)) and per-eval may grow to 0.5-2s, so B=50 is ~25-100s/seed.

Falls back gracefully to d=200 if d=784 TenSEAL fails (rotation key
size, scale overflow, etc.). Always reports d=200 as the primary
result.

Output: benchmarks/results/preactivation_realckks_d{D}.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle.adapters.tenseal_adapter import (
    HAVE_TENSEAL,
    TenSEALContext,
    make_tenseal_taylor3_fhe_fn,
)
from fhe_oracle.preactivation import PreactivationOracle


SEEDS = list(range(1, 11))


def _train_lr(d: int, seed: int = 42) -> tuple[np.ndarray, float]:
    """Same training loop as benchmarks/highdim_sweep.make_lr_mock."""
    rng = np.random.default_rng(seed)
    n_samples = max(200, 20 * d)
    X = rng.normal(0.0, 1.0, size=(n_samples, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b
    return w.astype(np.float64), float(b)


def _build_circuit(d: int):
    """Real-CKKS d-dim LR with Taylor-3 sigmoid."""
    w, b = _train_lr(d)
    ctx = TenSEALContext(seed=42)
    fhe_fn = make_tenseal_taylor3_fhe_fn(w, b, ctx)

    def plain(x):
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w, xa) + b)
        z = float(np.clip(z, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-z))

    return {"w": w, "b": b, "plain": plain, "fhe": fhe_fn,
            "ctx": ctx, "d": d, "bounds": [(-3.0, 3.0)] * d}


def _run_random(circuit, budget: int, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in circuit["bounds"]])
    highs = np.array([hi for _, hi in circuit["bounds"]])
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    t0 = time.perf_counter()
    best = 0.0
    n = 0
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            p = float(plain(x))
            f = float(fhe(x))
        except Exception:
            continue
        n += 1
        err = abs(p - f)
        if err > best:
            best = err
    return best, time.perf_counter() - t0, n


def _probe_eval_cost(circuit) -> float:
    """One TenSEAL eval timing (warmup-amortised)."""
    rng = np.random.default_rng(0)
    lows = np.array([lo for lo, _ in circuit["bounds"]])
    highs = np.array([hi for _, hi in circuit["bounds"]])
    x = rng.uniform(lows, highs)
    circuit["fhe"](x)  # warmup
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        circuit["fhe"](x)
    return (time.perf_counter() - t0) / n


def _run_one(d: int, out_dir: str, budget_pre: int = 50,
             budget_full_factor: int = 50) -> dict:
    out_path = os.path.join(out_dir, f"preactivation_realckks_d{d}.csv")
    print(f"\n=== Real-CKKS preactivation search at d={d} ===")
    print(f"  Output: {out_path}")
    print("Building TenSEAL circuit ...")
    t0 = time.perf_counter()
    circuit = _build_circuit(d)
    print(f"  Circuit built in {time.perf_counter() - t0:.2f}s")
    per_eval = _probe_eval_cost(circuit)
    print(f"  Per-eval cost (avg of 5): {per_eval * 1000:.1f} ms")
    full_budget = budget_full_factor * d
    print(
        f"  Budgets : preact={budget_pre}, "
        f"random_matched={budget_pre}, random_full={full_budget} "
        f"(estimated full random wall: {per_eval * full_budget:.1f} s/seed)"
    )

    rows: list[dict] = []
    t_start = time.perf_counter()

    print(f"\n[1/3] random matched B={budget_pre} d={d}")
    for seed in SEEDS:
        err, wall, n = _run_random(circuit, budget_pre, seed)
        rows.append({"d": d, "config": "random_matched", "seed": seed,
                     "max_error": err, "wall_clock_s": wall, "n_trials": n})
        print(f"  seed={seed} err={err:.4e} wall={wall:.2f}s n={n}")

    print(f"\n[2/3] preact k=1 B={budget_pre}")
    pre = PreactivationOracle(
        W=circuit["w"].reshape(1, -1), b=np.array([circuit["b"]]),
        plaintext_fn=circuit["plain"], fhe_fn=circuit["fhe"],
        input_bounds=circuit["bounds"],
        clip_penalty=0.05,
    )
    pre_results = pre.run(budget=budget_pre, seeds=SEEDS)
    for r in pre_results:
        rows.append({"d": d, "config": "preact_k1", "seed": r.seed,
                     "max_error": r.max_error,
                     "wall_clock_s": r.elapsed_seconds, "n_trials": r.n_trials})
        print(f"  seed={r.seed} err={r.max_error:.4e} wall={r.elapsed_seconds:.2f}s")

    # Run random_full with a tighter seed budget (3 seeds) and a cap
    # on per-seed budget at 5000 evals to keep d=784 wall-clock under
    # one hour. Even with a much larger budget than preact, random
    # should not match preact's advantage if the divergence is a pure
    # function of preactivation z = W·x + b.
    full_budget_capped = min(full_budget, 5000)
    full_seeds = SEEDS[:3]
    print(
        f"\n[3/3] random full B={full_budget_capped} "
        f"(unc apped 50d={full_budget}, capped to 5000) seeds={full_seeds}"
    )
    for seed in full_seeds:
        err, wall, n = _run_random(circuit, full_budget_capped, seed)
        rows.append({"d": d, "config": "random_full", "seed": seed,
                     "max_error": err, "wall_clock_s": wall, "n_trials": n})
        print(f"  seed={seed} err={err:.4e} wall={wall:.1f}s n={n}")

    elapsed = time.perf_counter() - t_start
    print(f"\nWall-clock for d={d}: {elapsed:.1f}s")

    fieldnames = ["d", "config", "seed", "max_error", "wall_clock_s", "n_trials"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")

    def med(cfg):
        vals = [r["max_error"] for r in rows if r["config"] == cfg]
        return float(np.median(vals)) if vals else 0.0

    summary = {
        "d": d,
        "median_random_matched": med("random_matched"),
        "median_preact_k1": med("preact_k1"),
        "median_random_full": med("random_full"),
        "elapsed_s": elapsed,
        "per_eval_ms": per_eval * 1000.0,
    }
    summary["preact_over_random_matched"] = (
        summary["median_preact_k1"] / max(1e-30, summary["median_random_matched"])
    )
    summary["preact_over_random_full"] = (
        summary["median_preact_k1"] / max(1e-30, summary["median_random_full"])
    )
    print(f"\n  median random_matched = {summary['median_random_matched']:.4e}")
    print(f"  median preact_k1      = {summary['median_preact_k1']:.4e}")
    print(f"  median random_full    = {summary['median_random_full']:.4e}")
    print(f"  preact / random_matched = {summary['preact_over_random_matched']:.3f}x")
    print(f"  preact / random_full    = {summary['preact_over_random_full']:.3f}x")
    return summary


def main() -> int:
    if not HAVE_TENSEAL:
        print("ERROR: TenSEAL not installed. Skipping.")
        return 0

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    summaries: list[dict] = []

    # Always run d=200 first (the conservative claim).
    try:
        summaries.append(_run_one(200, out_dir))
    except Exception as exc:
        print(f"d=200 FAILED: {exc}")

    # Try d=784. If it fails (rotation key memory, scale overflow, etc.),
    # log and continue.
    try:
        summaries.append(_run_one(784, out_dir))
    except Exception as exc:
        print(f"\nd=784 FAILED: {exc}")
        print("Falling back to d=200 only.")

    summary_path = os.path.join(out_dir, "preactivation_realckks_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        if summaries:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
