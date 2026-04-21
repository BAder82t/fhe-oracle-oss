# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Preactivation-search validation at d=128 on the LR mock (Proposal 1b).

Compares three search strategies on the existing make_lr_mock(d=128)
circuit:

1. ``random``     uniform random in d=128 at B=50 (cheap baseline)
2. ``random_full`` uniform random in d=128 at B=2000 (full-budget baseline,
                   matching the existing highdim_sweep.csv ratios)
3. ``preact_k1``  PreactivationOracle k=1 search at B=50

The headline claim Proposal 1 must validate: preact_k1 at B=50 should
match or beat full-d random at B=2000 on the median-error metric. If
yes, the "search dim = preactivation rank, not input dim" story holds
at d=128 and we have empirical license to extend to d=784.

Output: benchmarks/results/preactivation_d128_validation.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle.preactivation import PreactivationOracle
from highdim_sweep import make_lr_mock


SEEDS = list(range(1, 11))
D = 128


def _run_random(circuit, budget: int, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in circuit["bounds"]])
    highs = np.array([hi for _, hi in circuit["bounds"]])
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    t0 = time.perf_counter()
    best = 0.0
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            err = abs(float(plain(x)) - float(fhe(x)))
        except Exception:
            continue
        if err > best:
            best = err
    return best, time.perf_counter() - t0, budget


def _extract_lr_weights(circuit, d: int):
    """Recover the trained (W, b) used inside make_lr_mock by replaying
    the same training loop. ``make_lr_mock`` does not expose them, so
    we mirror the loop here (rng_seed=42 by default in highdim_sweep).
    """
    rng = np.random.default_rng(42)
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
    return w.copy(), float(b)


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preactivation_d128_validation.csv")

    print("Preactivation d=128 validation (Proposal 1b)")
    print("=" * 70)
    circuit = make_lr_mock(D)
    w, b = _extract_lr_weights(circuit, D)
    print(f"  Recovered W shape={w.shape}, b={b:+.4f}")

    rows: list[dict] = []
    t_start = time.perf_counter()

    # Baseline 1: random at matched-low budget B=50
    print("\n[1/3] random (B=50, d=128)")
    for seed in SEEDS:
        err, wall, n = _run_random(circuit, budget=50, seed=seed)
        rows.append({"config": "random_b50", "seed": seed,
                     "max_error": err, "wall_clock_s": wall, "n_trials": n})
        print(f"  seed={seed} err={err:.4e} wall={wall:.2f}s")

    # Baseline 2: random at full B=2000 (matches existing curve)
    print("\n[2/3] random (B=2000, d=128)")
    for seed in SEEDS:
        err, wall, n = _run_random(circuit, budget=2000, seed=seed)
        rows.append({"config": "random_b2000", "seed": seed,
                     "max_error": err, "wall_clock_s": wall, "n_trials": n})
        print(f"  seed={seed} err={err:.4e} wall={wall:.2f}s")

    # Preactivation k=1 search at B=50
    print("\n[3/3] preactivation k=1 search (B=50)")
    pre = PreactivationOracle(
        W=w.reshape(1, -1), b=np.array([b]),
        plaintext_fn=circuit["plain"], fhe_fn=circuit["fhe"],
        input_bounds=circuit["bounds"],
        clip_penalty=0.05,
    )
    pre_results = pre.run(budget=50, seeds=SEEDS)
    for r in pre_results:
        rows.append({"config": "preact_k1_b50", "seed": r.seed,
                     "max_error": r.max_error,
                     "wall_clock_s": r.elapsed_seconds, "n_trials": r.n_trials})
        print(
            f"  seed={r.seed} err={r.max_error:.4e} "
            f"wall={r.elapsed_seconds:.2f}s clip={r.clip_distance:.3e}"
        )

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total wall-clock: {elapsed:.1f}s")

    fieldnames = ["config", "seed", "max_error", "wall_clock_s", "n_trials"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary
    def _med(cfg: str) -> float:
        return float(np.median([
            r["max_error"] for r in rows if r["config"] == cfg
        ]))

    m_rand_lo = _med("random_b50")
    m_rand_hi = _med("random_b2000")
    m_preact = _med("preact_k1_b50")

    print("\n--- Summary ---")
    print(f"  median random_b50    : {m_rand_lo:.4e}")
    print(f"  median random_b2000  : {m_rand_hi:.4e}")
    print(f"  median preact_k1_b50 : {m_preact:.4e}")
    print(f"  preact / random_b50  : {m_preact / m_rand_lo:.3f}x  (matched budget)")
    print(f"  preact / random_b2000: {m_preact / m_rand_hi:.3f}x  (40x cheaper budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
