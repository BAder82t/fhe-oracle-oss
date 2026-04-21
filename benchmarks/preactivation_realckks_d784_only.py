# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Slim d=784 real-CKKS preactivation k=1 test (Proposal 1c, d=784 only).

Skips the random_full baseline (which is hours-long at d=784) and
runs only:
  - random_matched at B=50 (10 seeds) for the headline ratio
  - preact_k1 at B=50 (10 seeds) for the primary deliverable

If d=784 TenSEAL evaluation is too slow (>1s/eval), report and stop.

Output: benchmarks/results/preactivation_realckks_d784.csv
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
from preactivation_realckks_d784 import _train_lr, _run_random


SEEDS = list(range(1, 11))


def main() -> int:
    if not HAVE_TENSEAL:
        print("ERROR: TenSEAL not installed.")
        return 1

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preactivation_realckks_d784.csv")

    d = 784
    print(f"Slim d={d} real-CKKS preactivation k=1 test")
    print("=" * 70)

    print("Building TenSEAL context + LR weights ...", flush=True)
    t0 = time.perf_counter()
    ctx = TenSEALContext(seed=42)
    w, b = _train_lr(d)
    fhe = make_tenseal_taylor3_fhe_fn(w, b, ctx)
    def plain(x):
        z = float(np.dot(w, x) + b)
        z = float(np.clip(z, -500, 500))
        return 1.0 / (1.0 + np.exp(-z))
    bounds = [(-3.0, 3.0)] * d
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    # Per-eval cost probe.
    rng = np.random.default_rng(0)
    x_probe = rng.uniform(-3, 3, size=d)
    fhe(x_probe)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        fhe(x_probe)
    per_eval = (time.perf_counter() - t0) / 3
    print(f"  per-eval cost: {per_eval * 1000:.1f}ms", flush=True)
    if per_eval > 5.0:
        print("  per-eval >5s — d=784 TenSEAL too slow; stopping.", flush=True)
        return 0

    rows: list[dict] = []
    circuit = {"d": d, "plain": plain, "fhe": fhe, "bounds": bounds}

    print(f"\n[1/2] random matched B=50 d={d}", flush=True)
    for seed in SEEDS:
        err, wall, n = _run_random(circuit, 50, seed)
        rows.append({"d": d, "config": "random_matched", "seed": seed,
                     "max_error": err, "wall_clock_s": wall, "n_trials": n})
        print(f"  seed={seed} err={err:.4e} wall={wall:.2f}s", flush=True)

    print(f"\n[2/2] preact k=1 B=50", flush=True)
    pre = PreactivationOracle(
        W=w.reshape(1, -1), b=np.array([b]),
        plaintext_fn=plain, fhe_fn=fhe,
        input_bounds=bounds,
        clip_penalty=0.05,
    )
    pre_results = pre.run(budget=50, seeds=SEEDS)
    for r in pre_results:
        rows.append({"d": d, "config": "preact_k1", "seed": r.seed,
                     "max_error": r.max_error,
                     "wall_clock_s": r.elapsed_seconds, "n_trials": r.n_trials})
        print(f"  seed={r.seed} err={r.max_error:.4e} wall={r.elapsed_seconds:.2f}s", flush=True)

    fieldnames = ["d", "config", "seed", "max_error", "wall_clock_s", "n_trials"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)

    def med(cfg):
        return float(np.median([r["max_error"] for r in rows if r["config"] == cfg]))

    rm = med("random_matched")
    pk = med("preact_k1")
    print(f"\n  median random_matched B=50 : {rm:.4e}")
    print(f"  median preact_k1 B=50      : {pk:.4e}")
    print(f"  preact / random_matched    : {pk / max(1e-30, rm):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
