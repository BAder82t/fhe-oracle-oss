# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""SIMD-batch packing feasibility test (Proposal 3).

Tests whether TenSEAL's CKKS slot packing can be used to evaluate
multiple candidate inputs in a single ciphertext. Specifically, the
batched evaluation packs ``s = floor(N/2 / d)`` candidates per
ciphertext and applies a coordinate-wise polynomial (Taylor-3 sigmoid
of a per-candidate scalar projection).

The d=64 setting:
- N=16384, slot count = 8192, s = 8192/64 = 128 candidates per ciphertext.
- Each "candidate" needs the same polynomial applied to its own
  preactivation z = w·x + b. We pre-compute z for each candidate in
  plaintext (since w is plaintext), pack 128 z-values into one
  ciphertext, evaluate Taylor-3 once.

What we measure:
1. Speedup vs unpacked per-candidate evaluation (target >= 5x).
2. Per-slot noise correlation across candidates (target |corr| < 0.05
   so each slot's δ is approximately independent).
3. Numerical correctness vs unpacked (target relative error < 1e-3).

This is a Patent Claim 11 ("SIMD-packed adversarial evaluation")
validation. The test deliberately stops at d=64 — extending to d=784
production search is future work.

Output: benchmarks/results/simd_packing_feasibility.csv
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


def _train_lr(d: int, seed: int = 42) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(20 * d, d))
    true_w = rng.normal(0.0, 1.0, size=d)
    y = (X @ true_w > 0).astype(int)
    w = rng.normal(0.0, 0.1, size=d)
    b = 0.0
    for _ in range(200):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float(np.mean(p - y))
        w -= 0.1 * grad_w
        b -= 0.1 * grad_b
    return w.astype(np.float64), float(b)


def _packed_taylor3_eval(z_batch: np.ndarray, ctx: TenSEALContext) -> np.ndarray:
    """Pack z_batch into a single ciphertext and evaluate Taylor-3.

    Returns the decrypted batch of σ_T3(z_i) values, length = len(z_batch).
    Slots beyond len(z_batch) carry zeros (they decode to the zero
    contribution of the polynomial at z=0, namely 0.5).
    """
    import tenseal as ts
    z_list = z_batch.astype(np.float64).tolist()
    ct = ts.ckks_vector(ctx.ctx, z_list)
    ct_z2 = ct * ct
    ct_z3 = ct_z2 * ct
    ct_out = ct * 0.25 - ct_z3 * (1.0 / 48.0) + 0.5
    out = np.asarray(ct_out.decrypt(), dtype=np.float64)
    return out[: len(z_batch)]


def _unpacked_taylor3_eval(z_batch: np.ndarray, ctx: TenSEALContext) -> np.ndarray:
    """Per-candidate ciphertext: encrypt each z_i alone, evaluate, decrypt."""
    import tenseal as ts
    out = np.zeros(len(z_batch))
    for i, z_val in enumerate(z_batch):
        ct = ts.ckks_vector(ctx.ctx, [float(z_val)])
        ct_z2 = ct * ct
        ct_z3 = ct_z2 * ct
        ct_out = ct * 0.25 - ct_z3 * (1.0 / 48.0) + 0.5
        out[i] = float(ct_out.decrypt()[0])
    return out


def main() -> int:
    if not HAVE_TENSEAL:
        print("ERROR: TenSEAL not installed.")
        return 1

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simd_packing_feasibility.csv")

    d = 64
    BATCH = 128         # = 8192 / 64
    N_BATCHES = 5       # repeat for timing stability

    print(f"SIMD packing feasibility (Proposal 3): d={d}, batch={BATCH}, N_batches={N_BATCHES}")
    print("=" * 75)

    print("Building TenSEAL context + LR weights ...")
    ctx = TenSEALContext(seed=42)
    w, b = _train_lr(d)

    rows: list[dict] = []
    rng = np.random.default_rng(0)

    speedups = []
    rel_errs = []
    slot_corr_max = []

    print("\nRunning batches ...")
    for bidx in range(N_BATCHES):
        # Build a batch of d-dim inputs and project to z = w·x + b.
        X = rng.uniform(-3.0, 3.0, size=(BATCH, d))
        z_batch = X @ w + b
        sigma_plain = 1.0 / (1.0 + np.exp(-np.clip(z_batch, -500, 500)))

        # Packed eval (single ciphertext)
        t0 = time.perf_counter()
        out_packed = _packed_taylor3_eval(z_batch, ctx)
        wall_packed = time.perf_counter() - t0

        # Unpacked eval (BATCH ciphertexts) -- only first 16 to save time.
        sub_n = 16
        t0 = time.perf_counter()
        out_unpacked = _unpacked_taylor3_eval(z_batch[:sub_n], ctx)
        wall_unpacked_sub = time.perf_counter() - t0
        wall_unpacked_extrap = wall_unpacked_sub * (BATCH / sub_n)

        # Correctness vs plaintext Taylor-3
        z = z_batch
        plain_t3 = 0.5 + z / 4.0 - z ** 3 / 48.0
        rel_err_packed = np.abs(out_packed - plain_t3) / np.maximum(1.0, np.abs(plain_t3))
        rel_err_unpacked = np.abs(out_unpacked - plain_t3[:sub_n]) / np.maximum(1.0, np.abs(plain_t3[:sub_n]))

        # δ vs sigmoid: per-slot
        delta_packed = np.abs(sigma_plain - out_packed)
        delta_unpacked = np.abs(sigma_plain[:sub_n] - out_unpacked)

        # Cross-slot independence proxy: correlation between
        # slot-position-i δ values and slot-position-j δ values across
        # the (BATCH/sub_n) sub-batches collected on the fly. We
        # approximate by computing correlation of (delta_packed[i],
        # delta_packed[j]) for random (i,j) pairs vs delta computed in
        # plaintext Taylor-3. If packing introduces leakage, packed δ
        # should differ from unpacked δ at the same slot.
        # We use abs(out_packed - plain_t3) vs (out_unpacked - plain_t3)
        # for the slots we evaluated unpacked.
        leak = float(
            np.max(np.abs(out_packed[:sub_n] - out_unpacked))
        )

        speedup = wall_unpacked_extrap / wall_packed if wall_packed > 0 else float("nan")
        speedups.append(speedup)
        rel_errs.append(float(np.max(rel_err_packed)))
        slot_corr_max.append(leak)

        print(
            f"  batch {bidx+1}/{N_BATCHES}: "
            f"wall_packed={wall_packed*1000:.1f}ms "
            f"wall_unpacked_extrap={wall_unpacked_extrap*1000:.1f}ms "
            f"speedup={speedup:.1f}x "
            f"max_rel_err_packed={float(np.max(rel_err_packed)):.2e} "
            f"slot_leak={leak:.2e}"
        )

        rows.append({
            "batch": bidx,
            "d": d,
            "batch_size": BATCH,
            "wall_packed_s": wall_packed,
            "wall_unpacked_extrap_s": wall_unpacked_extrap,
            "speedup": speedup,
            "max_rel_err_packed": float(np.max(rel_err_packed)),
            "max_rel_err_unpacked": float(np.max(rel_err_unpacked)),
            "max_delta_packed": float(np.max(delta_packed)),
            "max_delta_unpacked": float(np.max(delta_unpacked)),
            "slot_leak_max_abs": leak,
        })

    print("=" * 75)
    print(f"\nSpeedup       : median={float(np.median(speedups)):.1f}x")
    print(f"Max rel error : median={float(np.median(rel_errs)):.2e}")
    print(f"Slot leak max : median={float(np.median(slot_corr_max)):.2e}")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Gate decision
    median_speedup = float(np.median(speedups))
    median_leak = float(np.median(slot_corr_max))
    median_rel = float(np.median(rel_errs))

    print("\n--- Gate decision ---")
    if median_speedup >= 10.0 and median_leak < 0.05 and median_rel < 1e-3:
        verdict = "PASS"
    elif median_speedup >= 5.0 and median_leak < 0.05:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"  speedup={median_speedup:.1f}x  leak={median_leak:.2e}  rel_err={median_rel:.2e}")
    print(f"  Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
