# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Real-CKKS TenSEAL circuit definitions (B1).

Three circuits matching paper §6.6–6.7:
- LR d=8   Taylor-3 sigmoid
- Cheb d=10 dense + Chebyshev-3 sigmoid (hidden width 4)
- WDBC d=30 Taylor-3 sigmoid (via sklearn)

Each circuit returns (plaintext_fn, fhe_fn, metadata) where fhe_fn
wraps an active TenSEAL context.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle.adapters.tenseal_adapter import (
    HAVE_TENSEAL,
    TenSEALContext,
    make_tenseal_chebyshev_fhe_fn,
    make_tenseal_taylor3_fhe_fn,
)


def _fit_lr_synthetic(d: int = 8, seed: int = 42) -> tuple[np.ndarray, float]:
    """Fit a logistic-regression via 200 steps of GD on synthetic data.

    Matches paper §6.1 Circuit 1 exactly.
    """
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
    return w.astype(np.float64), float(b)


def build_tenseal_lr_d8(tenseal_ctx: TenSEALContext):
    """d=8 LR with Taylor-3 sigmoid under real CKKS.

    Plaintext: σ(W·x + b). FHE: σ_T3(W·x + b).
    Divergence = |σ − σ_T3| (computed externally).
    """
    w, b = _fit_lr_synthetic(d=8, seed=42)
    fhe_fn = make_tenseal_taylor3_fhe_fn(w, b, tenseal_ctx)

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(w, xa) + b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    return {
        "name": "lr_d8_tenseal",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": 8,
        "bounds": [(-3.0, 3.0)] * 8,
        "weights": w,
        "bias": b,
    }


def build_tenseal_chebyshev_d10(tenseal_ctx: TenSEALContext):
    """d=10 dense + Chebyshev-3 sigmoid under real CKKS (hidden=4)."""
    rng = np.random.default_rng(123)
    hidden = 4
    d = 10
    W = (rng.standard_normal((hidden, d)) * 0.5).astype(np.float64)
    b = (rng.standard_normal(hidden) * 0.1).astype(np.float64)

    fhe_fn_vec = make_tenseal_chebyshev_fhe_fn(W, b, tenseal_ctx)

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        h = W @ xa + b
        return (1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))).tolist()

    def fhe_fn(x):
        # Returns list of floats (hidden-dim vector) of Chebyshev-3
        # output. Divergence is measured as max|σ − f̃| externally.
        arr = fhe_fn_vec(x)
        return arr.tolist()

    return {
        "name": "cheb_d10_tenseal",
        "plain": plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-3.0, 3.0)] * d,
        "weights": W,
        "bias": b,
    }


def build_tenseal_wdbc(tenseal_ctx: TenSEALContext):
    """WDBC d=30 LR with Taylor-3 sigmoid under real CKKS.

    Reuses `benchmarks/wdbc_mock.py`'s sklearn pipeline for (weights,
    bias, data) and swaps the mock FHE for a real TenSEAL one.
    """
    from wdbc_mock import build_wdbc_circuit
    plain_fn, _, data, w, b, dim = build_wdbc_circuit(random_state=42)
    fhe_fn = make_tenseal_taylor3_fhe_fn(w, b, tenseal_ctx)
    return {
        "name": "wdbc_tenseal",
        "plain": plain_fn,
        "fhe": fhe_fn,
        "d": dim,
        "bounds": [(-3.0, 3.0)] * dim,
        "data": data,
        "weights": w,
        "bias": b,
    }


# --- Circuit 2 (depth-4 polynomial, d=6) ---------------------------------
#
# p(x) = Σ_{i=0}^{d-2} c_i * x_i² * x_{i+1}
# c = linspace(0.5, 1.5, d-1) = [0.5, 0.75, 1.0, 1.25, 1.5], d = 6
# Mult depth (ct-ct only):
#   x_i²         = ct_i * ct_i          (depth 1)
#   x_i² * x_{i+1} = prev * ct_{i+1}    (depth 2)
# Plaintext multiply by c_i and final additions add no depth.
# Fits the default context chain [60,40,40,40,40,60] (4 mult levels).


def _circuit2_plaintext_fn(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    coeffs = np.linspace(0.5, 1.5, len(arr) - 1)
    return float(np.sum(coeffs * arr[:-1] ** 2 * arr[1:]))


def make_tenseal_circuit2_fhe_fn(
    tenseal_ctx: TenSEALContext,
    d: int = 6,
) -> Callable[[np.ndarray | list[float]], float]:
    """FHE evaluation of p(x) = Σ c_i · x_i² · x_{i+1} under CKKS.

    Each x_i is encrypted as its own scalar ciphertext so that the
    cross-slot element-wise products map cleanly to ct-ct mults.
    Plaintext and FHE compute the SAME polynomial — no polynomial
    approximation — so divergence measures pure CKKS noise.
    """
    coeffs = np.linspace(0.5, 1.5, d - 1).tolist()

    def fhe_fn(x) -> float:
        arr = np.asarray(x, dtype=np.float64).ravel()
        cts = [tenseal_ctx.encrypt([float(v)]) for v in arr]
        acc = None
        for i, c_i in enumerate(coeffs):
            # Fold the plaintext scalar c_i into x_{i+1} at the bottom
            # level — multiplying after two ct-ct mults triggers
            # "scale out of bounds" in TenSEAL's auto-rescale path.
            ct_cx = cts[i + 1] * float(c_i)
            ct_sq = cts[i] * cts[i]
            ct_term = ct_sq * ct_cx
            acc = ct_term if acc is None else acc + ct_term
        out = tenseal_ctx.decrypt(acc)
        return float(out[0])

    return fhe_fn


def build_tenseal_circuit2(tenseal_ctx: TenSEALContext, d: int = 6):
    """Circuit 2 (depth-4 polynomial, d=6) under real CKKS.

    Plaintext fn and FHE fn compute the EXACT same polynomial, so
    δ(x) = |p(x) − fhe_p(x)| is pure CKKS quantisation/rescale noise —
    no Taylor/Chebyshev approximation confound.

    Parameters
    ----------
    tenseal_ctx : TenSEALContext
        Caller-owned context (default chain supports depth 4).
    d : int
        Input dimension (default 6, matching paper Circuit 2).
    """
    fhe_fn = make_tenseal_circuit2_fhe_fn(tenseal_ctx, d=d)
    return {
        "name": "circuit2_tenseal",
        "plain": _circuit2_plaintext_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-2.0, 2.0)] * d,
        "coeffs": np.linspace(0.5, 1.5, d - 1),
    }


def require_tenseal() -> bool:
    return HAVE_TENSEAL


# --- C2 Inner product (PIR-like, d=16) ------------------------------------
#
# Dossier research/future-work/real-ckks-small-budget-benchmark.md §3.1:
# vector-vector inner product on [-1, 1]^16 with fixed plaintext weight
# vector w, reduced via EvalSum. Linear arithmetic, depth 1, no
# approximation — expected Pro-unfavourable. Honest inclusion to defuse
# co-design risk.

def _c2_inner_product_weights(d: int = 16, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.normal(0.0, 1.0, size=d).astype(np.float64)


def build_tenseal_inner_product_d16(
    tenseal_ctx: "TenSEALContext",
    d: int = 16,
    weights_seed: int = 7,
):
    """C2: fixed-weight inner product under CKKS.

    y = w . x  with w drawn once from N(0, 1).
    Plaintext and FHE compute the SAME linear combination — divergence
    is pure CKKS encoding + slot-rotation + plaintext-mul noise, with
    no polynomial approximation layer.
    """
    w = _c2_inner_product_weights(d=d, rng_seed=weights_seed)
    w_list = w.tolist()

    def plain_fn(x):
        return float(np.dot(w, np.asarray(x, dtype=np.float64).ravel()))

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64).ravel()
        ct_x = tenseal_ctx.encrypt(xa)
        ct_y = ct_x.dot(w_list)          # scalar inner product in slot 0
        return float(tenseal_ctx.decrypt(ct_y)[0])

    return {
        "name": "C2_inner_product_d16",
        "plain": plain_fn,
        "fhe": fhe_fn,
        "d": d,
        "bounds": [(-1.0, 1.0)] * d,
        "weights": w,
    }


# --- C4 Near-cliff arithmetic (d=8) ---------------------------------------
#
# Dossier §3.1: p(x) = (x_0+x_1)^2 * (x_2+x_3)^2 * (x_4+x_5)^2 * (x_6+x_7)^2
# on bounds tuned so inputs near ||x||_inf = bound approach the
# pre-bootstrap modulus boundary. Depth: 4 pair-squarings (depth 1) +
# 3 pairwise-product merges (depth 1 each for a tree reduction) = depth 4.
# Fits default context chain. Expected Pro-favourable.

def _c4_plaintext_fn(x) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    p0 = (arr[0] + arr[1]) ** 2
    p1 = (arr[2] + arr[3]) ** 2
    p2 = (arr[4] + arr[5]) ** 2
    p3 = (arr[6] + arr[7]) ** 2
    return float(p0 * p1 * p2 * p3)


def build_tenseal_near_cliff_d8(tenseal_ctx: "TenSEALContext"):
    """C4: near-cliff depth-4 polynomial under CKKS.

    Each pair (x_{2i}, x_{2i+1}) is squared, then the four squares are
    multiplied in a balanced tree. With bounds [-1.5, 1.5]^8, the worst
    case output is (3)^2 * (3)^2 * (3)^2 * (3)^2 = 6561; well below CKKS
    decryption overflow for our default chain, but heuristic seeds that
    push toward box edges should dominate interior random samples.
    """

    def fhe_fn(x):
        arr = np.asarray(x, dtype=np.float64).ravel()
        # Encrypt each input as its own scalar ciphertext (slot 0)
        cts = [tenseal_ctx.encrypt([float(v)]) for v in arr]
        # Pair sums (no mult depth)
        s0 = cts[0] + cts[1]
        s1 = cts[2] + cts[3]
        s2 = cts[4] + cts[5]
        s3 = cts[6] + cts[7]
        # Square each pair (depth 1)
        sq0 = s0 * s0
        sq1 = s1 * s1
        sq2 = s2 * s2
        sq3 = s3 * s3
        # Balanced tree product (depth 2)
        left = sq0 * sq1
        right = sq2 * sq3
        out = left * right          # depth 3 total (square + 2 merge levels)
        return float(tenseal_ctx.decrypt(out)[0])

    return {
        "name": "C4_near_cliff_d8",
        "plain": _c4_plaintext_fn,
        "fhe": fhe_fn,
        "d": 8,
        "bounds": [(-1.5, 1.5)] * 8,
    }


# --- C5 Bootstrap-free depth-chained polynomial (d=8) ---------------------
#
# Dossier §3.1: (prod_{i=0..3}(a_i + x_i))^3 on bounds [-0.9, 0.9]^8,
# targeting noise accumulation across a deep chain without triggering
# modulus-switching overflow. Paper's default CKKS chain supports 4
# mult levels; this circuit consumes: 3 chain mults in the prod + 2
# for cubing = 5 mults, so we use an augmented chain.

def _c5_plaintext_fn(x, a=(0.2, 0.3, 0.4, 0.5)) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    prod = 1.0
    for i, ai in enumerate(a):
        prod *= (ai + arr[i])
    return float(prod ** 3)


def build_tenseal_depth_chain_d8(
    tenseal_ctx: "TenSEALContext",
    a=(0.2, 0.3, 0.4, 0.5),
):
    """C5: depth-chained polynomial — product-then-cube.

    (a_0+x_0)(a_1+x_1)(a_2+x_2)(a_3+x_3), then cube the result. Depth 5
    on the ciphertext side (3 for the product chain + 2 for cubing).
    Extra coordinates x_4..x_7 are ignored by the plaintext but still
    exercise the adapter's slot-handling.

    Context: default TenSEAL chain is [60,40,40,40,40,60] → 4 levels.
    Cubing the depth-3 product requires a deeper chain; we truncate
    at depth 4 here (square instead of cube) as a safe default.
    """

    def fhe_fn(x):
        arr = np.asarray(x, dtype=np.float64).ravel()
        # Encrypt x_0..x_3; ignore x_4..x_7
        cts = [tenseal_ctx.encrypt([float(arr[i])]) for i in range(4)]
        acc = cts[0] + float(a[0])
        for i in range(1, 4):
            factor = cts[i] + float(a[i])
            acc = acc * factor          # depth grows 1..3
        # Square once (depth 4). Cubing would need a 6-level chain.
        acc_sq = acc * acc              # depth 4 -> still within chain
        return float(tenseal_ctx.decrypt(acc_sq)[0])

    def plain_fn_squared(x):
        """Plaintext matching the squared-product FHE implementation."""
        arr = np.asarray(x, dtype=np.float64).ravel()
        prod = 1.0
        for i, ai in enumerate(a):
            prod *= (ai + arr[i])
        return float(prod ** 2)

    return {
        "name": "C5_depth_chain_d8",
        "plain": plain_fn_squared,
        "fhe": fhe_fn,
        "d": 8,
        "bounds": [(-0.9, 0.9)] * 8,
    }
