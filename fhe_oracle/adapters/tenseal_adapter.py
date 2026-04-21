# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""TenSEAL CKKS adapter for FHE Oracle (B1).

Provides real CKKS encryption, homomorphic evaluation, and decryption
via TenSEAL. Used to validate A-spine results (A1/A2/A3) beyond
mocks.

Requires: `pip install tenseal`. All TenSEAL-dependent symbols are
gated behind `HAVE_TENSEAL`; the core library remains importable
without TenSEAL installed.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import tenseal as ts
    HAVE_TENSEAL = True
except ImportError:
    ts = None  # type: ignore
    HAVE_TENSEAL = False


CKKS_POLY_MODULUS_DEGREE = 16384
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40


class TenSEALContext:
    """Wrapper around a TenSEAL CKKS context with paper-matched parameters.

    CKKS parameters (paper §6.5-6.7):
    - poly_modulus_degree: 16384  (N)
    - coeff_mod_bit_sizes: [60, 40, 40, 40, 40, 60]  (4 mult levels)
    - global_scale: 2^40

    Taylor-3 (degree 3) consumes 2 ciphertext-ciphertext multiplies
    (z², z³) plus scalar multiplies — fits comfortably.
    """

    def __init__(self, seed: int = 42) -> None:
        if not HAVE_TENSEAL:
            raise ImportError(
                "TenSEAL is not installed. pip install tenseal"
            )
        self._seed = seed
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
        )
        self.ctx.global_scale = CKKS_GLOBAL_SCALE
        self.ctx.generate_galois_keys()

    def encrypt(self, x: np.ndarray | list[float]) -> Any:
        arr = np.asarray(x, dtype=np.float64).ravel()
        return ts.ckks_vector(self.ctx, arr.tolist())

    def decrypt(self, ct: Any) -> np.ndarray:
        return np.asarray(ct.decrypt(), dtype=np.float64)


def make_tenseal_taylor3_fhe_fn(
    weights: np.ndarray,
    bias: float,
    tenseal_ctx: TenSEALContext,
) -> Callable[[np.ndarray | list[float]], float]:
    """Build an FHE function computing σ_T3(z) = 0.5 + z/4 − z³/48
    under CKKS, where z = W · x + b.

    Returns a scalar float (σ_T3 evaluated at the first slot).
    """
    w_list = np.asarray(weights, dtype=np.float64).tolist()
    b_val = float(bias)

    def fhe_fn(x: np.ndarray | list[float]) -> float:
        xa = np.asarray(x, dtype=np.float64)
        ct_x = tenseal_ctx.encrypt(xa)
        # z = W · x + b  (encrypted scalar, slot 0)
        ct_z = ct_x.dot(w_list) + b_val
        # Taylor-3: 0.5 + z/4 − z³/48
        ct_z2 = ct_z * ct_z
        ct_z3 = ct_z2 * ct_z
        ct_result = ct_z * 0.25 - ct_z3 * (1.0 / 48.0) + 0.5
        out = tenseal_ctx.decrypt(ct_result)
        return float(out[0])

    return fhe_fn


def make_tenseal_chebyshev_fhe_fn(
    W: np.ndarray,
    b: np.ndarray,
    tenseal_ctx: TenSEALContext,
) -> Callable[[np.ndarray | list[float]], np.ndarray]:
    """Build an FHE function computing the dense + Chebyshev-3 layer.

    f(x) = 0.5 + 0.15 · h − h³ / 500, where h = W · x + b (vector).

    Returns a decrypted numpy array (shape matches b).
    """
    hidden = len(b)
    W_T_list = W.T.tolist()       # shape (d, hidden) — format ts.mm expects
    b_list = np.asarray(b, dtype=np.float64).tolist()

    def fhe_fn(x: np.ndarray | list[float]) -> np.ndarray:
        xa = np.asarray(x, dtype=np.float64)
        ct_x = tenseal_ctx.encrypt(xa)
        # h = W · x + b  (encrypted vector, `hidden` slots populated)
        ct_h = ct_x.mm(W_T_list) + b_list
        # Chebyshev-3 applied coordinate-wise
        ct_h2 = ct_h * ct_h
        ct_h3 = ct_h2 * ct_h
        ct_out = ct_h * 0.15 - ct_h3 * (1.0 / 500.0) + 0.5
        out = tenseal_ctx.decrypt(ct_out)
        return out[:hidden]

    return fhe_fn


class TenSEALTaylor3Adapter:
    """Adapter wrapping a Taylor-3 LR circuit for the FHEOracle.

    Implements the FHEAdapter protocol informally (duck-typed). The
    adapter's `encrypt`, `run_fhe_program`, and `decrypt` each
    correspond to one FHE operation on one fresh ciphertext per input
    `x`.
    """

    def __init__(self, weights: np.ndarray, bias: float, tenseal_ctx: TenSEALContext):
        self._weights = np.asarray(weights, dtype=np.float64)
        self._bias = float(bias)
        self._ctx = tenseal_ctx
        self._fhe_fn = make_tenseal_taylor3_fhe_fn(weights, bias, tenseal_ctx)

    def get_scheme_name(self) -> str:
        return "CKKS-TenSEAL"

    def encrypt(self, x: list[float]) -> Any:
        return self._ctx.encrypt(np.asarray(x))

    def run_fhe_program(self, ct: Any) -> Any:
        # Compute z = W·x + b under encryption and return the result CT.
        ct_z = ct.dot(self._weights.tolist()) + self._bias
        ct_z2 = ct_z * ct_z
        ct_z3 = ct_z2 * ct_z
        return ct_z * 0.25 - ct_z3 * (1.0 / 48.0) + 0.5

    def decrypt(self, ct: Any) -> list[float]:
        return self._ctx.decrypt(ct).tolist()

    def get_noise_budget(self, ct: Any) -> float:
        # CKKS has no binary noise budget; return scale as a
        # surrogate. TenSEAL's underlying SEAL library does not
        # expose invariant_noise_budget for CKKS (CKKS noise is
        # residual, not hard-fail).
        try:
            return float(ct.scale())
        except Exception:
            return float(CKKS_GLOBAL_SCALE)

    def get_mult_depth_used(self, ct: Any) -> int:
        # Taylor-3 consumes 2 ct-ct mults + 2 scalar mults = depth 3.
        return 3
