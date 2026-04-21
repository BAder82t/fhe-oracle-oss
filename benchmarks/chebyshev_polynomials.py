# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chebyshev-basis polynomial sigmoid approximations for C2.

Provides plaintext and TenSEAL-compatible polynomial approximations
of the logistic sigmoid σ(z) = 1 / (1 + exp(-z)) at degrees 3, 7,
and 15. Each coefficient vector is fit in the Chebyshev basis over
the interval [-5, 5] via ``numpy.polynomial.chebyshev.chebfit`` and
converted to the power basis (``cheb2poly``) for Horner evaluation.

Typical max approximation errors on [-5, 5]:

- Cheb-3  → ~1e-1  (comparable to the paper's Taylor-3)
- Cheb-7  → ~1e-3  (transition regime)
- Cheb-15 → ~1e-6  (comparable to CKKS residual noise)

Design note: the paper's Taylor-3 polynomial
``σ_T3(z) = 0.5 + z/4 - z^3/48`` is a specific hand-picked expansion
around 0; Cheb-3 fit on [-5, 5] is a *different* degree-3 polynomial
optimised for uniform error. The C2 experiment compares both.

CKKS parameter tables supported by TenSEAL (probed empirically on
this machine — see ``research/experiment-plan/results/C2-gate-
decision.md`` for the probe results):

- Taylor-3 / Cheb-3: N=16384, chain=[60, 40, 40, 40, 40, 60].
- Cheb-7: N=16384, chain=[60] + [40]*7 + [60].
- Cheb-15: N=32768, chain=[60] + [40]*16 + [60].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.polynomial import chebyshev as _cheb


@dataclass(frozen=True)
class PolyApprox:
    """A fitted polynomial approximation of the sigmoid.

    Attributes
    ----------
    degree : int
        Polynomial degree.
    domain : tuple[float, float]
        Fit interval.
    cheb_coeffs : np.ndarray
        Chebyshev-basis coefficients (length degree + 1).
    power_coeffs : np.ndarray
        Power-basis coefficients (length degree + 1). Index i is
        the coefficient of z**i.
    fit_error : float
        Empirical max |sigmoid - poly| on a dense grid over ``domain``.
    """

    degree: int
    domain: tuple[float, float]
    cheb_coeffs: np.ndarray
    power_coeffs: np.ndarray
    fit_error: float


def fit_cheb_sigmoid(
    degree: int,
    domain: tuple[float, float] = (-5.0, 5.0),
    n_fit: int = 1000,
) -> PolyApprox:
    """Fit a Chebyshev polynomial to the logistic sigmoid on ``domain``."""
    z = np.linspace(domain[0], domain[1], n_fit)
    sigma = 1.0 / (1.0 + np.exp(-z))
    cheb_coeffs = _cheb.chebfit(z, sigma, degree)
    power_coeffs = _cheb.cheb2poly(cheb_coeffs)
    # Empirical fit error on a denser grid
    z_dense = np.linspace(domain[0], domain[1], 10_000)
    sigma_dense = 1.0 / (1.0 + np.exp(-z_dense))
    fit = np.polynomial.polynomial.polyval(z_dense, power_coeffs)
    fit_error = float(np.max(np.abs(sigma_dense - fit)))
    return PolyApprox(
        degree=degree,
        domain=domain,
        cheb_coeffs=np.asarray(cheb_coeffs, dtype=np.float64),
        power_coeffs=np.asarray(power_coeffs, dtype=np.float64),
        fit_error=fit_error,
    )


def eval_poly_plaintext(z: float, approx: PolyApprox) -> float:
    """Evaluate the polynomial approximation at scalar z (plaintext)."""
    return float(
        np.polynomial.polynomial.polyval(float(z), approx.power_coeffs)
    )


def _horner_encrypted(ct_z: Any, power_coeffs: np.ndarray):
    """Horner evaluation under TenSEAL encryption.

    Computes ``c_0 + z*(c_1 + z*(c_2 + ... + z*c_n))`` where the
    coefficients are in power-basis order ``c_0, c_1, ..., c_n``.

    The first operation starts with a float accumulator (the
    highest-degree coefficient) multiplied by the ciphertext; every
    subsequent step is a ciphertext-ciphertext-or-scalar mul and an
    add. Consumes ``degree`` ciphertext-ciphertext multiplications.
    """
    coeffs = np.asarray(power_coeffs, dtype=np.float64)
    result: Any = float(coeffs[-1])
    for i in range(len(coeffs) - 2, -1, -1):
        c = float(coeffs[i])
        if isinstance(result, float):
            result = ct_z * result + c
        else:
            result = result * ct_z + c
    return result


def make_tenseal_poly_lr_fhe_fn(
    weights: np.ndarray,
    bias: float,
    approx: PolyApprox,
    tenseal_ctx: Any,
) -> Callable[[Any], float]:
    """Build an FHE function computing poly(W·x + b) under CKKS.

    Parameters
    ----------
    weights, bias
        Linear layer parameters.
    approx
        Polynomial approximation (from ``fit_cheb_sigmoid``). For the
        Taylor-3 baseline, pass ``taylor3_approx()``.
    tenseal_ctx
        A TenSEAL context with enough multiplicative levels for
        ``approx.degree`` ciphertext-ciphertext multiplications.

    The returned callable takes ``x`` (list or ndarray) and returns
    the scalar decrypted output from slot 0.
    """
    w_list = np.asarray(weights, dtype=np.float64).tolist()
    b_val = float(bias)

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        ct_x = tenseal_ctx.encrypt(xa)
        ct_z = ct_x.dot(w_list) + b_val
        ct_result = _horner_encrypted(ct_z, approx.power_coeffs)
        out = tenseal_ctx.decrypt(ct_result)
        return float(out[0])

    return fhe_fn


def taylor3_approx(domain: tuple[float, float] = (-5.0, 5.0)) -> PolyApprox:
    """Return the paper's Taylor-3 polynomial ``0.5 + z/4 - z^3/48``.

    Wraps the hand-coded Taylor-3 coefficients as a ``PolyApprox``
    so it can share the same plumbing as the Chebyshev approximations.
    """
    power_coeffs = np.array([0.5, 0.25, 0.0, -1.0 / 48.0], dtype=np.float64)
    cheb_coeffs = _cheb.poly2cheb(power_coeffs)
    z_dense = np.linspace(domain[0], domain[1], 10_000)
    sigma_dense = 1.0 / (1.0 + np.exp(-z_dense))
    fit = np.polynomial.polynomial.polyval(z_dense, power_coeffs)
    fit_error = float(np.max(np.abs(sigma_dense - fit)))
    return PolyApprox(
        degree=3,
        domain=domain,
        cheb_coeffs=np.asarray(cheb_coeffs, dtype=np.float64),
        power_coeffs=power_coeffs,
        fit_error=fit_error,
    )


# --- TenSEAL-context helper for the C2 ablation ---------------------------
#
# Scale and chain mid-prime bit size must match — CKKS rescales by the
# next unused prime after each ciphertext multiply, so the working scale
# bounces around ``scale_bits`` only when the chain's interior primes are
# the same width. Mismatched pairs (e.g. scale=2^30 with a 40-bit interior
# chain) decay the scale below decryption viability after a few multiplies.
#
# ``_poly_modulus_degree_for`` returns the smallest ``N`` whose total
# modulus-bit budget accommodates the chain ``[60] + [scale_bits]*m + [60]``
# where ``m`` is the requested level count.


_N_BUDGETS: list[tuple[int, int]] = [
    (8192, 218),
    (16384, 438),
    (32768, 881),
    (65536, 1772),
]


def _poly_modulus_degree_for(total_bits: int) -> int:
    for N, budget in _N_BUDGETS:
        if total_bits <= budget:
            return N
    raise ValueError(
        f"No supported poly_modulus_degree for total_bits={total_bits}; "
        f"max budget is {_N_BUDGETS[-1][1]} at N={_N_BUDGETS[-1][0]}"
    )


def build_tenseal_context(
    degree: int,
    scale_bits: int = 40,
    slack_levels: int = 1,
):
    """Build a TenSEAL CKKS context sized for Horner evaluation of degree ``degree``.

    Parameters
    ----------
    degree
        Polynomial degree (3, 7, or 15 are the probed values; any
        non-negative integer is accepted).
    scale_bits
        Working scale is ``2 ** scale_bits``. Also the width of each
        interior prime in the coefficient-modulus chain — this matches
        the working scale so rescaling keeps scale stable.
    slack_levels
        Extra interior primes beyond the ``degree`` needed for Horner
        evaluation. Default 1 reserves one level for the ``W·x + b``
        dot-product prefix that precedes Horner in
        :func:`make_tenseal_poly_lr_fhe_fn`. Set to ``0`` if evaluating
        a bare polynomial with no prefix; ``>=2`` adds noise-budget
        headroom.

    Raises
    ------
    ImportError
        If TenSEAL is not installed.
    ValueError
        If the resulting chain does not fit any supported poly_modulus_degree.
    """
    import tenseal as ts  # deferred so the module loads without TenSEAL

    n_interior = max(0, degree) + max(0, slack_levels)
    chain = [60] + [int(scale_bits)] * n_interior + [60]
    total_bits = sum(chain)
    N = _poly_modulus_degree_for(total_bits)

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=N,
        coeff_mod_bit_sizes=chain,
    )
    ctx.global_scale = 2 ** int(scale_bits)
    ctx.generate_galois_keys()

    class _CtxWrapper:
        def __init__(self, inner, scale_bits, N, chain, slack_levels):
            self.ctx = inner
            self.scale_bits = int(scale_bits)
            self.N = int(N)
            self.chain = list(chain)
            self.slack_levels = int(slack_levels)

        def encrypt(self, x):
            arr = np.asarray(x, dtype=np.float64).ravel()
            return ts.ckks_vector(self.ctx, arr.tolist())

        def decrypt(self, ct):
            return np.asarray(ct.decrypt(), dtype=np.float64)

    return _CtxWrapper(ctx, scale_bits, N, chain, slack_levels)
