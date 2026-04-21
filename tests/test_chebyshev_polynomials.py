# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for benchmarks/chebyshev_polynomials.py (C2)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_BENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmarks")
)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from chebyshev_polynomials import (  # noqa: E402
    _horner_encrypted,
    build_tenseal_context,
    eval_poly_plaintext,
    fit_cheb_sigmoid,
    make_tenseal_poly_lr_fhe_fn,
    taylor3_approx,
)

from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL  # noqa: E402


def test_cheb3_fit_error_bucket():
    approx = fit_cheb_sigmoid(3)
    assert 1e-2 <= approx.fit_error <= 1e0


def test_cheb7_fit_error_bucket():
    approx = fit_cheb_sigmoid(7)
    assert 1e-5 <= approx.fit_error <= 1e-2


def test_cheb15_fit_error_bucket():
    approx = fit_cheb_sigmoid(15)
    assert approx.fit_error < 1e-4


def test_cheb15_beats_cheb7_beats_cheb3():
    a3 = fit_cheb_sigmoid(3)
    a7 = fit_cheb_sigmoid(7)
    a15 = fit_cheb_sigmoid(15)
    assert a15.fit_error < a7.fit_error < a3.fit_error


def test_power_basis_roundtrip_matches_chebval():
    approx = fit_cheb_sigmoid(7)
    z = np.linspace(-4.0, 4.0, 50)
    from numpy.polynomial.chebyshev import chebval

    cheb_values = chebval(z, approx.cheb_coeffs)
    power_values = np.polynomial.polynomial.polyval(z, approx.power_coeffs)
    np.testing.assert_allclose(cheb_values, power_values, atol=1e-10)


def test_eval_poly_plaintext_matches_power_polyval():
    approx = fit_cheb_sigmoid(7)
    for z in [-4.0, -1.0, 0.0, 1.0, 3.5]:
        expected = float(
            np.polynomial.polynomial.polyval(z, approx.power_coeffs)
        )
        assert eval_poly_plaintext(z, approx) == pytest.approx(expected, abs=1e-12)


def test_taylor3_approx_matches_paper_formula():
    approx = taylor3_approx()
    np.testing.assert_allclose(
        approx.power_coeffs,
        np.array([0.5, 0.25, 0.0, -1.0 / 48.0]),
        atol=1e-15,
    )


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_horner_encrypted_matches_plaintext_within_noise_cheb3():
    approx = fit_cheb_sigmoid(3)
    ctx = build_tenseal_context(degree=3, scale_bits=40)
    for z_val in [-2.0, 0.3, 1.8]:
        ct = ctx.encrypt([z_val, z_val, z_val])
        ct_result = _horner_encrypted(ct, approx.power_coeffs)
        dec = ctx.decrypt(ct_result)
        expected = float(
            np.polynomial.polynomial.polyval(z_val, approx.power_coeffs)
        )
        assert abs(dec[0] - expected) < 1e-4


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_tenseal_poly_lr_fhe_fn_wraps_horner():
    approx = fit_cheb_sigmoid(3)
    ctx = build_tenseal_context(degree=3, scale_bits=40)
    w = np.array([0.5, -0.3, 0.2])
    b = 0.1
    fhe_fn = make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx)
    x = [1.0, -1.5, 0.5]
    z_plain = float(np.dot(w, x) + b)
    expected = eval_poly_plaintext(z_plain, approx)
    got = fhe_fn(x)
    assert abs(got - expected) < 1e-4


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_build_tenseal_context_accepts_arbitrary_degree():
    ctx = build_tenseal_context(degree=11, scale_bits=40)
    assert ctx.scale_bits == 40
    assert ctx.N in (16384, 32768, 65536)
    assert len(ctx.chain) >= 11 + 2


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_build_tenseal_context_scale_matches_interior_primes():
    ctx30 = build_tenseal_context(degree=5, scale_bits=30)
    assert ctx30.scale_bits == 30
    assert all(p == 30 for p in ctx30.chain[1:-1])
    ctx40 = build_tenseal_context(degree=5, scale_bits=40)
    assert ctx40.scale_bits == 40
    assert all(p == 40 for p in ctx40.chain[1:-1])


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_build_tenseal_context_adds_slack_levels():
    ctx = build_tenseal_context(degree=3, scale_bits=40, slack_levels=0)
    assert len(ctx.chain) - 2 == 3
    ctx2 = build_tenseal_context(degree=3, scale_bits=40, slack_levels=2)
    assert len(ctx2.chain) - 2 == 5
