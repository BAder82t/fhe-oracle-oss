# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for TenSEAL adapter (B1) — skipped if TenSEAL unavailable."""

from __future__ import annotations

import time

import numpy as np
import pytest

from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL

pytestmark = pytest.mark.skipif(
    not HAVE_TENSEAL, reason="TenSEAL not installed"
)


def _get_adapter_mod():
    from fhe_oracle.adapters.tenseal_adapter import (
        TenSEALContext,
        TenSEALTaylor3Adapter,
        make_tenseal_chebyshev_fhe_fn,
        make_tenseal_taylor3_fhe_fn,
    )
    return {
        "TenSEALContext": TenSEALContext,
        "TenSEALTaylor3Adapter": TenSEALTaylor3Adapter,
        "make_tenseal_chebyshev_fhe_fn": make_tenseal_chebyshev_fhe_fn,
        "make_tenseal_taylor3_fhe_fn": make_tenseal_taylor3_fhe_fn,
    }


def test_encrypt_decrypt_roundtrip():
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    x = np.array([1.0, 2.0, 3.0, -1.5, 0.0])
    ct = ctx.encrypt(x)
    x_dec = ctx.decrypt(ct)
    np.testing.assert_allclose(x_dec[:5], x, atol=1e-4)


def test_taylor3_sigmoid_matches_plaintext_small_z():
    """FHE Taylor-3 matches plaintext Taylor-3 to CKKS noise (~1e-5)."""
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    weights = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
    bias = 0.1
    fhe_fn = mod["make_tenseal_taylor3_fhe_fn"](weights, bias, ctx)
    x = np.array([1.0, -0.5, 0.3, 0.7, -1.0])
    z = float(np.dot(weights, x) + bias)
    expected = 0.5 + z / 4 - z ** 3 / 48
    actual = fhe_fn(x)
    # CKKS noise at these params is ~1e-6 to 1e-5
    assert abs(actual - expected) < 1e-4


def test_divergence_diverges_on_extreme_input():
    """At large |z|, Taylor-3 diverges from true sigmoid."""
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bias = 0.0
    fhe_fn = mod["make_tenseal_taylor3_fhe_fn"](weights, bias, ctx)
    x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # z = 15
    z = 15.0
    true_sig = 1.0 / (1.0 + np.exp(-z))
    fhe_result = fhe_fn(x)
    divergence = abs(true_sig - fhe_result)
    # Taylor-3 at z=15: 0.5 + 15/4 − 3375/48 = 4.25 - 70.3125 = -66.0625
    # vs true_sig ≈ 1.0 → divergence ~ 67
    assert divergence > 10.0


def test_fhe_eval_wall_clock_under_threshold():
    """Single FHE eval should take < 500ms on modern CPU."""
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    weights = np.array([0.5, -0.3, 0.8])
    bias = 0.1
    fhe_fn = mod["make_tenseal_taylor3_fhe_fn"](weights, bias, ctx)
    x = np.array([1.0, -0.5, 0.3])
    fhe_fn(x)  # warm-up
    t0 = time.perf_counter()
    for _ in range(3):
        fhe_fn(x)
    elapsed = (time.perf_counter() - t0) / 3
    assert elapsed < 0.5


def test_chebyshev_fhe_matches_plaintext():
    """Chebyshev-3 FHE output matches plaintext within CKKS noise."""
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    rng = np.random.default_rng(123)
    W = rng.standard_normal((4, 10)) * 0.5
    b = rng.standard_normal(4) * 0.1
    fhe_fn = mod["make_tenseal_chebyshev_fhe_fn"](W, b, ctx)
    x = np.array([0.5, -0.3, 0.2, 0.1, -0.5, 0.4, -0.1, 0.3, -0.2, 0.1])
    h = W @ x + b
    expected = 0.5 + 0.15 * h - h ** 3 / 500.0
    actual = fhe_fn(x)
    np.testing.assert_allclose(actual[:4], expected, atol=1e-3)


def test_adapter_protocol_methods():
    """TenSEALTaylor3Adapter implements the FHEAdapter protocol."""
    mod = _get_adapter_mod()
    ctx = mod["TenSEALContext"]()
    w = np.array([0.5, -0.3, 0.8])
    adapter = mod["TenSEALTaylor3Adapter"](w, 0.1, ctx)
    x = [1.0, -0.5, 0.3]
    ct = adapter.encrypt(x)
    ct_out = adapter.run_fhe_program(ct)
    result = adapter.decrypt(ct_out)
    assert len(result) >= 1
    assert adapter.get_scheme_name() == "CKKS-TenSEAL"
    assert adapter.get_mult_depth_used(ct_out) == 3
    assert adapter.get_noise_budget(ct_out) > 0
