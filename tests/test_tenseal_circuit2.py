# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Tests for Circuit 2 (depth-4 polynomial, d=6) on real CKKS."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")),
)

from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL

pytestmark = pytest.mark.skipif(
    not HAVE_TENSEAL, reason="TenSEAL not installed"
)


def _load():
    from fhe_oracle.adapters.tenseal_adapter import TenSEALContext
    from tenseal_circuits import (
        _circuit2_plaintext_fn,
        build_tenseal_circuit2,
        make_tenseal_circuit2_fhe_fn,
    )
    return {
        "TenSEALContext": TenSEALContext,
        "plaintext": _circuit2_plaintext_fn,
        "make_fhe": make_tenseal_circuit2_fhe_fn,
        "build": build_tenseal_circuit2,
    }


def test_plaintext_ones_equals_sum_coeffs():
    """p([1]*6) = Σ c_i = 0.5 + 0.75 + 1.0 + 1.25 + 1.5 = 5.0."""
    mod = _load()
    x = [1.0] * 6
    assert mod["plaintext"](x) == pytest.approx(5.0)


def test_plaintext_zeros_equals_zero():
    mod = _load()
    assert mod["plaintext"]([0.0] * 6) == pytest.approx(0.0)


def test_fhe_matches_plaintext_small_input():
    """CKKS noise on a small input should be well under 1e-3."""
    mod = _load()
    ctx = mod["TenSEALContext"]()
    fhe_fn = mod["make_fhe"](ctx, d=6)
    x = [0.5, -0.3, 0.2, 0.1, -0.4, 0.6]
    expected = mod["plaintext"](x)
    actual = fhe_fn(x)
    assert abs(expected - actual) < 1e-3


def test_fhe_matches_plaintext_ones():
    mod = _load()
    ctx = mod["TenSEALContext"]()
    fhe_fn = mod["make_fhe"](ctx, d=6)
    x = [1.0] * 6
    assert abs(fhe_fn(x) - 5.0) < 1e-3


def test_divergence_grows_with_magnitude():
    """CKKS noise at x=2 box corner > noise near zero."""
    mod = _load()
    ctx = mod["TenSEALContext"]()
    fhe_fn = mod["make_fhe"](ctx, d=6)

    x_big = [2.0] * 6
    plain_big = mod["plaintext"](x_big)
    fhe_big = fhe_fn(x_big)
    err_big = abs(plain_big - fhe_big)

    x_small = [0.05] * 6
    plain_small = mod["plaintext"](x_small)
    fhe_small = fhe_fn(x_small)
    err_small = abs(plain_small - fhe_small)

    assert plain_big == pytest.approx(40.0)
    assert err_big >= err_small


def test_build_circuit2_shape():
    mod = _load()
    ctx = mod["TenSEALContext"]()
    c = mod["build"](ctx)
    assert c["name"] == "circuit2_tenseal"
    assert c["d"] == 6
    assert c["bounds"] == [(-2.0, 2.0)] * 6
    assert callable(c["plain"])
    assert callable(c["fhe"])
    assert len(c["coeffs"]) == 5
    assert c["coeffs"][0] == pytest.approx(0.5)
    assert c["coeffs"][-1] == pytest.approx(1.5)
