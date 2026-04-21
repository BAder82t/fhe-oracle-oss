# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for benchmarks/mnist_mock.py (B3)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmarks")
)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

pytest.importorskip("sklearn")

from mnist_mock import build_mnist_circuit  # noqa: E402


def test_build_mnist_circuit_returns_valid_tuple():
    plain, fhe, data, w, b, d = build_mnist_circuit()
    assert callable(plain)
    assert callable(fhe)
    assert isinstance(data, np.ndarray)
    assert isinstance(w, np.ndarray)
    assert isinstance(b, float)
    assert isinstance(d, int)
    assert d == 64  # load_digits default
    assert w.shape == (d,)
    assert data.shape[1] == d


def test_divergence_is_nonzero_on_training_samples():
    plain, fhe, data, w, b, d = build_mnist_circuit()
    deltas = [abs(plain(row) - fhe(row)) for row in data[:50]]
    # At least one sample must show a meaningful divergence.
    assert max(deltas) > 0.01


def test_deterministic_random_state():
    _, _, _, w1, b1, _ = build_mnist_circuit(random_state=42)
    _, _, _, w2, b2, _ = build_mnist_circuit(random_state=42)
    np.testing.assert_allclose(w1, w2)
    assert b1 == b2


def test_dim_matches_expected():
    _, _, _, w, _, d = build_mnist_circuit()
    assert d == 64
    assert len(w) == 64


def test_plaintext_and_fhe_disagree_at_large_z():
    plain, fhe, data, w, b, _ = build_mnist_circuit()
    # Construct an input aligned with w (maximising |z|).
    unit = w / max(1e-12, float(np.linalg.norm(w)))
    # Scale so |z| >> 5; Taylor-3 should diverge heavily.
    x_big = 10.0 * unit
    p = plain(x_big)
    f = fhe(x_big)
    # Plaintext sigmoid is bounded in [0, 1]; Taylor-3 is unbounded.
    assert 0.0 <= p <= 1.0
    assert abs(p - f) > 1.0
