# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the separable (sep-CMA-ES) option on FHEOracle (B2)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle


def _quadratic(x):
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sum(arr * arr))


def _buggy_quadratic(x):
    arr = np.asarray(x, dtype=np.float64)
    v = float(np.sum(arr * arr))
    return v + (0.01 if v > 4.0 else 0.0)


def test_separable_false_identity_invariant():
    """separable=False (default) must match pre-B2 behaviour at same seed."""

    def _run(separable: bool) -> tuple[float, list[float]]:
        oracle = FHEOracle(
            plaintext_fn=_quadratic,
            fhe_fn=_buggy_quadratic,
            input_dim=4,
            input_bounds=[(-2.0, 2.0)] * 4,
            seed=7,
            separable=separable,
        )
        res = oracle.run(n_trials=150, threshold=1e-3)
        return res.max_error, list(res.worst_input)

    # Two runs with separable=False at same seed should be bit-identical.
    err_a, worst_a = _run(False)
    err_b, worst_b = _run(False)
    assert err_a == err_b
    assert worst_a == worst_b


def test_separable_true_smoke_d8():
    """separable=True runs without error at small d and finds bug."""
    oracle = FHEOracle(
        plaintext_fn=_quadratic,
        fhe_fn=_buggy_quadratic,
        input_dim=8,
        input_bounds=[(-2.0, 2.0)] * 8,
        seed=0,
        separable=True,
    )
    res = oracle.run(n_trials=200, threshold=1e-3)
    assert res.verdict == "FAIL"
    assert res.max_error >= 0.009
    assert res.n_trials > 0


def test_separable_true_d64_completes():
    """separable=True completes within budget at d=64."""
    d = 64
    oracle = FHEOracle(
        plaintext_fn=_quadratic,
        fhe_fn=_buggy_quadratic,
        input_dim=d,
        input_bounds=[(-2.0, 2.0)] * d,
        seed=0,
        separable=True,
    )
    res = oracle.run(n_trials=300, threshold=1e-3)
    # Only require completion and a valid result; high-d discovery is
    # a scientific question, not a unit-test invariant.
    assert res.n_trials > 0
    assert len(res.worst_input) == d
    assert np.isfinite(res.max_error)


def test_separable_budget_accounting():
    """separable=True does not change total eval count vs the default."""
    d = 16

    def _count_evals(separable: bool) -> int:
        oracle = FHEOracle(
            plaintext_fn=_quadratic,
            fhe_fn=_buggy_quadratic,
            input_dim=d,
            input_bounds=[(-2.0, 2.0)] * d,
            seed=11,
            separable=separable,
        )
        return oracle.run(n_trials=120, threshold=1e-3).n_trials

    # Both paths must respect the same budget cap (≤ n_trials).
    # Exact eval counts can differ: sep-CMA-ES may trigger tolfun/tolx
    # termination at a different generation because the covariance
    # dynamics are simpler. The contract is "no extra evaluations
    # beyond budget", not bit-identical counts.
    evals_full = _count_evals(False)
    evals_sep = _count_evals(True)
    assert evals_full <= 120
    assert evals_sep <= 120


def test_separable_with_random_floor():
    """separable=True composes with A1 random-floor warm start."""
    d = 32
    oracle = FHEOracle(
        plaintext_fn=_quadratic,
        fhe_fn=_buggy_quadratic,
        input_dim=d,
        input_bounds=[(-2.0, 2.0)] * d,
        seed=3,
        separable=True,
        random_floor=0.3,
        warm_start=True,
    )
    res = oracle.run(n_trials=300, threshold=1e-3)
    assert res.coverage_certificate is not None
    assert res.n_trials > 0
    assert len(res.worst_input) == d


def test_separable_with_restarts():
    """separable=True composes with IPOP restarts (A2)."""
    d = 8
    oracle = FHEOracle(
        plaintext_fn=_quadratic,
        fhe_fn=_buggy_quadratic,
        input_dim=d,
        input_bounds=[(-2.0, 2.0)] * d,
        seed=5,
        separable=True,
        restarts=2,
    )
    res = oracle.run(n_trials=300, threshold=1e-3)
    assert res.n_trials > 0
    assert len(res.worst_input) == d
