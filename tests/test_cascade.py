# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.cascade.CascadeSearch and evaluate_correlation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fhe_oracle.cascade import CascadeSearch, evaluate_correlation


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _taylor3(z: float) -> float:
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _cheb15(z: float) -> float:
    """Stand-in for a high-degree poly with similar peak-error structure."""
    return 0.5 + z / 4.0 - z ** 3 / 48.0 + z ** 5 / 3840.0


def _plain_lr(w, b):
    def f(x):
        z = float(np.dot(w, x) + b)
        return _sigmoid(z)
    return f


def _fhe_taylor(w, b):
    def f(x):
        z = float(np.dot(w, x) + b)
        return _taylor3(z)
    return f


def _fhe_cheb(w, b):
    def f(x):
        z = float(np.dot(w, x) + b)
        return _cheb15(z)
    return f


def test_correlation_high_for_shared_peak_error():
    rng = np.random.default_rng(0)
    d = 6
    w = rng.normal(size=d)
    b = 0.0
    plain = _plain_lr(w, b)
    cheap = _fhe_taylor(w, b)
    expensive = _fhe_cheb(w, b)
    samples = [rng.uniform(-3.0, 3.0, size=d) for _ in range(60)]
    out = evaluate_correlation(cheap, expensive, plain, samples)
    assert out["n_samples"] == 60
    assert out["spearman"] > 0.7, f"expected high rank correlation, got {out}"


def test_correlation_low_for_uncorrelated_fns():
    rng = np.random.default_rng(0)
    d = 4

    def plain(x):
        return 0.0

    def cheap(x):
        return float(x[0])

    def expensive(x):
        # Uncorrelated with cheap.
        return float(np.sin(7.0 * x[2]))

    samples = [rng.uniform(-1, 1, size=d) for _ in range(100)]
    out = evaluate_correlation(cheap, expensive, plain, samples)
    assert abs(out["spearman"]) < 0.3


def test_cascade_random_finds_winner_via_recheck():
    rng = np.random.default_rng(0)
    d = 4
    w = rng.normal(size=d)
    b = 0.0
    plain = _plain_lr(w, b)
    cheap = _fhe_taylor(w, b)
    expensive = _fhe_cheb(w, b)
    cs = CascadeSearch(
        cheap_fhe_fn=cheap,
        expensive_fhe_fn=expensive,
        plaintext_fn=plain,
        input_bounds=[(-3.0, 3.0)] * d,
        top_k=10,
    )
    out = cs.run(budget_cheap=200, seeds=[1, 2, 3], search_kind="random")
    assert len(out) == 3
    for r in out:
        assert r.n_evals_cheap == 200
        assert r.n_evals_expensive == 10
        assert r.max_error_expensive >= 0.0


def test_cascade_cma_returns_one_per_seed():
    rng = np.random.default_rng(1)
    d = 4
    w = rng.normal(size=d)
    b = 0.0
    cs = CascadeSearch(
        cheap_fhe_fn=_fhe_taylor(w, b),
        expensive_fhe_fn=_fhe_cheb(w, b),
        plaintext_fn=_plain_lr(w, b),
        input_bounds=[(-3.0, 3.0)] * d,
        top_k=5,
    )
    out = cs.run(budget_cheap=120, seeds=[7, 8], search_kind="cma")
    assert [r.seed for r in out] == [7, 8]
    for r in out:
        assert r.n_evals_cheap > 0
        assert len(r.x) == d


def test_cascade_unknown_kind_raises():
    cs = CascadeSearch(
        cheap_fhe_fn=lambda x: 0.0,
        expensive_fhe_fn=lambda x: 0.0,
        plaintext_fn=lambda x: 0.0,
        input_bounds=[(-1, 1)] * 2,
    )
    with pytest.raises(ValueError):
        cs.run(budget_cheap=10, seeds=[1], search_kind="bogus")
