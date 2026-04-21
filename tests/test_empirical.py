# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.empirical.EmpiricalSearch (A3)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.empirical import EmpiricalResult, EmpiricalSearch


def _make_plant_data(n: int = 50, d: int = 4, plant_idx: int = 0,
                     plant_err: float = 1.5) -> tuple[np.ndarray, callable]:
    """Return (data, divergence_fn) where only plant_idx-th row triggers big δ.

    data[plant_idx] is a "hot" point where divergence = plant_err.
    All other rows have divergence = 1e-5.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=(n, d))

    plant = data[plant_idx].copy()
    # Separate well from other samples so jitter can't confuse them.
    plant += 100.0
    data[plant_idx] = plant

    def div_fn(x):
        # Recognise the plant by proximity to its planted coordinates.
        if np.linalg.norm(x - plant) < 10.0:
            return float(plant_err)
        return 1e-5

    return data, div_fn


# --- Deterministic ---

def test_deterministic_same_seed():
    data, div_fn = _make_plant_data()
    r1 = EmpiricalSearch(div_fn, data, threshold=0.1, budget=100,
                         jitter_std=0.0, seed=7).run()
    r2 = EmpiricalSearch(div_fn, data, threshold=0.1, budget=100,
                         jitter_std=0.0, seed=7).run()
    assert r1.max_error == r2.max_error
    assert np.array_equal(r1.worst_input, r2.worst_input)
    assert r1.hits == r2.hits


# --- Finds known failure ---

def test_finds_planted_failure():
    data, div_fn = _make_plant_data(n=30, d=4, plant_idx=7, plant_err=2.0)
    res = EmpiricalSearch(div_fn, data, threshold=0.5, budget=200,
                          jitter_std=0.0, seed=0).run()
    # With 200 draws from 30 rows, the plant is almost certainly hit.
    assert res.verdict == "FAIL"
    assert res.max_error == pytest.approx(2.0, rel=1e-6)
    assert res.hits > 0


# --- Jitter effect ---

def test_jitter_zero_gives_exact_samples():
    data, div_fn = _make_plant_data()
    res = EmpiricalSearch(div_fn, data, threshold=0.5, budget=50,
                          jitter_std=0.0, seed=1).run()
    # No jitter → samples are exact rows; no error in the call itself.
    assert res.n_trials == 50


def test_jitter_nonzero_runs_without_error():
    data, div_fn = _make_plant_data()
    res = EmpiricalSearch(div_fn, data, threshold=0.5, budget=50,
                          jitter_std=1.0, seed=2).run()
    assert res.n_trials == 50


# --- PASS/FAIL verdict ---

def test_verdict_pass_when_all_below_threshold():
    data = np.zeros((10, 3))
    def div_fn(x):
        return 0.001
    res = EmpiricalSearch(div_fn, data, threshold=0.01, budget=30,
                          jitter_std=0.0, seed=0).run()
    assert res.verdict == "PASS"
    assert res.hits == 0


def test_verdict_fail_when_all_above_threshold():
    data = np.zeros((10, 3))
    def div_fn(x):
        return 0.1
    res = EmpiricalSearch(div_fn, data, threshold=0.01, budget=30,
                          jitter_std=0.0, seed=0).run()
    assert res.verdict == "FAIL"
    assert res.hits == 30
    assert res.mu_hat == 1.0


# --- mu_hat correctness ---

def test_mu_hat_half_and_half():
    """Divergence alternates above/below threshold → mu_hat ≈ 0.5."""
    data = np.arange(100, dtype=np.float64).reshape(100, 1)
    def div_fn(x):
        return 0.1 if int(x[0]) % 2 == 0 else 0.001
    # Jitter=0 to keep the x->even/odd mapping stable.
    res = EmpiricalSearch(div_fn, data, threshold=0.01, budget=1000,
                          jitter_std=0.0, seed=0).run()
    # With uniform sampling from 100 rows, ~50% should be even.
    assert 0.4 <= res.mu_hat <= 0.6
    assert res.hits == int(res.mu_hat * 1000)


# --- Validation ---

def test_empty_data_raises():
    def div_fn(x):
        return 0.0
    with pytest.raises(ValueError):
        EmpiricalSearch(div_fn, np.zeros((0, 3)), threshold=0.1, budget=10,
                        jitter_std=0.0, seed=0)


def test_budget_zero_raises():
    def div_fn(x):
        return 0.0
    data = np.zeros((5, 2))
    with pytest.raises(ValueError):
        EmpiricalSearch(div_fn, data, threshold=0.1, budget=0,
                        jitter_std=0.0, seed=0)


def test_negative_jitter_raises():
    def div_fn(x):
        return 0.0
    data = np.zeros((5, 2))
    with pytest.raises(ValueError):
        EmpiricalSearch(div_fn, data, threshold=0.1, budget=10,
                        jitter_std=-0.1, seed=0)


def test_one_d_data_raises():
    def div_fn(x):
        return 0.0
    with pytest.raises(ValueError):
        EmpiricalSearch(div_fn, np.zeros(10), threshold=0.1, budget=10,
                        jitter_std=0.0, seed=0)
