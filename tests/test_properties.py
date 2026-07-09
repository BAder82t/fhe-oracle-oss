# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.properties."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.properties import AdditivityFitness, ScalarLinearityFitness


def _additive_fn(x):
    return float(np.sum(np.asarray(x)))


def _non_additive_fn(x):
    # constant offset breaks additivity: f(a+b) != f(a)+f(b) by exactly 1.
    return float(np.sum(np.asarray(x))) + 1.0


def test_additivity_fitness_zero_for_additive_fn():
    fitness = AdditivityFitness(_additive_fn, dim=3)
    a = [1.0, 2.0, 3.0]
    b = [0.5, -1.0, 2.0]
    assert fitness.score(a + b) == pytest.approx(0.0, abs=1e-9)


def test_additivity_fitness_nonzero_for_non_additive_fn():
    fitness = AdditivityFitness(_non_additive_fn, dim=3)
    a = [1.0, 2.0, 3.0]
    b = [0.5, -1.0, 2.0]
    assert fitness.score(a + b) == pytest.approx(1.0, abs=1e-9)


def test_additivity_fitness_swallows_exceptions():
    def raises(x):
        raise RuntimeError("boom")

    fitness = AdditivityFitness(raises, dim=2)
    assert fitness.score([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_additivity_fitness_handles_vector_output():
    # additive vector-valued fn: elementwise sum-doubling.
    def vector_fn(x):
        arr = np.asarray(x)
        return [float(arr[0] * 2), float(arr[1] * 2)]

    fitness = AdditivityFitness(vector_fn, dim=2)
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    assert fitness.score(a + b) == pytest.approx(0.0, abs=1e-9)

    def non_additive_vector_fn(x):
        arr = np.asarray(x)
        return [float(arr[0] * 2) + 1.0, float(arr[1] * 2)]

    fitness2 = AdditivityFitness(non_additive_vector_fn, dim=2)
    assert fitness2.score(a + b) == pytest.approx(1.0, abs=1e-9)


def test_oracle_search_finds_additivity_violation():
    fitness = AdditivityFitness(_non_additive_fn, dim=2)
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        input_dim=4,  # 2 * dim (a, b packed)
        input_bounds=[(-2.0, 2.0)] * 4,
        fitness=fitness,
        seed=0,
    )
    result = oracle.run(n_trials=100, threshold=1e-3)
    assert result.verdict == "FAIL"
    assert result.max_error == pytest.approx(1.0, abs=1e-6)


def test_oracle_search_does_not_flag_truly_additive_fn():
    fitness = AdditivityFitness(_additive_fn, dim=2)
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        input_dim=4,
        input_bounds=[(-2.0, 2.0)] * 4,
        fitness=fitness,
        seed=0,
    )
    result = oracle.run(n_trials=100, threshold=1e-3)
    assert result.verdict == "PASS"


def _linear_fn(x):
    return float(np.sum(np.asarray(x)))


def _affine_fn(x):
    return float(np.sum(np.asarray(x))) + 1.0


def test_scalar_linearity_fitness_zero_for_linear_fn():
    fitness = ScalarLinearityFitness(_linear_fn, dim=3)
    x = [1.0, 2.0, 3.0]
    c = 2.0
    assert fitness.score(x + [c]) == pytest.approx(0.0, abs=1e-9)


def test_scalar_linearity_fitness_nonzero_for_affine_fn():
    fitness = ScalarLinearityFitness(_affine_fn, dim=3)
    x = [1.0, 2.0, 3.0]
    c = 2.0
    # f(cx) = sum(cx)+1, c*f(x) = c*sum(x)+c -> diff = |1 - c| = 1.0
    assert fitness.score(x + [c]) == pytest.approx(1.0, abs=1e-9)


def test_scalar_linearity_fitness_clips_c_to_bounds():
    fitness = ScalarLinearityFitness(_linear_fn, dim=2, c_bounds=(-1.0, 1.0))
    # c=5.0 should clip to 1.0 -> diff=0 for a linear fn regardless.
    assert fitness.score([1.0, 1.0, 5.0]) == pytest.approx(0.0, abs=1e-9)


def test_oracle_search_finds_linearity_violation():
    fitness = ScalarLinearityFitness(_affine_fn, dim=2, c_bounds=(-3.0, 3.0))
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        input_dim=3,  # dim + 1 (x..., c)
        input_bounds=[(-2.0, 2.0), (-2.0, 2.0), (-3.0, 3.0)],
        fitness=fitness,
        seed=0,
    )
    result = oracle.run(n_trials=100, threshold=1e-3)
    assert result.verdict == "FAIL"
    assert result.max_error > 0.5
