# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.differential."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import EvaluationError

from fhe_oracle.adapters.base import FHEAdapter
from fhe_oracle.differential import CrossAdapterFitness, differential_test


class _FakeAdapter(FHEAdapter):
    """Plain-Python stand-in for a real FHE library, for testing.

    ``encrypt``/``decrypt`` are identity pass-throughs; ``run_fhe_program``
    applies ``fn`` to the "ciphertext" (a plain list here). This lets the
    differential-testing harness be exercised without a real FHE library,
    the same way tests/test_core.py uses plain functions instead of FHE.
    """

    def __init__(self, fn):
        self._fn = fn

    def encrypt(self, x):
        return list(x)

    def decrypt(self, ciphertext):
        return list(ciphertext)

    def run_fhe_program(self, ciphertext):
        return [self._fn(ciphertext)]

    def get_noise_budget(self, ciphertext):
        return 0.0

    def get_mult_depth_used(self, ciphertext):
        return 0

    def get_scheme_name(self):
        return "fake"


def _square_sum(x):
    return float(np.sum(np.asarray(x) ** 2))


def test_differential_test_finds_disagreement():
    def buggy(x):
        v = _square_sum(x)
        return v + (0.5 if v > 2.0 else 0.0)

    a = _FakeAdapter(_square_sum)
    b = _FakeAdapter(buggy)

    result = differential_test(
        a, b, input_dim=3, input_bounds=[(-2.0, 2.0)] * 3, n_trials=200,
        threshold=1e-3, seed=1,
    )
    assert result.verdict == "FAIL"
    assert result.max_error >= 0.4


def test_differential_test_identical_adapters_pass():
    a = _FakeAdapter(_square_sum)
    b = _FakeAdapter(_square_sum)

    result = differential_test(
        a, b, input_dim=3, input_bounds=[(-2.0, 2.0)] * 3, n_trials=100,
        threshold=1e-6, seed=0,
    )
    assert result.verdict == "PASS"
    assert result.max_error < 1e-9


def test_cross_adapter_fitness_direct():
    a = _FakeAdapter(lambda x: 1.0)
    b = _FakeAdapter(lambda x: 3.0)
    fitness = CrossAdapterFitness(a, b)
    assert fitness.score([0.0, 0.0]) == 2.0


def test_cross_adapter_fitness_rejects_mismatched_output_lengths():
    a = _FakeAdapter(lambda x: [1.0, 2.0, 3.0])
    b = _FakeAdapter(lambda x: [1.0, 5.0])  # shorter output
    fitness = CrossAdapterFitness(a, b)
    with pytest.raises(EvaluationError, match="shape mismatch"):
        fitness.score([0.0])


def test_cross_adapter_fitness_rejects_exceptions():
    def raises(x):
        raise RuntimeError("boom")

    a = _FakeAdapter(raises)
    b = _FakeAdapter(lambda x: 1.0)
    fitness = CrossAdapterFitness(a, b)
    with pytest.raises(EvaluationError, match="boom"):
        fitness.score([0.0])
