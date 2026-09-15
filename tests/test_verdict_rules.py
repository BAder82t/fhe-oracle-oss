# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verdict rules: seeded reproducibility, search-time violations, class flips."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle


def _square(x):
    return float(np.sum(np.asarray(x) ** 2))


@pytest.mark.parametrize("kwargs", [{}, {"restarts": 1}])
def test_seed_zero_is_reproducible(kwargs):
    def run():
        oracle = FHEOracle(_square, lambda x: _square(x) * 1.01, input_dim=3,
                           input_bounds=[(-1.0, 1.0)] * 3, seed=0, **kwargs)
        return oracle.run(n_trials=60, threshold=10.0)

    a, b = run(), run()
    assert a.worst_input == b.worst_input
    assert a.max_error == b.max_error


@pytest.mark.parametrize("seed", range(10))
def test_violation_seen_during_search_fails(seed):
    # True error 0.009 sits below the 0.01 threshold; noise crosses it.
    rng = np.random.default_rng(seed)
    seen = []

    def fhe(x):
        value = _square(x) + 0.009 + rng.normal(0.0, 0.001)
        seen.append(abs(value - _square(x)))
        return value

    oracle = FHEOracle(_square, fhe, input_dim=2,
                       input_bounds=[(-1.0, 1.0)] * 2, seed=seed)
    result = oracle.run(n_trials=40, threshold=0.01)
    assert max(seen[:-1]) >= 0.01  # the scenario really crosses during search
    assert result.verdict == "FAIL"
    assert result.max_error >= 0.01
    assert result.search_max_error >= 0.01
    assert result.remeasured_error == pytest.approx(seen[-1])


def test_deterministic_pass_is_unchanged():
    oracle = FHEOracle(_square, _square, input_dim=2,
                       input_bounds=[(-1.0, 1.0)] * 2, seed=1)
    result = oracle.run(n_trials=30, threshold=1e-6)
    assert result.verdict == "PASS"
    assert result.search_max_error == 0.0
    assert result.class_flip is None


def _scores(x):
    s = 0.001 if x[0] >= 0 else -0.001
    return [0.5 + s, 0.5 - s]


def _flipped_scores(x):
    # Tiny absolute error, but the class flips in part of the box.
    p = _scores(x)
    return p[::-1] if x[1] > 0.5 else p


@pytest.mark.parametrize("mode", ["rank_inversion", "combined"])
def test_class_flip_fails_in_rank_modes(mode):
    oracle = FHEOracle(_scores, _flipped_scores, input_dim=2,
                       input_bounds=[(-1.0, 1.0)] * 2, seed=3,
                       multi_output=True, multi_output_mode=mode)
    result = oracle.run(n_trials=60, threshold=0.01)
    assert result.max_error < 0.01
    assert result.class_flip is True
    assert result.verdict == "FAIL"


def test_small_error_without_flip_passes_in_rank_mode():
    oracle = FHEOracle(_scores, lambda x: [v + 1e-4 for v in _scores(x)],
                       input_dim=2, input_bounds=[(-1.0, 1.0)] * 2, seed=3,
                       multi_output=True, multi_output_mode="rank_inversion")
    result = oracle.run(n_trials=60, threshold=0.01)
    assert result.class_flip is False
    assert result.verdict == "PASS"
