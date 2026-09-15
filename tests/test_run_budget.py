# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""run() must not evaluate more than n_trials; result labels its mode."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle


def _square(x):
    return float(np.sum(np.asarray(x) ** 2))


@pytest.mark.parametrize("kwargs", [
    {},
    {"restarts": 2},
    {"random_floor": 0.3},
    {"diversity_injection": True},
    {"separable": True},
    {"use_heuristic_seeds": True},
])
@pytest.mark.parametrize("n_trials", [50, 61])
def test_run_never_exceeds_n_trials(kwargs, n_trials):
    # d=3 gives popsize 7, which divides neither budget.
    calls = 0

    def fhe(x):
        nonlocal calls
        calls += 1
        return _square(x) + 1e-6

    oracle = FHEOracle(_square, fhe, input_dim=3,
                       input_bounds=[(-1.0, 1.0)] * 3, seed=0, **kwargs)
    result = oracle.run(n_trials=n_trials, threshold=1.0)
    assert result.n_trials <= n_trials
    assert calls <= n_trials + 1  # plus the final re-measurement


@pytest.mark.parametrize("dim,n_trials,kwargs", [
    (1, 1, {}),
    (3, 5, {"diversity_injection": True}),
    (13, 9, {"separable": True}),
    (30, 15, {"random_floor": 0.3}),
    (30, 23, {"use_heuristic_seeds": True}),
    (30, 60, {"restarts": 2}),
])
def test_run_budget_edge_cases(dim, n_trials, kwargs):
    # Budgets below one generation, restarts and injected seeds overshot.
    calls = 0

    def fhe(x):
        nonlocal calls
        calls += 1
        return _square(x) + 1e-6

    oracle = FHEOracle(_square, fhe, input_dim=dim,
                       input_bounds=[(-1.0, 1.0)] * dim, seed=0, **kwargs)
    result = oracle.run(n_trials=n_trials, threshold=1.0)
    assert result.n_trials <= n_trials
    assert calls <= n_trials + 1


def test_scheme_label_names_fhe_fn_mode():
    oracle = FHEOracle(_square, _square, input_dim=2,
                       input_bounds=[(-1.0, 1.0)] * 2, seed=0)
    assert oracle.run(n_trials=10, threshold=1.0).scheme == "fhe_fn"
