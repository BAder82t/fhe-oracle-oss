# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional batch_fhe_fn: same search, same budget, strict error handling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fhe_oracle import EvaluationError, FHEOracle


def _plain(x):
    return float(np.sum(np.asarray(x) ** 2))


def _fhe(x):
    v = _plain(x)
    return v + 1e-3 * v * (10.0 if v > 1.5 else 1.0)


def _oracle(**kwargs):
    return FHEOracle(_plain, _fhe, input_dim=3,
                     input_bounds=[(-1.0, 1.0)] * 3, seed=4, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {},
    {"restarts": 1},
    {"random_floor": 0.3},
    {"diversity_injection": True},
])
def test_batch_matches_sequential_search(kwargs):
    batches = []

    def batch(xs):
        batches.append(len(xs))
        return [_fhe(x) for x in xs]

    seq = _oracle(**kwargs).run(n_trials=61, threshold=0.01)
    bat = _oracle(batch_fhe_fn=batch, **kwargs).run(n_trials=61, threshold=0.01)
    assert bat.worst_input == seq.worst_input
    assert bat.max_error == seq.max_error
    assert bat.verdict == seq.verdict
    assert bat.n_trials == seq.n_trials
    assert sum(batches) == bat.n_trials  # the re-measurement uses fhe_fn


def test_batch_multi_output_matches_sequential():
    def plain(x):
        return [x[0], -x[0]]

    def fhe(x):
        return [x[0] + 0.02 * (x[1] > 0), -x[0]]

    def run(**kw):
        return FHEOracle(plain, fhe, input_dim=2, input_bounds=[(-1.0, 1.0)] * 2,
                         seed=2, multi_output=True, **kw).run(n_trials=40, threshold=0.01)

    seq = run()
    bat = run(batch_fhe_fn=lambda xs: [fhe(x) for x in xs])
    assert (bat.worst_input, bat.max_error, bat.verdict, bat.class_flip) == (
        seq.worst_input, seq.max_error, seq.verdict, seq.class_flip)


@pytest.mark.parametrize("batch,match", [
    (lambda xs: (_ for _ in ()).throw(RuntimeError("pool died")), "pool died"),
    (lambda xs: [_fhe(x) for x in xs][:-1], "outputs for"),
    (lambda xs: [float("nan")] * len(xs), "finite"),
])
def test_batch_failures_abort(batch, match):
    with pytest.raises(EvaluationError, match=match):
        _oracle(batch_fhe_fn=batch).run(n_trials=30, threshold=0.01)


def test_batch_requires_fhe_fn_and_builtin_fitness():
    batch = lambda xs: [_fhe(x) for x in xs]  # noqa: E731
    with pytest.raises(ValueError, match="batch_fhe_fn"):
        FHEOracle(_plain, fitness=SimpleNamespace(score=lambda x: 0.0),
                  input_dim=2, batch_fhe_fn=batch)
    with pytest.raises(ValueError, match="batch_fhe_fn"):
        FHEOracle(_plain, adapter=SimpleNamespace(evaluate=_fhe),
                  input_dim=2, batch_fhe_fn=batch)
