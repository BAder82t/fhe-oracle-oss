# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for FHEOracle.shrink()."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle, OracleResult, ShrinkResult


def _square(x):
    return float(np.sum(np.asarray(x) ** 2))


def _hot_zone_bug(x):
    v = _square(x)
    return v + (0.5 if v > 2.0 else 0.0)


def _make_fail_oracle(dim=3, seed=1, n_trials=200):
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_hot_zone_bug,
        input_dim=dim,
        input_bounds=[(-2.0, 2.0)] * dim,
        seed=seed,
    )
    result = oracle.run(n_trials=n_trials, threshold=1e-3)
    assert result.verdict == "FAIL"
    return oracle, result


def test_shrink_reduces_norm_toward_reference():
    oracle, result = _make_fail_oracle()
    shrunk = oracle.shrink(result, max_evals=200)
    assert isinstance(shrunk, ShrinkResult)
    assert shrunk.shrunk_norm <= shrunk.original_norm
    assert shrunk.max_error >= result.threshold
    assert shrunk.n_evals <= 200
    assert shrunk.threshold == result.threshold
    assert shrunk.original_input == result.worst_input


def test_shrink_respects_max_evals_budget():
    oracle, result = _make_fail_oracle(dim=5, seed=2, n_trials=300)
    shrunk = oracle.shrink(result, max_evals=30)
    assert shrunk.n_evals <= 30


def test_shrink_requires_fail_result():
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        seed=0,
    )
    result = oracle.run(n_trials=50, threshold=1e-3)
    assert result.verdict == "PASS"
    with pytest.raises(ValueError):
        oracle.shrink(result)


def test_shrink_reference_dimension_mismatch_raises():
    oracle, result = _make_fail_oracle()
    with pytest.raises(ValueError):
        oracle.shrink(result, reference=[0.0, 0.0])  # wrong length (dim=3)


def test_shrink_uniformly_failing_landscape_collapses_to_reference():
    # divergence is 999 everywhere in bounds -> shrink can push all the
    # way to the reference point.
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 999.0,
        input_dim=2,
        input_bounds=[(-1.0, 1.0)] * 2,
        seed=0,
    )
    result = oracle.run(n_trials=20, threshold=1e-3)
    assert result.verdict == "FAIL"
    shrunk = oracle.shrink(result, reference=[0.0, 0.0], max_evals=200)
    assert shrunk.shrunk_norm == pytest.approx(0.0, abs=1e-3)
    assert shrunk.shrunk_input == pytest.approx([0.0, 0.0], abs=1e-3)


def test_shrink_witness_already_at_reference_is_noop():
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 999.0,
        input_dim=2,
        input_bounds=[(-1.0, 1.0)] * 2,
        seed=0,
    )
    manual_result = OracleResult(
        verdict="FAIL",
        max_error=999.0,
        worst_input=[0.0, 0.0],
        threshold=1e-3,
        n_trials=0,
        elapsed_seconds=0.0,
    )
    shrunk = oracle.shrink(manual_result, reference=[0.0, 0.0], max_evals=50)
    assert shrunk.original_norm == 0.0
    assert shrunk.shrunk_norm == 0.0
    assert shrunk.shrunk_input == [0.0, 0.0]
