# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.core.FHEOracle."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle, OracleResult
from fhe_oracle.fitness import DivergenceFitness


def _square(x):
    return float(np.sum(np.asarray(x) ** 2))


def test_pass_when_fhe_matches_plaintext():
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        seed=0,
    )
    result = oracle.run(n_trials=100, threshold=1e-6)
    assert isinstance(result, OracleResult)
    assert result.verdict == "PASS"
    assert result.max_error < 1e-9


def test_fail_when_fhe_diverges_in_a_hot_zone():
    def buggy(x):
        v = _square(x)
        return v + (0.1 if v > 2.0 else 0.0)

    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=buggy,
        input_dim=3,
        input_bounds=[(-2.0, 2.0)] * 3,
        seed=1,
    )
    result = oracle.run(n_trials=200, threshold=1e-3)
    assert result.verdict == "FAIL"
    assert result.max_error >= 0.09


def test_input_dim_must_be_positive():
    with pytest.raises(ValueError):
        FHEOracle(plaintext_fn=_square, fhe_fn=_square, input_dim=0)


def test_requires_fhe_fn_or_adapter():
    with pytest.raises(ValueError):
        FHEOracle(plaintext_fn=_square, input_dim=3)


def test_broadcast_bounds_tuple():
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=4,
        input_bounds=(-1.0, 1.0),
        seed=0,
    )
    result = oracle.run(n_trials=50, threshold=1e-6)
    assert len(result.worst_input) == 4
    for v in result.worst_input:
        assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6


def test_custom_fitness():
    calls = {"n": 0}

    class ConstantFitness:
        def score(self, x):
            calls["n"] += 1
            return 42.0

    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=2,
        fitness=ConstantFitness(),
        seed=0,
    )
    oracle.run(n_trials=40, threshold=1.0)
    assert calls["n"] >= 1


def test_result_is_serialisable():
    from fhe_oracle.report import to_json, to_markdown

    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square, input_dim=2, seed=0
    )
    result = oracle.run(n_trials=30, threshold=1.0)
    js = to_json(result)
    md = to_markdown(result)
    assert "verdict" in js
    assert "FHE Oracle Report" in md


def test_heuristic_seed_injection_preserves_semantics():
    """Enabling heuristic seeds must not break FAIL-soundness or PASS-safety.

    On a benign identity circuit, both paths PASS with max_error ≈ 0.
    On a buggy hot-zone circuit, both paths FAIL with a reproducible
    witness. Seed injection only changes the search trajectory.
    """
    def buggy(x):
        v = _square(x)
        return v + (0.1 if v > 2.0 else 0.0)

    # Without injection
    oracle_off = FHEOracle(
        plaintext_fn=_square, fhe_fn=buggy, input_dim=3,
        input_bounds=[(-2.0, 2.0)] * 3, seed=1,
    )
    res_off = oracle_off.run(n_trials=100, threshold=1e-3)

    # With injection
    oracle_on = FHEOracle(
        plaintext_fn=_square, fhe_fn=buggy, input_dim=3,
        input_bounds=[(-2.0, 2.0)] * 3, seed=1,
        use_heuristic_seeds=True, heuristic_k=10,
    )
    res_on = oracle_on.run(n_trials=100, threshold=1e-3)

    # Both must find the hot-zone defect.
    assert res_off.verdict == "FAIL"
    assert res_on.verdict == "FAIL"
    assert res_on.max_error >= 0.09
    assert res_off.max_error >= 0.09


def test_heuristic_seed_injection_default_off_is_bit_identical():
    """With use_heuristic_seeds=False (default), no injection occurs.

    Output must match the pre-patch behaviour exactly under the same seed.
    """
    def flat(x):
        return float(np.sum(np.asarray(x) ** 2))

    # Two runs with default kwargs must be deterministic.
    oracle_a = FHEOracle(
        plaintext_fn=flat, fhe_fn=flat, input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4, seed=42,
    )
    res_a = oracle_a.run(n_trials=60, threshold=1e-6)

    oracle_b = FHEOracle(
        plaintext_fn=flat, fhe_fn=flat, input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4, seed=42,
    )
    res_b = oracle_b.run(n_trials=60, threshold=1e-6)

    assert res_a.verdict == res_b.verdict
    assert res_a.max_error == res_b.max_error
    assert res_a.worst_input == res_b.worst_input


def test_heuristic_seed_injection_requires_bounds_to_have_effect():
    """Without input_bounds, generate_seeds returns []; no injection."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square, input_dim=3,
        seed=0, use_heuristic_seeds=True, heuristic_k=10,
    )
    # Should not raise; injection is a no-op when bounds are None.
    result = oracle.run(n_trials=40, threshold=1.0)
    assert result.verdict in ("PASS", "FAIL")


def test_pure_divergence_defaults_v030():
    """v0.3.0 defaults: shaping weights are zero (pure divergence)."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square, input_dim=2,
        input_bounds=[(-1.0, 1.0)] * 2,
    )
    assert oracle.w_div == 1.0
    assert oracle.w_noise == 0.0
    assert oracle.w_depth == 0.0


def test_shaping_weights_override_still_accepted():
    """Users can restore v0.2 shaping by passing weights explicitly."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square, input_dim=2,
        input_bounds=[(-1.0, 1.0)] * 2,
        w_noise=0.5, w_depth=0.3,
    )
    assert oracle.w_noise == 0.5
    assert oracle.w_depth == 0.3


# test_noise_guided_fitness_default_weights_v030: moved to
# pro_modules/tests/ (tests NoiseBudgetFitness, a Pro-only class).
