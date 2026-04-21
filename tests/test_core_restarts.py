# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for FHEOracle IPOP/BIPOP restarts (A2)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle


def _square(x):
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _mock_fhe_hot_zone(x):
    v = _square(x)
    return v + (0.1 if v > 4.0 else 0.0)


def _constant_flat(x):
    """Constant circuit — forces stall on every generation."""
    return 0.0


# --- restarts=0 identity (regression guard) ---

def test_restarts_zero_identity_no_random_floor():
    """restarts=0, random_floor=0 → bit-identical to A1 baseline."""
    oracle_a = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square,
        input_dim=4, input_bounds=[(-1.0, 1.0)] * 4,
        seed=42, restarts=0,
    )
    res_a = oracle_a.run(n_trials=60, threshold=1e-6)

    oracle_b = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square,
        input_dim=4, input_bounds=[(-1.0, 1.0)] * 4,
        seed=42,
    )
    res_b = oracle_b.run(n_trials=60, threshold=1e-6)

    assert res_a.max_error == res_b.max_error
    assert res_a.worst_input == res_b.worst_input
    assert res_a.n_restarts_used == 0
    assert res_b.n_restarts_used == 0


def test_restarts_zero_identity_with_random_floor():
    """restarts=0, rho=0.3 → bit-identical to A1's rho=0.3 path."""
    oracle_a = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=0, random_floor=0.3, restarts=0,
    )
    res_a = oracle_a.run(n_trials=500, threshold=1e-3)

    oracle_b = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=0, random_floor=0.3,
    )
    res_b = oracle_b.run(n_trials=500, threshold=1e-3)

    assert res_a.max_error == res_b.max_error
    assert res_a.worst_input == res_b.worst_input


# --- Budget accounting ---

def test_restarts_2_budget_accounting_no_rho():
    """B=500, ρ=0.0, restarts=2 → total ≤ B + max_popsize overshoot."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=1, restarts=2,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    # Budget contract: never exceed B by more than one popsize generation.
    # Worst case: restarts=2 with doubling → popsize up to ~20 for d=3.
    assert result.n_trials <= 500 + 40
    assert result.n_restarts_used <= 2


def test_restarts_2_with_random_floor():
    """B=500, ρ=0.3, restarts=2 → random=150, cma ≤ 350 + overshoot."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=2, random_floor=0.3, restarts=2,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    assert result.coverage_certificate is not None
    assert result.coverage_certificate.budget_rand == 150
    assert result.n_trials <= 500 + 40


# --- Stall detection triggers restart ---

def test_stall_triggers_restart():
    """Flat fitness forces stall on every generation → all restarts fire."""
    oracle = FHEOracle(
        plaintext_fn=_constant_flat, fhe_fn=_constant_flat,
        input_dim=3, input_bounds=[(-1.0, 1.0)] * 3,
        seed=5, restarts=3, stall_generations=2, stall_tol=1e-9,
    )
    result = oracle.run(n_trials=500, threshold=1e-6)
    # With a flat circuit, every generation improves best_score by < tol
    # so stall fires quickly and all 3 restarts should trigger.
    assert result.n_restarts_used >= 2  # tolerate budget cutoff on last


# --- Global best preserved across restarts ---

def test_global_best_preserved_across_restarts():
    """Restart loop must track the global max across all runs."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=0, restarts=3,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    # Some run (first or restart) should find the hot-zone defect.
    assert result.verdict == "FAIL"
    assert result.max_error >= 0.09


# --- Composition with A1 warm-start ---

def test_composition_a1_warmstart_plus_restarts():
    """ρ=0.3 + restarts=2: random floor runs first, CMA-ES restarts."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=0, random_floor=0.3, warm_start=True, restarts=2,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    assert result.coverage_certificate is not None
    assert result.coverage_certificate.budget_rand == 150
    assert result.n_restarts_used <= 2
    assert result.verdict == "FAIL"


# --- Validation ---

def test_negative_restarts_raises():
    with pytest.raises(ValueError):
        FHEOracle(
            plaintext_fn=_square, fhe_fn=_square, input_dim=3,
            input_bounds=[(-1.0, 1.0)] * 3, restarts=-1,
        )


def test_popsize_factor_under_one_raises():
    with pytest.raises(ValueError):
        FHEOracle(
            plaintext_fn=_square, fhe_fn=_square, input_dim=3,
            input_bounds=[(-1.0, 1.0)] * 3, restarts=1,
            restart_popsize_factor=0.5,
        )


def test_stall_generations_zero_raises():
    with pytest.raises(ValueError):
        FHEOracle(
            plaintext_fn=_square, fhe_fn=_square, input_dim=3,
            input_bounds=[(-1.0, 1.0)] * 3, restarts=1,
            stall_generations=0,
        )


def test_restarts_require_bounds():
    """restarts > 0 without bounds raises at run() time."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_square, input_dim=3,
        seed=0, restarts=2,
    )
    with pytest.raises(ValueError, match="restarts"):
        oracle.run(n_trials=100, threshold=1.0)


# --- BIPOP ---

def test_bipop_runs_without_error():
    """BIPOP loop completes; n_restarts_used is within spec."""
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=7, restarts=3, bipop=True,
    )
    result = oracle.run(n_trials=1000, threshold=1e-3)
    assert result.n_restarts_used <= 3
    assert result.verdict == "FAIL"


# --- restarts=0 certificate still works ---

def test_restarts_zero_does_not_break_certificate():
    oracle = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_hot_zone,
        input_dim=3, input_bounds=[(-3.0, 3.0)] * 3,
        seed=0, random_floor=0.3, restarts=0,
    )
    result = oracle.run(n_trials=500, threshold=0.05)
    assert result.coverage_certificate is not None
    assert result.n_restarts_used == 0
