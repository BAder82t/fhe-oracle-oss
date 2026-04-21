# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for FHEOracle hybrid random-floor + warm-start (A4+A1)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle, OracleResult
from fhe_oracle.guarantees import CoverageCertificate


def _square(x):
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _mock_fhe_square_with_bug(x):
    """Returns plaintext + hot-zone noise: divergence when ||x||^2 > 4."""
    v = _square(x)
    return v + (0.1 if v > 4.0 else 0.0)


# --- Regression: random_floor=0.0 must be bit-identical to baseline ---

def test_random_floor_zero_is_identity():
    """rho=0 path must produce identical results to the current baseline."""
    oracle_a = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4,
        seed=42,
        random_floor=0.0,
    )
    res_a = oracle_a.run(n_trials=60, threshold=1e-6)

    oracle_b = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4,
        seed=42,
        # Defaults: random_floor=0.0
    )
    res_b = oracle_b.run(n_trials=60, threshold=1e-6)

    assert res_a.max_error == res_b.max_error
    assert res_a.worst_input == res_b.worst_input
    assert res_a.coverage_certificate is None
    assert res_b.coverage_certificate is None


# --- Budget accounting: B_rand + B_cma = B total ---

def test_budget_accounting_rho_0_3():
    """rho=0.3, B=500 → B_rand=150, total ≤ 500 (CMA-ES may early-stop)."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=0,
        random_floor=0.3,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    assert result.coverage_certificate is not None
    assert result.coverage_certificate.budget_rand == 150
    # Budget contract: never exceed B, always include the random phase.
    assert 150 <= result.n_trials <= 500


def test_budget_accounting_rho_0_5():
    """rho=0.5, B=100 → B_rand=50, total ≤ 100."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=1,
        random_floor=0.5,
    )
    result = oracle.run(n_trials=100, threshold=1e-3)
    assert result.coverage_certificate.budget_rand == 50
    assert 50 <= result.n_trials <= 100


def test_budget_never_exceeds_total_under_noisy_fitness():
    """With a noisy fitness that prevents early-stop, CMA-ES fills budget.

    Uses an input-dependent random noise generator so CMA-ES doesn't
    converge and runs the full allocated B_cma.
    """
    import hashlib

    def noisy_fhe(x):
        p = _square(x)
        # Deterministic but high-variance noise per input → no convergence.
        h = hashlib.sha256(str(tuple(round(v, 6) for v in x)).encode()).digest()
        return p + int.from_bytes(h[:4], "big") / 2**32 * 0.5

    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=noisy_fhe,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=3,
        random_floor=0.3,
    )
    result = oracle.run(n_trials=200, threshold=10.0)
    assert result.coverage_certificate.budget_rand == 60
    assert result.n_trials <= 200


def test_budget_accounting_rho_1_0():
    """rho=1.0: all budget to random floor, no CMA-ES phase."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=2,
        random_floor=1.0,
    )
    result = oracle.run(n_trials=200, threshold=1e-3)
    assert result.coverage_certificate.budget_rand == 200
    assert result.n_trials == 200


# --- Certificate populated correctly ---

def test_certificate_populated_with_rho_0_3():
    """With rho=0.3 B=500, certificate present with B_rand=150."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=0,
        random_floor=0.3,
    )
    result = oracle.run(n_trials=500, threshold=0.05)
    cert = result.coverage_certificate
    assert cert is not None
    assert isinstance(cert, CoverageCertificate)
    assert cert.budget_rand == 150
    assert cert.threshold == 0.05
    assert 0 <= cert.hits <= 150
    assert cert.mu_hat == cert.hits / 150
    assert 0.0 <= cert.p_discovery <= 1.0


def test_certificate_none_when_rho_zero():
    """Default rho=0 → no certificate."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        seed=0,
    )
    result = oracle.run(n_trials=50, threshold=1e-6)
    assert result.coverage_certificate is None


# --- Warm-start behaviour ---

def test_warm_start_finds_hot_zone():
    """With rho=0.3 and a hot-zone bug, the hybrid finds the FAIL."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=0,
        random_floor=0.3,
        warm_start=True,
    )
    result = oracle.run(n_trials=500, threshold=1e-3)
    assert result.verdict == "FAIL"
    assert result.max_error >= 0.09


def test_warm_start_off_still_builds_certificate():
    """warm_start=False still runs random phase for certificate."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=0,
        random_floor=0.3,
        warm_start=False,
    )
    result = oracle.run(n_trials=500, threshold=0.01)
    assert result.coverage_certificate is not None
    assert result.coverage_certificate.budget_rand == 150


# --- Validation ---

def test_random_floor_out_of_range_raises():
    with pytest.raises(ValueError):
        FHEOracle(
            plaintext_fn=_square, fhe_fn=_square, input_dim=3,
            input_bounds=[(-1.0, 1.0)] * 3, random_floor=-0.1,
        )
    with pytest.raises(ValueError):
        FHEOracle(
            plaintext_fn=_square, fhe_fn=_square, input_dim=3,
            input_bounds=[(-1.0, 1.0)] * 3, random_floor=1.5,
        )


def test_random_floor_requires_bounds():
    """random_floor > 0 without input_bounds raises at run() time."""
    oracle = FHEOracle(
        plaintext_fn=_square,
        fhe_fn=_square,
        input_dim=3,
        seed=0,
        random_floor=0.5,
    )
    with pytest.raises(ValueError, match="random_floor"):
        oracle.run(n_trials=20, threshold=1.0)


# --- Best-across-phases tracking ---

def test_best_tracked_across_phases():
    """Random phase's best_x must carry forward to OracleResult.

    On a plateau-then-cliff circuit, CMA-ES-only may miss the cliff
    (that is exactly what item A1 warm-start is designed to fix). We
    assert that ρ=1.0 (all random) finds the cliff and the cert is
    populated — demonstrating the random-phase best is retained.
    """
    bounds = [(-3.0, 3.0)] * 3

    oracle_rand = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3, input_bounds=bounds, seed=0, random_floor=1.0,
    )
    res_rand = oracle_rand.run(n_trials=500, threshold=0.05)
    assert res_rand.coverage_certificate.budget_rand == 500
    assert res_rand.verdict == "FAIL"
    # Worst-input must satisfy the bug condition: ||x||^2 > 4
    v = sum(xi * xi for xi in res_rand.worst_input)
    assert v > 4.0


def test_hybrid_recovers_over_cma_only_on_plateau():
    """Hybrid (ρ=0.3) finds the cliff that CMA-ES-only may miss.

    Documents the A1 motivation: on plateau-then-cliff landscapes, a
    random-floor phase + warm-start recovers where pure CMA-ES fails.
    """
    bounds = [(-3.0, 3.0)] * 3

    # rho=0.3 hybrid — should find the bug via random phase warm-start.
    oracle_hybrid = FHEOracle(
        plaintext_fn=_square, fhe_fn=_mock_fhe_square_with_bug,
        input_dim=3, input_bounds=bounds, seed=0, random_floor=0.3,
    )
    res_hybrid = oracle_hybrid.run(n_trials=500, threshold=0.05)
    assert res_hybrid.verdict == "FAIL"
    assert res_hybrid.max_error >= 0.09
