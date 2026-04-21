# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.hybrid.run_hybrid (A3)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.hybrid import HybridResult, run_hybrid


def _identity(x):
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _buggy_hot_zone(x):
    v = _identity(x)
    return v + (0.1 if v > 4.0 else 0.0)


# --- Oracle-only mode (data=None) ---

def test_oracle_only_mode_data_none():
    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=_buggy_hot_zone,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        threshold=1e-3,
        oracle_budget=200,
        oracle_seed=0,
        random_floor=0.3,
        data=None,
    )
    assert res.empirical_result is None
    assert res.source == "oracle"
    assert res.union_verdict == res.oracle_result.verdict
    assert res.max_error == res.oracle_result.max_error


def test_oracle_only_identity_to_direct_run():
    """run_hybrid(data=None, ...) must produce the same oracle result as FHEOracle.run()."""
    from fhe_oracle import FHEOracle

    direct = FHEOracle(
        plaintext_fn=_identity,
        fhe_fn=_buggy_hot_zone,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        seed=42,
        random_floor=0.3,
    ).run(n_trials=200, threshold=1e-3)

    hybrid = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=_buggy_hot_zone,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        threshold=1e-3,
        oracle_budget=200,
        oracle_seed=42,
        random_floor=0.3,
        data=None,
    )

    assert direct.max_error == hybrid.oracle_result.max_error
    assert direct.worst_input == hybrid.oracle_result.worst_input


# --- Union verdict logic ---

def test_union_fail_when_oracle_fails_empirical_passes():
    """Oracle FAILs (hot zone exists in box corners), empirical PASSes."""
    # Empirical data stays well inside ||x||^2 < 4 (no hot zone).
    data = np.zeros((20, 3))  # all zeros → divergence = 0.
    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=_buggy_hot_zone,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        threshold=1e-3,
        oracle_budget=200,
        oracle_seed=0,
        random_floor=0.3,
        data=data,
        empirical_budget=50,
        jitter_std=0.0,
    )
    assert res.oracle_result.verdict == "FAIL"
    assert res.empirical_result is not None
    assert res.empirical_result.verdict == "PASS"
    assert res.union_verdict == "FAIL"
    assert res.source == "oracle"


def test_union_fail_when_empirical_fails_oracle_passes():
    """Oracle PASSes (benign circuit), empirical FAILs (planted bug in data).

    Plaintext and FHE match except on inputs near a specific planted point.
    Since the oracle searches [-1, 1]^3 and the plant is at [10,10,10],
    oracle won't find it; empirical samples directly from data and will.
    """
    plant = np.array([10.0, 10.0, 10.0])
    data = np.tile(plant, (20, 1))

    def plain_fn(x):
        return float(np.sum(np.asarray(x) ** 2))

    def fhe_fn(x):
        xa = np.asarray(x)
        # Plant-local divergence. Oracle box is [-1, 1]^3 so it misses.
        if np.linalg.norm(xa - plant) < 5.0:
            return plain_fn(x) + 0.5
        return plain_fn(x)

    res = run_hybrid(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_fn,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        threshold=0.01,
        oracle_budget=100,
        oracle_seed=0,
        random_floor=0.0,
        data=data,
        empirical_budget=50,
        jitter_std=0.0,
    )
    assert res.oracle_result.verdict == "PASS"
    assert res.empirical_result is not None
    assert res.empirical_result.verdict == "FAIL"
    assert res.union_verdict == "FAIL"
    assert res.source == "empirical"


def test_union_pass_when_both_pass():
    """Benign circuit + benign data → both PASS, union PASS."""
    data = np.zeros((20, 3))
    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=_identity,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        threshold=1.0,
        oracle_budget=50,
        oracle_seed=0,
        random_floor=0.0,
        data=data,
        empirical_budget=30,
        jitter_std=0.0,
    )
    assert res.oracle_result.verdict == "PASS"
    assert res.empirical_result.verdict == "PASS"
    assert res.union_verdict == "PASS"


def test_union_fail_when_both_fail():
    data = np.tile([10.0, 10.0, 10.0], (20, 1))

    def fhe_bug(x):
        return float(np.sum(np.asarray(x) ** 2)) + 1.0   # always diverges by 1.0

    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=fhe_bug,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        threshold=0.01,
        oracle_budget=100,
        oracle_seed=0,
        random_floor=0.0,
        data=data,
        empirical_budget=30,
        jitter_std=0.0,
    )
    assert res.oracle_result.verdict == "FAIL"
    assert res.empirical_result.verdict == "FAIL"
    assert res.union_verdict == "FAIL"


# --- Budget forwarding ---

def test_separate_budgets_honoured():
    data = np.zeros((10, 3))
    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=_buggy_hot_zone,
        input_dim=3,
        input_bounds=[(-3.0, 3.0)] * 3,
        threshold=1e-3,
        oracle_budget=200,
        oracle_seed=0,
        random_floor=0.3,
        data=data,
        empirical_budget=300,
        jitter_std=0.0,
        empirical_seed=1,
    )
    # Oracle respects budget (possibly less on early-stop).
    assert res.oracle_result.n_trials <= 200 + 40
    # Empirical uses exact budget.
    assert res.empirical_result.n_trials == 300


# --- Source tracks larger max_error ---

def test_source_tracks_larger_max_error():
    """When empirical's max_error exceeds oracle's, source='empirical'."""
    plant = np.array([10.0, 10.0, 10.0])
    data = np.tile(plant, (5, 1))

    def fhe_fn(x):
        xa = np.asarray(x)
        if np.linalg.norm(xa - plant) < 5.0:
            return float(np.sum(xa ** 2)) + 99.0  # huge divergence
        return float(np.sum(xa ** 2))

    res = run_hybrid(
        plaintext_fn=_identity,
        fhe_fn=fhe_fn,
        input_dim=3,
        input_bounds=[(-1.0, 1.0)] * 3,
        threshold=0.01,
        oracle_budget=50,
        oracle_seed=0,
        random_floor=0.0,
        data=data,
        empirical_budget=20,
        jitter_std=0.0,
    )
    assert res.source == "empirical"
    assert res.max_error == res.empirical_result.max_error
    assert res.max_error >= 99.0
