# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for auto-configuration probe (fhe_oracle.autoconfig)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.autoconfig import (
    AutoOracle,
    ProbeResult,
    Regime,
    _detect_plateau_cliff,
    classify_landscape,
)


# --- classify_landscape --------------------------------------------------


def test_saturation_detection():
    """Constant divergence should classify as FULL_DOMAIN_SATURATION."""
    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 1.0,
        bounds=[(-1.0, 1.0)] * 5,
        seed=1,
    )
    assert result.regime == Regime.FULL_DOMAIN_SATURATION
    assert isinstance(result, ProbeResult)
    assert result.probe_divergences.shape == (50,)
    assert "random" in result.recommendation["strategy"]


def test_standard_detection():
    """Smooth quadratic divergence should fall through to STANDARD."""
    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.001 * float(np.sum(np.asarray(x) ** 2)),
        bounds=[(-3.0, 3.0)] * 8,
        seed=1,
    )
    assert result.regime == Regime.STANDARD
    assert result.recommendation["strategy"] == "cma_es"


def test_preactivation_detection():
    """Divergence of form |Wx+b|^3 should detect PREACTIVATION."""
    W = np.array([[1.0, 2.0, -1.0]])
    b = np.array([0.5])

    def fhe_fn(x):
        z = float((W @ np.asarray(x) + b)[0])
        return abs(z) ** 3

    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * 3,
        W=W,
        b=b,
        seed=3,
    )
    assert result.regime == Regime.PREACTIVATION_DOMINATED
    assert result.recommendation["strategy"] == "preactivation"
    assert result.recommendation["preactivation_rank"] == 1


def test_plateau_detection():
    """Mostly flat with rare cliff -- either PLATEAU or STANDARD."""
    def fhe_fn(x):
        if np.max(np.abs(np.asarray(x))) > 2.9:
            return 100.0
        return 0.01

    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * 10,
        n_probes=200,
        seed=7,
    )
    assert result.regime in (Regime.PLATEAU_THEN_CLIFF, Regime.STANDARD)


def test_classify_requires_positive_n_probes():
    with pytest.raises(ValueError):
        classify_landscape(
            plaintext_fn=lambda x: 0.0,
            fhe_fn=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            n_probes=0,
        )


def test_probe_is_reproducible():
    """Same seed -> same divergences."""
    kwargs = dict(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: float(np.sum(np.asarray(x) ** 2)),
        bounds=[(-2.0, 2.0)] * 4,
        n_probes=30,
    )
    r1 = classify_landscape(seed=17, **kwargs)
    r2 = classify_landscape(seed=17, **kwargs)
    np.testing.assert_array_equal(r1.probe_divergences, r2.probe_divergences)


def test_all_zero_divergence_is_saturation():
    """Identical functions -> constant-zero div -> SATURATION (all
    probes 'exceed' 50% of zero max)."""
    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.0,
        bounds=[(-1.0, 1.0)] * 3,
        seed=1,
    )
    assert result.regime == Regime.FULL_DOMAIN_SATURATION


# --- AutoOracle --------------------------------------------------------


def test_auto_oracle_standard_runs():
    """AutoOracle should complete on a STANDARD landscape."""
    oracle = AutoOracle(
        plaintext_fn=lambda x: float(np.sum(np.asarray(x))),
        fhe_fn=lambda x: float(
            np.sum(np.asarray(x)) + 0.01 * float(np.sum(np.asarray(x) ** 2))
        ),
        bounds=[(-3.0, 3.0)] * 4,
        n_probes=20,
    )
    result = oracle.run(n_trials=100, seed=1)
    assert hasattr(result, "regime")
    assert hasattr(result, "strategy_used")
    assert result.max_error > 0


def test_auto_oracle_budget_accounting():
    """Caller asks for n_trials total -- probes are subtracted."""
    oracle = AutoOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: float(np.sum(np.asarray(x))),
        bounds=[(-1.0, 1.0)] * 3,
        n_probes=30,
    )
    result = oracle.run(n_trials=80, seed=5)
    # probe run uses 30 evals; remaining 50 go to the inner search.
    # OracleResult.n_trials only counts inner-oracle evals. CMA-ES may
    # overshoot by at most one population (~7 at d=3) because it only
    # checks the budget after emitting a full generation.
    assert result.n_trials <= 50 + 10


def test_auto_oracle_preactivation_dispatch():
    """With W, b and a preactivation-dominated landscape, dispatch to
    PreactivationOracle and report strategy_used='preactivation'."""
    rng = np.random.default_rng(0)
    d, k = 10, 1
    W = rng.normal(size=(k, d))
    b = rng.normal(size=(k,))

    def fhe_fn(x):
        z = float((W @ np.asarray(x) + b)[0])
        return abs(z) ** 3 / 10.0

    oracle = AutoOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        W=W,
        b=b,
        n_probes=30,
    )
    result = oracle.run(n_trials=80, seed=2)
    assert result.strategy_used == "preactivation"
    assert result.regime == Regime.PREACTIVATION_DOMINATED.value
    assert result.max_error > 0


def test_auto_oracle_saturation_dispatch():
    """A saturated landscape dispatches to random_only."""
    oracle = AutoOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 2.0,
        bounds=[(-1.0, 1.0)] * 3,
        n_probes=20,
    )
    result = oracle.run(n_trials=60, seed=1)
    assert result.strategy_used == "random_only"
    assert result.regime == Regime.FULL_DOMAIN_SATURATION.value


# --- _detect_plateau_cliff --------------------------------------------


def test_chebyshev_tenseal_detected():
    """Chebyshev TenSEAL probe stats trigger PLATEAU_THEN_CLIFF
    after the threshold relaxation. Plateau ~0.098, cliff ~0.3."""
    rng = np.random.RandomState(42)
    plateau = rng.normal(0.098, 0.01, 47)
    cliff = rng.uniform(0.25, 0.40, 3)
    divs = np.concatenate([plateau, cliff])
    assert _detect_plateau_cliff(divs), \
        "Should detect plateau-then-cliff on Chebyshev-like stats"


def test_standard_not_misclassified_as_plateau():
    """Heavy-tail exponential is STANDARD, not plateau-cliff."""
    rng = np.random.RandomState(42)
    divs = rng.exponential(0.5, 200)
    assert not _detect_plateau_cliff(divs), \
        "Exponential distribution should NOT be plateau-cliff"


def test_saturation_not_misclassified_as_plateau():
    """Saturated landscape (all values near 0.96) should NOT
    trigger plateau-cliff; CV is tiny but no cliff exists."""
    divs = np.random.RandomState(42).normal(0.96, 0.01, 50)
    assert not _detect_plateau_cliff(divs), \
        "Saturation should NOT be classified as plateau-cliff"


def test_plateau_cliff_gap_test():
    """Tight plateau + far max triggers gap test (Test C)."""
    rng = np.random.RandomState(0)
    plateau = rng.normal(1.0, 0.05, 95)
    cliff = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    divs = np.concatenate([plateau, cliff])
    assert _detect_plateau_cliff(divs)


def test_plateau_cliff_dominant_plateau():
    """Dominant-plateau test (B) catches narrow cliffs at small probe
    sizes when CV alone is too high to fire Test A."""
    rng = np.random.RandomState(1)
    plateau = rng.normal(1.0, 0.05, 45)
    cliff = np.array([20.0, 25.0, 30.0, 28.0, 22.0])
    divs = np.concatenate([plateau, cliff])
    assert _detect_plateau_cliff(divs)


def test_plateau_cliff_zero_noise_plateau():
    """A perfectly flat plateau with a few outliers fires (Test B)."""
    plateau = np.full(47, 0.098)
    cliff = np.array([0.30, 0.34, 0.36])
    divs = np.concatenate([plateau, cliff])
    assert _detect_plateau_cliff(divs)


def test_plateau_cliff_rejects_small_sample():
    """n<5 returns False (insufficient evidence)."""
    assert not _detect_plateau_cliff(np.array([1.0, 1.0, 100.0]))


def test_plateau_cliff_rejects_zero_landscape():
    """All-zero divergences -> False (median is 0)."""
    assert not _detect_plateau_cliff(np.zeros(50))


def test_classify_chebyshev_like_promotes_to_plateau():
    """End-to-end classify_landscape on a Chebyshev-like circuit
    (most probes near 0.1, rare cliff near 0.4) classifies as
    PLATEAU_THEN_CLIFF after the relaxed thresholds."""
    rng_state = {"i": 0}
    plateau_vals = np.concatenate([
        np.full(47, 0.098),
        np.array([0.30, 0.34, 0.36]),
    ])
    rng = np.random.RandomState(7)
    rng.shuffle(plateau_vals)

    def fhe_fn(x):
        idx = rng_state["i"] % plateau_vals.size
        rng_state["i"] += 1
        return float(plateau_vals[idx])

    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * 5,
        n_probes=50,
        seed=0,
        second_pass_probes=0,
    )
    assert result.regime == Regime.PLATEAU_THEN_CLIFF


def test_classify_second_pass_rescues_borderline():
    """Borderline first-pass with no cliff samples, rescued when the
    second pass finally hits the cliff. First 50 probes: all plateau
    plus a mild bump (max/med=3 but no separation). Next 50 probes:
    introduces the actual cliff."""
    rng = np.random.RandomState(11)
    first_pass = np.concatenate([
        rng.normal(0.10, 0.012, 49),  # tight plateau, CV~0.12
        np.array([0.30]),              # mild bump: max/med ~ 3 -> borderline
    ])
    second_pass = np.concatenate([
        rng.normal(0.10, 0.012, 47),
        np.array([0.45, 0.50, 0.42]),  # real cliff
    ])
    pool = np.concatenate([first_pass, second_pass])
    counter = {"i": 0}

    def fhe_fn(x):
        v = float(pool[counter["i"] % pool.size])
        counter["i"] += 1
        return v

    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * 4,
        n_probes=50,
        seed=0,
        second_pass_probes=50,
    )
    assert result.regime == Regime.PLATEAU_THEN_CLIFF


def test_classify_standard_stays_standard():
    """The structured-but-non-plateau LR-like landscape should
    still classify as STANDARD after the threshold relaxation."""
    result = classify_landscape(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.001 * float(np.sum(np.asarray(x) ** 2)),
        bounds=[(-3.0, 3.0)] * 8,
        n_probes=50,
        seed=1,
    )
    assert result.regime == Regime.STANDARD


def test_auto_oracle_rejects_undersized_budget():
    oracle = AutoOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.0,
        bounds=[(-1.0, 1.0)],
        n_probes=50,
    )
    with pytest.raises(ValueError):
        oracle.run(n_trials=40)
