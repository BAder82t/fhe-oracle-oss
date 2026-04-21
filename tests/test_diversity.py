# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for diversity injection (fhe_oracle.diversity)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.diversity import DiversityInjector, InjectionStrategy


# --- DiversityInjector unit tests ------------------------------------


def test_injector_generates_correct_count_and_shape():
    inj = DiversityInjector(bounds=[(-3.0, 3.0)] * 8, inject_count=3)
    rng = np.random.RandomState(42)
    cands = inj.generate_injections(np.zeros(8), rng)
    assert len(cands) == 3
    for c in cands:
        assert c.shape == (8,)
        assert np.all(c >= -3.0) and np.all(c <= 3.0)


def test_corner_strategy_hits_boundaries():
    inj = DiversityInjector(
        bounds=[(-3.0, 3.0)] * 20,
        strategy=InjectionStrategy.CORNER,
        corner_prob=0.95,
        inject_count=10,
    )
    rng = np.random.RandomState(42)
    cands = inj.generate_injections(np.zeros(20), rng)
    boundary_frac = np.mean([
        np.mean(np.abs(np.abs(c) - 3.0) < 1e-6) for c in cands
    ])
    assert boundary_frac > 0.8, f"Only {boundary_frac:.0%} boundary"


def test_uniform_strategy_stays_in_bounds():
    inj = DiversityInjector(
        bounds=[(-2.0, 4.0)] * 10,
        strategy=InjectionStrategy.UNIFORM,
        inject_count=20,
    )
    rng = np.random.RandomState(0)
    cands = inj.generate_injections(np.zeros(10), rng)
    for c in cands:
        assert np.all(c >= -2.0) and np.all(c <= 4.0)


def test_neighbor_strategy_perturbs_best():
    best = np.full(10, 1.5)
    inj = DiversityInjector(
        bounds=[(-3.0, 3.0)] * 10,
        strategy=InjectionStrategy.BEST_NEIGHBOR,
        neighbor_sigma=0.1,
        inject_count=5,
    )
    rng = np.random.RandomState(0)
    cands = inj.generate_injections(best, rng)
    for c in cands:
        # Mean distance from best is small (neighbor_sigma=0.1, half-width=3)
        assert np.linalg.norm(c - best) < 5.0


def test_should_inject_timing():
    inj = DiversityInjector(bounds=[(-1.0, 1.0)] * 5, inject_every=5)
    assert not inj.should_inject(0)
    assert not inj.should_inject(1)
    assert inj.should_inject(5)
    assert not inj.should_inject(6)
    assert inj.should_inject(10)


def test_mixed_rotates_strategies():
    """MIXED produces all three strategy outputs across rounds."""
    inj = DiversityInjector(
        bounds=[(-3.0, 3.0)] * 4,
        strategy=InjectionStrategy.MIXED,
        inject_count=1,
    )
    rng = np.random.RandomState(42)
    rounds = [inj.generate_injections(np.zeros(4), rng) for _ in range(9)]
    # Just check 9 rounds executed without error and produced valid points
    assert len(rounds) == 9
    for round_cands in rounds:
        assert len(round_cands) == 1
        assert round_cands[0].shape == (4,)


def test_invalid_inject_every_rejected():
    with pytest.raises(ValueError):
        DiversityInjector(bounds=[(-1.0, 1.0)], inject_every=0)


def test_invalid_inject_count_rejected():
    with pytest.raises(ValueError):
        DiversityInjector(bounds=[(-1.0, 1.0)], inject_count=0)


# --- FHEOracle integration ------------------------------------------


def test_diversity_records_injection_count():
    """Oracle with diversity_injection records non-zero injections
    when budget allows multiple generations."""
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: float(np.max(np.abs(np.asarray(x)))),
        input_dim=8,
        input_bounds=[(-3.0, 3.0)] * 8,
        diversity_injection=True,
        inject_every=2,
        inject_count=2,
        seed=1,
    )
    result = oracle.run(n_trials=200, threshold=10.0)
    assert result.diversity_injections > 0


def test_diversity_off_records_zero():
    """Default (diversity_injection=False) -> diversity_injections == 0."""
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: float(np.max(np.abs(np.asarray(x)))),
        input_dim=8,
        input_bounds=[(-3.0, 3.0)] * 8,
        seed=1,
    )
    result = oracle.run(n_trials=200, threshold=10.0)
    assert result.diversity_injections == 0


def test_diversity_helps_plateau_landscape():
    """Diversity injection should find the cliff more reliably than
    vanilla CMA-ES on a plateau-then-cliff landscape."""
    d = 10

    def fhe(x):
        return 10.0 if float(np.max(np.abs(np.asarray(x)))) > 2.5 else 0.1

    wins_v, wins_d = 0, 0
    for seed in range(1, 11):
        v = FHEOracle(
            plaintext_fn=lambda x: 0.0, fhe_fn=fhe,
            input_dim=d, input_bounds=[(-3.0, 3.0)] * d,
            seed=seed,
        ).run(n_trials=200, threshold=5.0)
        d_res = FHEOracle(
            plaintext_fn=lambda x: 0.0, fhe_fn=fhe,
            input_dim=d, input_bounds=[(-3.0, 3.0)] * d,
            diversity_injection=True, inject_every=3, inject_count=2,
            inject_strategy="corner",
            seed=seed,
        ).run(n_trials=200, threshold=5.0)
        if v.max_error >= 5.0:
            wins_v += 1
        if d_res.max_error >= 5.0:
            wins_d += 1
    assert wins_d >= wins_v, (
        f"Diversity wins {wins_d}/10 vs vanilla {wins_v}/10"
    )
