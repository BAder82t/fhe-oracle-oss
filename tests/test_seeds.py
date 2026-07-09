# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the fallback corner/random seed generator (fhe_oracle.seeds)."""

from __future__ import annotations

import numpy as np

from fhe_oracle.seeds import fallback_corner_seeds


def test_returns_empty_list_when_k_non_positive():
    rng = np.random.default_rng(0)
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    assert fallback_corner_seeds(rng, bounds, k=0) == []
    assert fallback_corner_seeds(rng, bounds, k=-5) == []


def test_returns_empty_list_when_bounds_empty():
    rng = np.random.default_rng(0)
    assert fallback_corner_seeds(rng, [], k=10) == []


def test_returns_k_seeds_of_correct_dimension():
    rng = np.random.default_rng(0)
    bounds = [(-1.0, 1.0), (0.0, 5.0), (-2.0, 2.0)]
    seeds = fallback_corner_seeds(rng, bounds, k=8)
    assert len(seeds) == 8
    assert all(len(s) == 3 for s in seeds)


def test_seeds_are_within_bounds():
    rng = np.random.default_rng(1)
    bounds = [(-1.0, 1.0), (0.0, 5.0), (-2.0, 2.0)]
    seeds = fallback_corner_seeds(rng, bounds, k=20)
    for s in seeds:
        for value, (low, high) in zip(s, bounds):
            assert low <= value <= high


def test_corner_half_are_axis_aligned_corner_values():
    rng = np.random.default_rng(2)
    bounds = [(-1.0, 1.0), (0.0, 5.0)]
    k = 10
    seeds = fallback_corner_seeds(rng, bounds, k=k)
    n_corner = k // 2
    corner_seeds = seeds[:n_corner]
    for s in corner_seeds:
        for value, (low, high) in zip(s, bounds):
            assert value == low or value == high


def test_is_deterministic_given_seeded_rng():
    bounds = [(-1.0, 1.0), (0.0, 5.0)]
    seeds_a = fallback_corner_seeds(np.random.default_rng(42), bounds, k=6)
    seeds_b = fallback_corner_seeds(np.random.default_rng(42), bounds, k=6)
    assert seeds_a == seeds_b


def test_default_k_is_ten():
    rng = np.random.default_rng(0)
    bounds = [(-1.0, 1.0)]
    assert len(fallback_corner_seeds(rng, bounds)) == 10
