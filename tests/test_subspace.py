# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for random-subspace embedding (fhe_oracle.subspace)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.subspace import SubspaceOracle


# --- geometry ----------------------------------------------------------


def test_projection_orthonormal():
    """QR-generated projection has orthonormal columns."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.0,
        bounds=[(-1.0, 1.0)] * 100,
        subspace_dim=20,
    )
    rng = np.random.default_rng(42)
    R = oracle._make_projection(rng)
    assert R.shape == (100, 20)
    np.testing.assert_allclose(R.T @ R, np.eye(20), atol=1e-10)


def test_z_to_x_clipping():
    """Mapped x always lies inside the input box."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.0,
        bounds=[(-3.0, 3.0)] * 50,
        subspace_dim=10,
    )
    rng = np.random.default_rng(42)
    R = oracle._make_projection(rng)
    for _ in range(100):
        z = rng.standard_normal(10) * 5.0
        x = oracle._z_to_x(z, R)
        assert np.all(x >= -3.0 - 1e-9)
        assert np.all(x <= 3.0 + 1e-9)


def test_z_bounds_are_finite():
    """_z_bounds returns finite, positive-width boxes."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 0.0,
        bounds=[(-2.0, 4.0)] * 30,
        subspace_dim=8,
    )
    rng = np.random.default_rng(7)
    R = oracle._make_projection(rng)
    z_lo, z_hi = oracle._z_bounds(R)
    assert np.all(np.isfinite(z_lo))
    assert np.all(np.isfinite(z_hi))
    assert np.all(z_hi - z_lo > 0.0)


# --- search behaviour --------------------------------------------------


def test_subspace_finds_bug():
    """A simple one-feature bug at large x[0] is reachable through the
    subspace search."""
    d = 200

    def fhe_fn(x):
        x_arr = np.asarray(x)
        return 0.01 * float(x_arr[0]) ** 3 if x_arr[0] > 0 else 0.0

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=30,
        n_projections=5,
    )
    result = oracle.run(n_trials=500, seed=42)
    # The bug grows as x[0]^3; with 500 evals across 5 subspaces we
    # should comfortably clear 0.05.
    assert result.max_error > 0.05
    assert result.strategy_used == "subspace"
    assert result.subspace_dim == 30
    assert result.n_projections == 5


def test_subspace_high_dim_makes_progress():
    """At d=200 with a directional bug, subspace CMA-ES finds a
    non-trivial divergence and reports the expected metadata. This
    documents the value proposition: the search scales without
    requiring ~50d = 10000 evaluations that full-d CMA-ES would
    demand.
    """
    d = 200
    # Bug depends on a known 3-coord direction; full-d CMA-ES would
    # need ~50d = 10k evals, subspace search succeeds in 500.
    def fhe_fn(x):
        x_arr = np.asarray(x)
        z = float(x_arr[0]) + 2.0 * float(x_arr[1]) - float(x_arr[2])
        return abs(z) ** 3 / 10.0

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=50,
        n_projections=3,
    )
    result = oracle.run(n_trials=500, seed=42)

    # Theoretical max: z_max = 3 + 6 + 3 = 12 -> fhe_max = 172.8. We
    # expect subspace search to recover a meaningful fraction even
    # though d=200 > k=50. A loose 10% threshold is a stable gate that
    # does not flake under seed variance.
    assert result.max_error > 17.0
    assert result.strategy_used == "subspace"
    assert 0 <= result.projection_index < 3


def test_subspace_invalid_k():
    """k must be in [1, d]."""
    with pytest.raises(ValueError):
        SubspaceOracle(
            plaintext_fn=lambda x: 0,
            fhe_fn=lambda x: 0,
            bounds=[(-1.0, 1.0)] * 10,
            subspace_dim=20,
        )


# --- v0.3.1 geometry overhaul -----------------------------------------


def test_z_bounds_ball_not_sliver():
    """Ball-radius bounds should be much wider than the legacy
    intersection-of-constraints bounds at high d."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0,
        fhe_fn=lambda x: 0,
        bounds=[(-3.0, 3.0)] * 500,
        subspace_dim=50,
    )
    rng = np.random.default_rng(42)
    R = oracle._make_projection(rng)

    z_lo_ball, z_hi_ball = oracle._z_bounds_ball(R=R)
    z_lo_old, z_hi_old = oracle._z_bounds_intersection(R)

    ball_widths = z_hi_ball - z_lo_ball
    old_widths = z_hi_old - z_lo_old
    # The intersection bounds tighten with d via the
    # max-of-d-Gaussians ~ sqrt(2 log d) factor; the ball median
    # avoids that and is ~5x wider at d=500 with k=50. Use 3x as a
    # stable lower bound that still guards the regression.
    assert np.mean(ball_widths) > 3 * np.mean(old_widths), (
        f"Ball mean={np.mean(ball_widths):.2f} vs intersection "
        f"mean={np.mean(old_widths):.2f}"
    )


def test_z_bounds_ball_finite_positive():
    """Ball bounds are finite, symmetric, and have positive width."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0,
        fhe_fn=lambda x: 0,
        bounds=[(-2.0, 4.0)] * 100,
        subspace_dim=20,
    )
    rng = np.random.default_rng(11)
    R = oracle._make_projection(rng)
    z_lo, z_hi = oracle._z_bounds_ball(R=R)
    assert np.all(np.isfinite(z_lo))
    assert np.all(np.isfinite(z_hi))
    assert np.all(z_hi - z_lo > 0)
    np.testing.assert_allclose(z_lo, -z_hi, atol=1e-10)


def test_generate_anchors_midpoint_first():
    """First anchor is always the box midpoint."""
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0,
        fhe_fn=lambda x: 0,
        bounds=[(-3.0, 3.0)] * 50,
        subspace_dim=10,
        n_anchors=3,
    )
    anchors = oracle._generate_anchors()
    assert len(anchors) == 3
    midpoint = 0.5 * (oracle.lo + oracle.hi)
    np.testing.assert_allclose(anchors[0], midpoint)
    # Other anchors are not midpoints (probabilistic; with d=50 the
    # chance that all coordinates land on the midpoint is essentially 0).
    assert not np.allclose(anchors[1], midpoint)


def test_user_anchor_disables_multi_anchor():
    """Explicit ``anchor=`` overrides multi-anchor and forces n_anchors=1."""
    custom = np.linspace(-1.0, 1.0, 50)
    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0,
        fhe_fn=lambda x: 0,
        bounds=[(-3.0, 3.0)] * 50,
        subspace_dim=10,
        n_anchors=4,
        anchor=custom,
    )
    assert oracle.n_anchors == 1
    anchors = oracle._generate_anchors()
    assert len(anchors) == 1
    np.testing.assert_allclose(anchors[0], custom)


def test_multi_anchor_finds_corners():
    """Multi-anchor search reaches corner bugs that midpoint misses.

    The bug = sum_i max(0, x_i - 2). At x = midpoint = 0, the bug is 0.
    At a box corner (x_i = 3 for all i), the bug is d * 1 = d.
    """
    d = 200

    def fhe_fn(x):
        return float(np.sum(np.maximum(0.0, np.asarray(x) - 2.0)))

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=30,
        n_projections=3,
        n_anchors=2,
    )
    result = oracle.run(n_trials=500, seed=42)
    assert result.max_error > 50, (
        f"Expected >50 (corner bug), got {result.max_error:.2f}"
    )


def test_subspace_beats_random_lr_mock_d200():
    """Geometry fix: subspace should match or beat random on the
    lr_mock_d200 circuit that failed in v0.3.0 benchmarks (ratio 0.29).

    Mirrors the v0.3.0 benchmark circuit exactly: sigmoid plaintext,
    sigmoid + hot-zone-amplified noise FHE side. The bug grows with
    ``||x||^2`` so corner anchors should help.
    """
    d = 200
    rng = np.random.RandomState(42)
    w = rng.normal(0.0, 1.0, d)
    bias = 0.5

    def plaintext_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        z = float(w @ xa + bias)
        return 1.0 / (1.0 + np.exp(-z))

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        plain = plaintext_fn(xa)
        h = int(abs(hash(tuple(round(float(v), 9) for v in xa))) % (2**31))
        local = np.random.RandomState(h)
        noise = float(local.normal(0.0, 1e-4))
        z_proxy = float(np.dot(xa, xa))
        if z_proxy > 4.0 and abs(plain - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 4.0)
            noise *= amp
        return plain + noise

    oracle = SubspaceOracle(
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=50,
        n_projections=3,
        n_anchors=2,
        clip_penalty=0.1,
    )
    result = oracle.run(n_trials=500, seed=1)

    rng2 = np.random.RandomState(1)
    random_max = 0.0
    for _ in range(500):
        x = rng2.uniform(-3.0, 3.0, d)
        random_max = max(random_max, abs(plaintext_fn(x) - fhe_fn(x)))
    assert result.max_error >= random_max * 0.8, (
        f"Subspace {result.max_error:.4f} vs random {random_max:.4f}"
    )


def test_clip_penalty_steers_toward_bounds():
    """Clip penalty allows out-of-bounds exploration but prefers
    in-bounds inputs."""
    d = 100

    def fhe_fn(x):
        return float(np.max(np.abs(np.asarray(x))))

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=20,
        n_projections=1,
        n_anchors=1,
        clip_penalty=0.1,
    )
    result = oracle.run(n_trials=200, seed=42)
    best_x = np.asarray(result.worst_input)
    assert np.max(np.abs(best_x)) > 2.0, (
        f"Best input max coord {np.max(np.abs(best_x)):.2f} -- stuck near midpoint"
    )


def test_random_fallback_records_metadata():
    """When the subspace search is no better than the probe,
    fallback_taken is set on the returned result."""
    d = 50

    def fhe_fn(x):
        return 0.0  # zero divergence -> sub-search cannot beat probe (also 0)

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-3.0, 3.0)] * d,
        subspace_dim=10,
        n_projections=3,
        n_anchors=2,
        fallback_threshold=1.10,
    )
    result = oracle.run(n_trials=200, seed=0)
    assert hasattr(result, "fallback_taken")
    assert hasattr(result, "probe_max")


def test_subspace_result_x_in_bounds():
    """Returned worst_input lies inside the input box."""
    d = 50

    def fhe_fn(x):
        return float(np.sum(np.asarray(x) ** 2)) * 0.001

    oracle = SubspaceOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=fhe_fn,
        bounds=[(-2.0, 2.0)] * d,
        subspace_dim=8,
        n_projections=2,
    )
    result = oracle.run(n_trials=100, seed=1)
    x = np.asarray(result.worst_input)
    assert x.size == d
    assert np.all(x >= -2.0 - 1e-9)
    assert np.all(x <= 2.0 + 1e-9)
