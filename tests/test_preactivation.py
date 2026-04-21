# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.preactivation.PreactivationOracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fhe_oracle.preactivation import PreactivationOracle


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------


def test_z_to_x_round_trip_identity_for_in_bounds_z():
    """For z in the achievable z-box and W full row-rank, projection
    plus W-mapping recovers z up to numerical precision."""
    rng = np.random.default_rng(0)
    d = 8
    W = rng.normal(size=(1, d))
    b = np.array([0.1])
    pre = PreactivationOracle(
        W=W, b=b,
        plaintext_fn=lambda x: 0.0, fhe_fn=lambda x: 0.0,
        input_bounds=[(-3.0, 3.0)] * d,
    )
    # Generate a target z that is reachable: z = W·x + b for x in box
    x_target = rng.uniform(-2.0, 2.0, size=d)
    z_target = (W @ x_target + b).ravel()
    x_proj, clip_dist = pre.z_to_x(z_target)
    z_back = (W @ x_proj + b).ravel()
    np.testing.assert_allclose(z_back, z_target, atol=1e-9)
    # No clipping should be required when x_target is well inside box
    assert clip_dist < 1e-9


def test_z_bounds_contain_random_inputs():
    """For random x in the input box, z = W·x + b lies inside z_bounds."""
    rng = np.random.default_rng(7)
    d = 16
    W = rng.normal(size=(2, d))
    b = rng.normal(size=2)
    pre = PreactivationOracle(
        W=W, b=b,
        plaintext_fn=lambda x: 0.0, fhe_fn=lambda x: 0.0,
        input_bounds=[(-3.0, 3.0)] * d,
    )
    z_lo, z_hi = pre.z_bounds()
    for _ in range(50):
        x = rng.uniform(-3.0, 3.0, size=d)
        z = (W @ x + b).ravel()
        assert np.all(z >= z_lo - 1e-9), f"z={z} below z_lo={z_lo}"
        assert np.all(z <= z_hi + 1e-9), f"z={z} above z_hi={z_hi}"


def test_clip_penalty_punishes_out_of_bounds_projection():
    """A z that requires heavy clipping should score lower than a feasible z."""
    rng = np.random.default_rng(3)
    d = 4
    W = rng.normal(size=(1, d))
    b = np.array([0.0])

    def plain(x):
        return 0.0

    def fhe(x):
        return 0.5  # constant divergence of 0.5 everywhere

    pre = PreactivationOracle(
        W=W, b=b,
        plaintext_fn=plain, fhe_fn=fhe,
        input_bounds=[(-1.0, 1.0)] * d,
        clip_penalty=10.0,
    )
    fitness = pre._build_fitness()
    # Feasible z: 0 maps to x = pseudoinverse(W)·(-b) = 0, no clipping.
    s_in = fitness.score(np.array([0.0]))
    # Infeasible z: pick z far outside the achievable range.
    z_lo, z_hi = pre.z_bounds()
    z_far = np.array([z_hi[0] + 10.0])  # well outside reachable z
    s_out = fitness.score(z_far)
    assert s_in == pytest.approx(0.5, abs=1e-9)
    assert s_out < s_in, f"clip penalty should punish out-of-box; in={s_in} out={s_out}"


# ---------------------------------------------------------------------
# Search behaviour
# ---------------------------------------------------------------------


def test_k1_lr_finds_high_divergence_at_d8_low_budget():
    """For an LR-style affine + sigmoid model with a noise spike at
    large |z|, k=1 search at B=50 should localise the spike where
    pure random in d=8 at the same budget would only luck into it."""
    rng = np.random.default_rng(42)
    d = 8
    w = rng.normal(size=d)
    w = w / np.linalg.norm(w)
    b = 0.0

    def plain(x):
        z = float(np.dot(w, x) + b)
        return _sigmoid(z)

    def fhe(x):
        z = float(np.dot(w, x) + b)
        # Spike: divergence large in a thin shell |z| > 2.0
        spike = 0.4 if abs(z) > 2.0 else 0.0
        return _sigmoid(z) + spike

    pre = PreactivationOracle(
        W=w.reshape(1, -1), b=np.array([b]),
        plaintext_fn=plain, fhe_fn=fhe,
        input_bounds=[(-3.0, 3.0)] * d,
        clip_penalty=0.05,
    )
    results = pre.run(budget=50, seeds=range(1, 6), random_floor=0.2)
    errors = [r.max_error for r in results]
    median_err = float(np.median(errors))
    # All five seeds should find the spike with high probability.
    assert median_err >= 0.35, f"k=1 search should hit the spike: {errors}"


def test_run_returns_one_result_per_seed():
    rng = np.random.default_rng(11)
    d = 6
    W = rng.normal(size=(1, d))
    b = np.array([0.0])
    pre = PreactivationOracle(
        W=W, b=b,
        plaintext_fn=lambda x: 0.0, fhe_fn=lambda x: float(np.sum(np.asarray(x) ** 2)),
        input_bounds=[(-1.0, 1.0)] * d,
    )
    seeds = [1, 2, 3, 4]
    out = pre.run(budget=30, seeds=seeds)
    assert [r.seed for r in out] == seeds
    for r in out:
        assert len(r.x) == d
        assert len(r.z) == 1
        assert r.n_trials > 0


def test_validation_errors():
    with pytest.raises(ValueError):
        PreactivationOracle(
            W=np.zeros((2, 4)), b=np.zeros(3),  # bias mismatch
            plaintext_fn=lambda x: 0.0, fhe_fn=lambda x: 0.0,
            input_bounds=[(-1, 1)] * 4,
        )
    with pytest.raises(ValueError):
        PreactivationOracle(
            W=np.zeros((1, 4)), b=np.zeros(1),
            plaintext_fn=lambda x: 0.0, fhe_fn=lambda x: 0.0,
            input_bounds=[(-1, 1)] * 5,  # wrong d
        )
