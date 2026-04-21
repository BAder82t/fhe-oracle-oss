# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for multi-output fitness (fhe_oracle.multi_output)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.multi_output import MultiOutputFitness, MultiOutputMode


def test_max_absolute_matches_scalar_diff():
    """MAX_ABSOLUTE mode mirrors the plain max-of-abs-diff fitness."""
    fitness = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.55, 0.35, 0.1]),
        mode=MultiOutputMode.MAX_ABSOLUTE,
    )
    assert abs(fitness(np.zeros(3)) - 0.05) < 1e-12


def test_rank_inversion_detected():
    """argmax mismatch yields a score above the inversion bonus."""
    fitness = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.29, 0.31, 0.4]),
        mode=MultiOutputMode.RANK_INVERSION,
    )
    score = fitness(np.zeros(3))
    assert score > 1.0


def test_near_flip_gets_high_score():
    """Inputs near a decision flip score higher than safe-margin inputs."""
    f_close = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.51, 0.49, 0.0]),
        mode=MultiOutputMode.RANK_INVERSION,
    )
    f_safe = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.8, 0.15, 0.05]),
        mode=MultiOutputMode.RANK_INVERSION,
    )
    assert f_close(np.zeros(3)) > f_safe(np.zeros(3))


def test_combined_includes_both():
    """COMBINED mode is the sum of MAX_ABSOLUTE and weighted RANK."""
    p = np.array([0.6, 0.3, 0.1])
    f = np.array([0.55, 0.35, 0.1])
    abs_part = MultiOutputFitness(
        lambda x: p, lambda x: f, mode=MultiOutputMode.MAX_ABSOLUTE
    )(np.zeros(3))
    rank_part = MultiOutputFitness(
        lambda x: p, lambda x: f, mode=MultiOutputMode.RANK_INVERSION
    )(np.zeros(3))
    combined = MultiOutputFitness(
        lambda x: p, lambda x: f,
        mode=MultiOutputMode.COMBINED, rank_weight=2.0,
    )(np.zeros(3))
    assert abs(combined - (abs_part + 2.0 * rank_part)) < 1e-9


def test_detailed_report_has_expected_fields():
    fitness = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.55, 0.35, 0.1]),
    )
    r = fitness.detailed_report(np.zeros(3))
    for key in (
        "plaintext_output", "fhe_output", "per_output_error",
        "max_absolute_error", "decision_flipped", "fhe_top2_margin",
        "fhe_ranking", "plaintext_class", "fhe_class",
    ):
        assert key in r
    assert r["decision_flipped"] is False
    assert r["plaintext_class"] == 0
    assert r["fhe_class"] == 0


def test_decision_flip_in_report():
    fitness = MultiOutputFitness(
        plaintext_fn=lambda x: np.array([0.6, 0.3, 0.1]),
        fhe_fn=lambda x: np.array([0.29, 0.31, 0.4]),
    )
    r = fitness.detailed_report(np.zeros(3))
    assert r["decision_flipped"] is True
    assert r["plaintext_class"] == 0
    assert r["fhe_class"] == 2


def test_scalar_output_falls_back_gracefully():
    """A scalar output (k=1) should not crash; returns abs diff."""
    fitness = MultiOutputFitness(
        plaintext_fn=lambda x: 0.5,
        fhe_fn=lambda x: 0.7,
        mode=MultiOutputMode.RANK_INVERSION,
    )
    assert abs(fitness(np.zeros(2)) - 0.2) < 1e-12


def test_score_returns_zero_on_exception():
    """Exceptions in plaintext/fhe yield 0 (no propagation)."""
    def boom(x):
        raise RuntimeError("bad")

    fitness = MultiOutputFitness(plaintext_fn=boom, fhe_fn=boom)
    assert fitness.score(np.zeros(3)) == 0.0
