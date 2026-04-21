# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.diagnostics and the C5 change-point analysis."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fhe_oracle.diagnostics import ComponentLog, InstrumentedFitness

_ANALYSIS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "benchmarks", "analysis")
)
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from changepoint_analysis import pettitt_test, rolling_correlation  # noqa: E402


def test_component_log_records_and_arrays():
    log = ComponentLog()
    for i in range(10):
        log.record(
            np.array([float(i)] * 3),
            divergence=float(i) * 0.1,
            noise_term=float(i) * 0.01,
            depth_term=float(i) * 0.02,
            fitness=float(i),
        )
    arr = log.to_arrays()
    assert set(arr) == {
        "eval_index",
        "divergence",
        "noise_term",
        "depth_term",
        "fitness",
        "x_norm",
    }
    assert arr["divergence"].shape == (10,)
    np.testing.assert_allclose(arr["divergence"], np.arange(10) * 0.1)
    np.testing.assert_allclose(arr["eval_index"], np.arange(10))


def test_component_log_to_csv(tmp_path):
    log = ComponentLog()
    log.record(np.array([1.0, 2.0]), 0.5, 0.1, 0.2, 0.7)
    log.record(np.array([0.0, 0.0]), 0.0, 0.0, 0.0, 0.0)
    path = tmp_path / "log.csv"
    log.to_csv(str(path))
    text = path.read_text().strip().splitlines()
    assert text[0].startswith("eval_index,divergence")
    assert len(text) == 3  # header + 2 rows


def test_instrumented_fitness_logs_components():
    def plain(x):
        return float(np.sum(np.asarray(x)))

    def fhe(x):
        arr = np.asarray(x)
        return float(np.sum(arr) + 0.01 * np.max(np.abs(arr)))

    fit = InstrumentedFitness(plain, fhe, dim=3, w_div=1.0, w_noise=0.5, w_depth=0.3)
    s = fit.score([1.0, 0.0, 0.0])
    assert len(fit.log.evaluations) == 1
    ev = fit.log.evaluations[0]
    assert ev["divergence"] == pytest.approx(0.01, abs=1e-9)
    expected_noise = min(1.0, 1.0 / (np.sqrt(3.0) * 3.0))
    assert ev["noise_term"] == pytest.approx(expected_noise, abs=1e-9)
    assert ev["depth_term"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert s == pytest.approx(
        1.0 * 0.01 + 0.5 * expected_noise + 0.3 * (1.0 / 3.0), abs=1e-9
    )


def test_instrumented_fitness_clamps_at_one():
    def plain(x):
        return 0.0

    def fhe(x):
        return 0.0

    fit = InstrumentedFitness(plain, fhe, dim=2)
    fit.score([100.0, 100.0])
    ev = fit.log.evaluations[0]
    assert ev["noise_term"] == pytest.approx(1.0)
    assert ev["depth_term"] == pytest.approx(1.0)


def test_instrumented_fitness_handles_plaintext_exception():
    def plain(x):
        raise RuntimeError("boom")

    def fhe(x):
        return 0.0

    fit = InstrumentedFitness(plain, fhe, dim=2)
    s = fit.score([1.0, 1.0])
    ev = fit.log.evaluations[0]
    assert ev["divergence"] == 0.0
    assert ev["noise_term"] > 0.0
    assert s >= 0.0


def test_instrumented_fitness_vector_output_uses_max_abs():
    def plain(x):
        return [1.0, 2.0, 3.0]

    def fhe(x):
        return [1.0, 2.0, 3.5]

    fit = InstrumentedFitness(plain, fhe, dim=3)
    fit.score([0.0, 0.0, 0.0])
    assert fit.log.evaluations[0]["divergence"] == pytest.approx(0.5)


def test_rolling_correlation_perfectly_correlated_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 3.0 * x + 0.5
    r = rolling_correlation(x, y, window=50)
    assert np.all(r > 0.999)


def test_rolling_correlation_independent_is_near_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    r = rolling_correlation(x, y, window=50)
    assert np.abs(np.mean(r)) < 0.1


def test_pettitt_test_detects_known_change_point():
    rng = np.random.default_rng(7)
    seg_a = rng.normal(0.0, 1.0, size=250)
    seg_b = rng.normal(2.0, 1.0, size=250)
    x = np.concatenate([seg_a, seg_b])
    cp, p = pettitt_test(x)
    assert 230 <= cp <= 270
    assert p < 0.01


def test_pettitt_test_on_constant_data_rejects_change_point():
    x = np.ones(500)
    cp, p = pettitt_test(x)
    assert p > 0.05


def test_pettitt_test_stationary_noise_rejects_change_point():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=500)
    _, p = pettitt_test(x)
    assert p > 0.05
