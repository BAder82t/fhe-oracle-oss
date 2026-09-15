# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.diagnostics and the C5 change-point analysis."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fhe_oracle import EvaluationError
from fhe_oracle.adapters.base import FHEAdapter
from fhe_oracle.diagnostics import (
    ComponentLog,
    InstrumentedFitness,
    OperationStep,
    OperationTrace,
    StructureReport,
    TracingCircuit,
    characterize_structure,
    localize_fault,
    per_op_trace,
)

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
    with pytest.raises(EvaluationError, match="boom"):
        fit.score([1.0, 1.0])
    assert fit.log.evaluations == []


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


def _make_step(name, step_error):
    return OperationStep(
        name=name,
        plaintext_value=0.0,
        fhe_value=step_error,
        step_error=step_error,
        cumulative_error=step_error,
    )


def _make_trace(step_errors, total_divergence=None):
    ops = [_make_step(f"op_{i}", e) for i, e in enumerate(step_errors)]
    total = total_divergence if total_divergence is not None else max(step_errors)
    return OperationTrace(
        input_x=np.array([0.0]),
        total_divergence=total,
        plaintext_output=0.0,
        fhe_output=total,
        operations=ops,
    )


def test_localize_fault_returns_first_step_crossing_threshold():
    trace = _make_trace([0.001, 0.002, 0.5, 0.001], total_divergence=0.5)
    step = localize_fault(trace)
    assert step.name == "op_2"


def test_localize_fault_falls_back_to_largest_step_error():
    trace = _make_trace([0.01, 0.012, 0.011, 0.009])
    step = localize_fault(trace, threshold=1.0)
    assert step.name == "op_1"  # largest step_error (0.012)


def test_localize_fault_custom_threshold():
    trace = _make_trace([0.1, 0.2, 0.9], total_divergence=0.9)
    step = localize_fault(trace, threshold=0.5)
    assert step.name == "op_2"


def test_localize_fault_zero_threshold_returns_first_step():
    # threshold=0 -> every non-negative step_error "crosses" it, so the
    # FIRST step wins regardless of magnitude.
    trace = _make_trace([0.0, 0.001, 0.5], total_divergence=0.5)
    step = localize_fault(trace, threshold=0.0)
    assert step.name == "op_0"


def test_localize_fault_tie_breaks_to_first_occurrence():
    trace = _make_trace([0.3, 0.3, 0.1], total_divergence=0.3)
    step = localize_fault(trace, threshold=1.0)  # forces fallback path
    assert step.name == "op_0"  # max() with ties returns the first max


def test_localize_fault_raises_on_empty_operations():
    trace = OperationTrace(
        input_x=np.array([0.0]),
        total_divergence=0.0,
        plaintext_output=0.0,
        fhe_output=0.0,
        operations=[],
    )
    with pytest.raises(ValueError):
        localize_fault(trace)


def test_characterize_structure_detects_low_rank_ridge_function():
    rng = np.random.default_rng(0)
    dim = 20
    true_rank = 3
    weights = rng.standard_normal((true_rank, dim))

    def ridge_fn(x):
        z = weights @ np.asarray(x)
        return float(np.sum(np.sin(z)))

    bounds = [(-1.0, 1.0)] * dim
    report = characterize_structure(ridge_fn, dim, bounds, n_samples=150, seed=1)
    assert isinstance(report, StructureReport)
    assert report.dim == dim
    assert report.effective_rank <= true_rank + 2
    assert report.effective_rank < dim


def test_characterize_structure_full_rank_function_reports_high_rank():
    # Independent per-coordinate nonlinearity with SIMILAR per-coordinate
    # gradient magnitude (coeffs close to 1.0) -- unlike a widely-scaled
    # separable function, this has no small subset of dimensions that
    # dominate variance, so effective_rank should approach dim.
    dim = 10
    rng = np.random.default_rng(2)
    coeffs = rng.uniform(0.8, 1.2, size=dim)

    def full_rank_fn(x):
        arr = np.asarray(x)
        return float(np.sum(np.sin(coeffs * arr * 3.0)))

    bounds = [(-1.0, 1.0)] * dim
    report = characterize_structure(
        full_rank_fn, dim, bounds, n_samples=150, seed=2
    )
    assert report.effective_rank >= dim - 2


def test_characterize_structure_validates_bounds_length():
    with pytest.raises(ValueError):
        characterize_structure(lambda x: 0.0, dim=3, bounds=[(-1.0, 1.0)] * 2)


def test_characterize_structure_validates_dim_positive():
    with pytest.raises(ValueError):
        characterize_structure(lambda x: 0.0, dim=0, bounds=[])


def test_characterize_structure_validates_n_samples():
    with pytest.raises(ValueError):
        characterize_structure(
            lambda x: 0.0, dim=3, bounds=[(-1.0, 1.0)] * 3, n_samples=1
        )


def test_characterize_structure_constant_fn_is_inconclusive():
    report = characterize_structure(
        lambda x: 42.0, dim=5, bounds=[(-1.0, 1.0)] * 5, n_samples=20, seed=0
    )
    assert report.effective_rank == 0
    assert report.variance_explained == []
    assert "inconclusive" in report.recommendation


class _FakeScalarAdapter(FHEAdapter):
    """Identity-encrypt fake adapter: ciphertext IS the plain float.

    Just enough of FHEAdapter to exercise TracingCircuit without a
    real FHE library, mirroring the fake-adapter pattern used in
    tests/test_differential.py.
    """

    def encrypt(self, x):
        return float(x[0]) if isinstance(x, (list, tuple)) else float(x)

    def decrypt(self, ciphertext):
        return [float(ciphertext)]

    def run_fhe_program(self, ciphertext):
        return ciphertext

    def get_noise_budget(self, ciphertext):
        return 0.0

    def get_mult_depth_used(self, ciphertext):
        return 0

    def get_scheme_name(self):
        return "fake"


def _build_traced_circuit(buggy: bool):
    adapter = _FakeScalarAdapter()

    def ct_double(ct):
        return ct * 2.0

    def ct_square(ct):
        return ct * ct + (0.5 if buggy else 0.0)

    def ct_plus_one(ct):
        return ct + 1.0

    def p_double(p):
        # first plaintext step: p is the raw input (a list), matching
        # what encrypt(x) reduces to a scalar ciphertext from.
        return p[0] * 2.0

    def p_square(p):
        return p * p

    def p_plus_one(p):
        return p + 1.0

    return TracingCircuit(
        adapter=adapter,
        steps=[
            ("double", ct_double),
            ("square", ct_square),
            ("plus_one", ct_plus_one),
        ],
        plaintext_steps=[p_double, p_square, p_plus_one],
    )


def test_tracing_circuit_matches_when_correct():
    circuit = _build_traced_circuit(buggy=False)
    trace = circuit.trace([3.0])
    assert len(trace) == 3
    assert [s.name for s in trace] == ["double", "square", "plus_one"]
    for step in trace:
        assert step.step_error == pytest.approx(0.0, abs=1e-9)
    # (3*2)^2 + 1 = 37
    assert circuit([3.0]) == pytest.approx(37.0)


def test_tracing_circuit_localizes_injected_fault():
    circuit = _build_traced_circuit(buggy=True)
    trace = circuit.trace([3.0])
    assert trace[0].step_error == pytest.approx(0.0, abs=1e-9)  # double: fine
    assert trace[1].step_error == pytest.approx(0.5, abs=1e-9)  # square: buggy
    assert trace[2].step_error == pytest.approx(0.5, abs=1e-9)  # propagates

    op_trace = per_op_trace(
        [3.0],
        plaintext_fn=lambda x: (x[0] * 2.0) ** 2 + 1.0,
        fhe_fn=circuit,
    )
    assert isinstance(op_trace, OperationTrace)
    fault = localize_fault(op_trace)
    assert fault.name == "square"


def test_tracing_circuit_empty_steps_passes_through_unchanged():
    adapter = _FakeScalarAdapter()
    circuit = TracingCircuit(adapter=adapter, steps=[], plaintext_steps=[])
    assert circuit.trace([3.0]) == []
    assert circuit([3.0]) == pytest.approx(3.0)


def test_tracing_circuit_rejects_mismatched_step_lengths():
    adapter = _FakeScalarAdapter()
    with pytest.raises(ValueError):
        TracingCircuit(
            adapter=adapter,
            steps=[("a", lambda ct: ct)],
            plaintext_steps=[lambda p: p, lambda p: p],
        )
