# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the C4 per_op_trace diagnostic tool."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL
from fhe_oracle.diagnostics import (
    OperationStep,
    OperationTrace,
    per_op_trace,
)


def test_mock_trace_one_step_captures_divergence():
    def plain(x):
        return float(np.sum(np.asarray(x)))

    def fhe(x):
        arr = np.asarray(x)
        return float(np.sum(arr) + 0.03)

    trace = per_op_trace([1.0, 2.0, 3.0], plain, fhe)
    assert isinstance(trace, OperationTrace)
    assert trace.total_divergence == pytest.approx(0.03, abs=1e-9)
    assert len(trace.operations) == 1
    assert trace.operations[0].step_error == pytest.approx(0.03, abs=1e-9)
    assert trace.operations[0].cumulative_error == pytest.approx(0.03, abs=1e-9)


def test_mock_trace_vector_output_uses_max_abs():
    def plain(x):
        return [1.0, 2.0, 3.0]

    def fhe(x):
        return [1.0, 2.0, 3.4]

    trace = per_op_trace([0.0, 0.0, 0.0], plain, fhe)
    assert trace.total_divergence == pytest.approx(0.4, abs=1e-9)


def test_mock_trace_to_csv_roundtrip(tmp_path):
    def plain(x):
        return 1.0

    def fhe(x):
        return 1.25

    trace = per_op_trace([0.0], plain, fhe)
    csv_path = tmp_path / "trace.csv"
    trace.to_csv(str(csv_path))
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("step_index,name,plaintext_value")
    assert len(lines) == 2


def test_mock_trace_summary_is_nonempty_and_contains_divergence():
    def plain(x):
        return 0.5

    def fhe(x):
        return 0.55

    trace = per_op_trace([0.0], plain, fhe)
    s = trace.summary()
    assert "OperationTrace" in s
    assert "total_divergence" in s


def test_mock_trace_honors_custom_operation_name():
    def plain(x):
        return 1.0

    def fhe(x):
        return 1.1

    trace = per_op_trace([0.0], plain, fhe, operation_names=["custom"])
    assert trace.operations[0].name == "custom"


def test_trace_method_is_used_if_provided():
    class FakeTracingFn:
        def __call__(self, x):
            return 10.0

        def trace(self, x):
            return [
                OperationStep(
                    name="op_0",
                    plaintext_value=0.0,
                    fhe_value=0.1,
                    step_error=0.1,
                    cumulative_error=0.1,
                    noise_budget=None,
                ),
                OperationStep(
                    name="op_1",
                    plaintext_value=2.0,
                    fhe_value=2.2,
                    step_error=0.2,
                    cumulative_error=0.2,
                    noise_budget=None,
                ),
            ]

    def plain(x):
        return 10.0

    fhe = FakeTracingFn()
    trace = per_op_trace([0.0], plain, fhe)
    assert len(trace.operations) == 2
    assert trace.operations[1].step_error == 0.2


@pytest.mark.skipif(not HAVE_TENSEAL, reason="tenseal not installed")
def test_tenseal_tracing_fn_produces_six_steps_on_lr_d8():
    from fhe_oracle.adapters.tenseal_adapter import TenSEALContext
    from fhe_oracle.diagnostics import TracingTenSEALFn

    ctx = TenSEALContext()
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 1.0, size=8)
    b = 0.5

    def plaintext_fn(x):
        z = float(np.dot(w, x) + b)
        return 0.5 + z * 0.25 - (z ** 3) * (1.0 / 48.0)

    tracing = TracingTenSEALFn(w, b, ctx)
    x = np.array([1.0, -0.5, 0.2, -1.1, 0.3, 0.8, -0.6, 0.9])
    trace = per_op_trace(x, plaintext_fn, tracing)
    assert len(trace.operations) == 6
    names = [op.name for op in trace.operations]
    assert names == [
        "z_preactivation",
        "z_over_4",
        "z_squared",
        "z_cubed",
        "z3_over_48",
        "sigma_t3",
    ]
    assert all(op.step_error >= 0.0 for op in trace.operations)
    assert all(
        trace.operations[i].cumulative_error
        >= trace.operations[i - 1].cumulative_error
        for i in range(1, len(trace.operations))
    )
    final_step_err = trace.operations[-1].step_error
    assert final_step_err == pytest.approx(trace.total_divergence, abs=1e-6)
    assert trace.operations[-1].cumulative_error >= final_step_err
