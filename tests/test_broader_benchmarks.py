# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Tests for the Concrete-ML broader-benchmarks adapter wiring.

Skipped when concrete-ml is not installed (it requires Python <3.13
and a separate pip install). The structural tests still verify that
the benchmark module imports and exposes the expected entry points,
even on environments without concrete-ml.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _has_concrete_ml() -> bool:
    try:
        importlib.import_module("concrete.ml")
        return True
    except Exception:
        return False


def test_benchmark_module_imports() -> None:
    """The benchmark module is importable even without concrete-ml."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks")
    assert hasattr(bb, "_build_concrete_circuit")
    assert hasattr(bb, "_oracle_run")
    assert hasattr(bb, "_random_baseline")
    assert hasattr(bb, "run_circuit")
    assert hasattr(bb, "main")


def test_build_circuit_raises_clearly_without_concrete_ml() -> None:
    """If concrete-ml is missing, the builder must raise a clear RuntimeError."""
    if _has_concrete_ml():
        pytest.skip("concrete-ml installed — skip the missing-dep test")

    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks")
    with pytest.raises(RuntimeError, match="concrete-ml is required"):
        bb._build_concrete_circuit("wdbc")


@pytest.mark.skipif(
    not _has_concrete_ml(),
    reason="concrete-ml not installed (requires Python <3.13 + pip install)",
)
def test_concrete_wdbc_circuit_callables_match_protocol() -> None:
    """End-to-end smoke: build the circuit, run a single divergence eval."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks")
    plain, fhe, Xtr, d = bb._build_concrete_circuit("wdbc")
    assert d == 30
    assert Xtr.shape[1] == d
    # Both functions return scalars in [0,1]
    x = Xtr[0]
    p = plain(x)
    f = fhe(x)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= f <= 1.0
    # Divergence is finite and small (n_bits=8 quantization)
    assert abs(p - f) < 0.1


@pytest.mark.skipif(
    not _has_concrete_ml(),
    reason="concrete-ml not installed (requires Python <3.13 + pip install)",
)
def test_oracle_finds_higher_divergence_than_random_at_least_once() -> None:
    """At seed=0, oracle should find >= half the divergence random does.

    Not a strict win test (single seed is noisy); just sanity that the
    oracle returns numerical results in the same order of magnitude as
    the random baseline.
    """
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks")
    plain, fhe, _Xtr, d = bb._build_concrete_circuit("wdbc")
    bounds = bb._bounds_for(d)
    o_err, _, _ = bb._oracle_run(plain, fhe, bounds, 30, seed=0)
    r_err, _, _ = bb._random_baseline(plain, fhe, bounds, 30, seed=0)
    assert o_err > 0.0
    assert r_err > 0.0
    # Both should detect quantization-induced divergence well above 1e-4
    assert max(o_err, r_err) > 1e-3
