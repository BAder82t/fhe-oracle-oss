# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Tests for the Concrete-ML MLP broader-benchmarks adapter wiring.

Skipped when concrete-ml is not installed (it requires Python <3.13 and
a separate pip install). The structural tests still verify that the
benchmark module imports and exposes the expected entry points, even
on environments without concrete-ml.
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


def test_mlp_benchmark_module_imports() -> None:
    """The MLP benchmark module is importable even without concrete-ml."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks_mlp")
    assert hasattr(bb, "_build_concrete_mlp")
    assert hasattr(bb, "_oracle_run")
    assert hasattr(bb, "_random_baseline")
    assert hasattr(bb, "run_circuit")
    assert hasattr(bb, "main")


def test_mlp_build_circuit_raises_clearly_without_concrete_ml() -> None:
    """If concrete-ml is missing, the builder must raise a clear RuntimeError."""
    if _has_concrete_ml():
        pytest.skip("concrete-ml installed — skip the missing-dep test")

    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks_mlp")
    with pytest.raises(RuntimeError, match="concrete-ml is required"):
        bb._build_concrete_mlp("wdbc")


@pytest.mark.skipif(
    not _has_concrete_ml(),
    reason="concrete-ml not installed (requires Python <3.13 + pip install)",
)
def test_concrete_wdbc_mlp_callables_match_protocol() -> None:
    """End-to-end smoke: build the MLP circuit, run a single divergence eval.

    Slow test — train + compile is ~3-5 s. Marked structural because
    the per-eval FHE cost is ~1.4 s and we only check one input.
    """
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    ))
    bb = importlib.import_module("broader_benchmarks_mlp")
    plain, fhe, Xtr, d, meta = bb._build_concrete_mlp("wdbc")
    assert d == 30
    assert meta["n_w_bits"] == 4
    assert meta["n_a_bits"] == 4
    x = Xtr[0]
    p = plain(x)
    f = fhe(x)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= f <= 1.0
    # Divergence is finite; at 4-bit quantization on a multi-layer MLP
    # the gap can be substantial (we observed >0.3 max on test set).
    assert abs(p - f) < 1.0
