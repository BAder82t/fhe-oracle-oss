# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""benchmarks/library_comparison.py must surface backend failures, never score them as 0.0.

Concrete backends are fakes injected into sys.modules; nothing here runs real TFHE.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")))
lc = importlib.import_module("library_comparison")

W8 = np.linspace(-1.0, 1.0, 8)
B8 = 0.1


class _BackendDown(RuntimeError):
    pass


def _raise(*args, **kwargs):
    raise _BackendDown("backend down")


def _install_concrete_ml(monkeypatch, *, compile_fails=False, predict_fails=False):
    class LogisticRegression:
        def __init__(self, n_bits):
            pass

        def fit(self, X, y):
            return self

        def compile(self, X):
            if compile_fails:
                _raise()

        def predict_proba(self, X, fhe="disable"):
            if predict_fails:
                _raise()
            return np.array([[0.4, 0.6]])

    pkg, ml = types.ModuleType("concrete"), types.ModuleType("concrete.ml")
    sklearn = types.ModuleType("concrete.ml.sklearn")
    sklearn.LogisticRegression = LogisticRegression
    pkg.ml, ml.sklearn = ml, sklearn
    for name, mod in (("concrete", pkg), ("concrete.ml", ml), ("concrete.ml.sklearn", sklearn)):
        monkeypatch.setitem(sys.modules, name, mod)


def _install_concrete_fhe(monkeypatch, *, compile_fails=False, run_fails=False):
    class Compiled:
        def __init__(self, fn):
            self.fn = fn

        def encrypt_run_decrypt(self, x):
            if run_fails:
                _raise()
            return self.fn(x)

    class Compiler:
        def __init__(self, fn):
            self.fn = fn

        def compile(self, inputset):
            if compile_fails:
                _raise()
            return Compiled(self.fn)

    pkg, fhe = types.ModuleType("concrete"), types.ModuleType("concrete.fhe")
    fhe.compiler = lambda spec: Compiler
    pkg.fhe = fhe
    monkeypatch.setitem(sys.modules, "concrete", pkg)
    monkeypatch.setitem(sys.modules, "concrete.fhe", fhe)


def test_concrete_ml_prediction_failure_raises(monkeypatch):
    _install_concrete_ml(monkeypatch, predict_fails=True)
    fhe_fn = lc.concrete_ml_lr_d8(W8, B8)
    with pytest.raises(_BackendDown):
        fhe_fn([0.0] * 8)


def test_concrete_ml_compile_failure_raises(monkeypatch):
    _install_concrete_ml(monkeypatch, compile_fails=True)
    with pytest.raises(_BackendDown):
        lc.concrete_ml_lr_d8(W8, B8)


def test_concrete_ml_success_returns_class_one_probability(monkeypatch):
    _install_concrete_ml(monkeypatch)
    assert lc.concrete_ml_lr_d8(W8, B8)([0.0] * 8) == 0.6


def test_concrete_execution_failure_raises(monkeypatch):
    _install_concrete_fhe(monkeypatch, run_fails=True)
    fhe_fn = lc.concrete_squared_dot(W8, B8)
    with pytest.raises(_BackendDown):
        fhe_fn([0.0] * 8)


def test_concrete_compile_failure_raises(monkeypatch):
    _install_concrete_fhe(monkeypatch, compile_fails=True)
    with pytest.raises(_BackendDown):
        lc.concrete_squared_dot(W8, B8)


def test_concrete_success_matches_plaintext_on_grid(monkeypatch):
    _install_concrete_fhe(monkeypatch)
    x = [0.2 * k for k in range(-4, 4)]
    expected = lc.unified_plaintext(np.round(W8 * 5) / 5, round(B8 * 25) / 25)(x)
    assert lc.concrete_squared_dot(W8, B8)(x) == pytest.approx(expected)


def test_run_one_error_row_has_no_fake_zero_error():
    row = lc.run_one("fake", "0", lambda x: 0.5, _raise, "c", seed=0, n_trials=60, threshold=1e-2)
    assert "backend down" in row.verdict
    assert math.isnan(row.max_error)
