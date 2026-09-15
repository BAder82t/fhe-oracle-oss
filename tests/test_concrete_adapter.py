# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ConcreteAdapter and the predict_proba route against fakes of the Concrete / Concrete ML API.

concrete-ml is not installed (no Python 3.13 support); nothing here runs real TFHE.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.adapters.concrete import ConcreteAdapter
from fhe_oracle.fitness import EvaluationError

W4 = np.array([0.8, -1.2, 0.5, 1.1])
B4 = 0.2
X_STEP = 0.01
W_STEP = 0.01
BOUNDS = [(-3.0, 3.0)] * 4


class _FakeCircuit:
    """Mimics concrete.fhe.Circuit: integer-only encrypt, then run and decrypt."""

    def __init__(self, shape):
        self.shape = shape
        self.q_w = np.round(W4 / W_STEP).astype(np.int64)
        self.q_b = int(round(B4 / (X_STEP * W_STEP)))
        self.statistics = {"programmable_bootstrap_count": 0}
        self.inputs: list[np.ndarray] = []

    def encrypt(self, q_x):
        arr = np.asarray(q_x)
        if not np.issubdtype(arr.dtype, np.integer) or arr.shape != self.shape:
            raise ValueError(
                f"Expected argument 0 to be EncryptedTensor<int, shape={self.shape}> "
                f"but it's {arr.dtype}{arr.shape}"
            )
        self.inputs.append(arr)
        return ("encrypted", arr)

    def run(self, value):
        return ("encrypted", value[1] @ self.q_w[:, None] + self.q_b)

    def decrypt(self, value):
        return value[1]


class _FakeLogisticRegression:
    """Mimics a compiled binary concrete.ml.sklearn.LogisticRegression."""

    def __init__(self, fail: bool = False):
        self.fhe_circuit = _FakeCircuit((1, len(W4)))
        self.calls: list[tuple[tuple[int, ...], str]] = []
        self.fail = fail

    def quantize_input(self, X):
        return np.round(np.asarray(X, dtype=np.float64) / X_STEP).astype(np.int64)

    def dequantize_output(self, q_y):
        return np.asarray(q_y, dtype=np.float64) * X_STEP * W_STEP

    def post_processing(self, y):
        p = 1.0 / (1.0 + np.exp(-np.asarray(y).ravel()))
        return np.vstack([1 - p, p]).T

    def predict_proba(self, X, fhe="disable"):
        X = np.asarray(X, dtype=np.float64)
        self.calls.append((X.shape, fhe))
        if self.fail:
            raise RuntimeError("keygen failed")
        c = self.fhe_circuit
        q_y = np.array([c.decrypt(c.run(c.encrypt(q[None])))[0] for q in self.quantize_input(X)])
        return self.post_processing(self.dequantize_output(q_y))


@pytest.fixture
def fake_concrete(monkeypatch):
    monkeypatch.setitem(sys.modules, "concrete", types.ModuleType("concrete"))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-(W4 @ np.asarray(x, dtype=np.float64) + B4))))


def _quantised_adapter(model):
    return ConcreteAdapter(
        model.fhe_circuit,
        quantize_fn=lambda x: model.quantize_input(np.reshape(x, (1, -1))),
        dequantize_fn=lambda q: model.post_processing(model.dequantize_output(q))[:, 1],
    )


def test_float_inputs_need_quantize_fn(fake_concrete):
    model = _FakeLogisticRegression()
    adapter = ConcreteAdapter(model.fhe_circuit)
    with pytest.raises(ValueError, match="quantize_fn"):
        adapter.encrypt([0.25, -0.5, 1.0, 0.1])
    assert model.fhe_circuit.inputs == []


def test_integer_inputs_pass_through_without_quantize_fn(fake_concrete):
    circuit = _FakeCircuit((4,))
    x = [1.0, -2.0, 3.0, 0.0]
    expected = float(circuit.q_w @ np.array(x, dtype=np.int64) + circuit.q_b)
    assert ConcreteAdapter(circuit).evaluate(x) == [expected]
    assert circuit.inputs[0].dtype == np.int64


def test_quantised_adapter_matches_predict_proba(fake_concrete):
    model = _FakeLogisticRegression()
    x = [0.25, -0.5, 1.0, 0.1]
    expected = model.predict_proba(np.reshape(x, (1, -1)), fhe="execute")[0, 1]
    assert _quantised_adapter(model).evaluate(x) == [expected]
    assert all(a.dtype == np.int64 and a.shape == (1, 4) for a in model.fhe_circuit.inputs)


def test_quantised_adapter_through_oracle(fake_concrete):
    model = _FakeLogisticRegression()
    result = FHEOracle(
        plaintext_fn=_sigmoid,
        adapter=_quantised_adapter(model),
        input_dim=4,
        input_bounds=BOUNDS,
        seed=0,
    ).run(n_trials=40, threshold=0.05)
    assert result.verdict == "PASS"
    assert 0.0 < result.max_error < 0.05


def test_predict_proba_route_simulate_then_execute():
    from fhe_oracle.adapters.concrete import predict_proba_fhe_fn

    model = _FakeLogisticRegression()
    result = FHEOracle(
        plaintext_fn=_sigmoid,
        fhe_fn=predict_proba_fhe_fn(model, fhe="simulate"),
        input_dim=4,
        input_bounds=BOUNDS,
        seed=0,
    ).run(n_trials=40, threshold=0.05)
    assert set(model.calls) == {((1, 4), "simulate")}

    witness = result.worst_input
    confirmed = abs(_sigmoid(witness) - predict_proba_fhe_fn(model)(witness))
    assert model.calls[-1] == ((1, 4), "execute")
    assert confirmed == pytest.approx(result.max_error)


def test_predict_proba_fhe_fn_rejects_unknown_mode():
    from fhe_oracle.adapters.concrete import predict_proba_fhe_fn

    with pytest.raises(ValueError, match="fhe"):
        predict_proba_fhe_fn(_FakeLogisticRegression(), fhe="exec")


def test_predict_proba_fhe_fn_propagates_backend_errors():
    from fhe_oracle.adapters.concrete import predict_proba_fhe_fn

    oracle = FHEOracle(
        plaintext_fn=_sigmoid,
        fhe_fn=predict_proba_fhe_fn(_FakeLogisticRegression(fail=True)),
        input_dim=4,
        input_bounds=BOUNDS,
        seed=0,
    )
    with pytest.raises(EvaluationError, match="keygen failed"):
        oracle.run(n_trials=10)
