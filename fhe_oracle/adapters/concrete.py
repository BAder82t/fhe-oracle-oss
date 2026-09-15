# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapter for Zama Concrete ML / Concrete Python.

Installation
------------
    pip install concrete-ml

Supported route
---------------
Wrap a compiled Concrete ML classifier as ``fhe_fn``. Search cheaply with
``fhe="simulate"``, then re-measure the witness with ``fhe="execute"``::

    from fhe_oracle import FHEOracle
    from fhe_oracle.adapters.concrete import predict_proba_fhe_fn

    result = FHEOracle(
        plaintext_fn, fhe_fn=predict_proba_fhe_fn(model, fhe="simulate"),
        input_dim=d, input_bounds=bounds,
    ).run(n_trials=500)
    w = result.worst_input
    confirmed_error = abs(plaintext_fn(w) - predict_proba_fhe_fn(model)(w))

``ConcreteAdapter`` splits encrypt/run/decrypt for a compiled circuit, which takes
integer inputs; Concrete ML models need ``quantize_fn`` and ``dequantize_fn``::

    adapter = ConcreteAdapter(
        model.fhe_circuit,
        quantize_fn=lambda x: model.quantize_input(x.reshape(1, -1)),
        dequantize_fn=lambda q: model.post_processing(model.dequantize_output(q))[:, 1],
    )

Unit tests use fakes of the Concrete API; both paths were checked by hand with
concrete-ml 1.9.0 / concrete-python 2.10.0 (Python 3.11, LogisticRegression).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import FHEAdapter

_FHE_MODES = ("execute", "simulate", "disable")


def predict_proba_fhe_fn(
    model: Any, fhe: str = "execute", class_index: int = 1
) -> Callable[[Any], float]:
    """Return ``fhe_fn(x)``: ``model.predict_proba`` on one row, column ``class_index``."""
    if fhe not in _FHE_MODES:
        raise ValueError(f"fhe must be one of {_FHE_MODES}, got {fhe!r}")

    def fhe_fn(x: Any) -> float:
        row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return float(np.asarray(model.predict_proba(row, fhe=fhe))[0, class_index])

    return fhe_fn


class ConcreteAdapter(FHEAdapter):
    """FHEAdapter wrapping a compiled Concrete Python circuit."""

    def __init__(
        self,
        circuit: Any,
        fhe_fn: Callable[[Any, Any], Any] | None = None,
        mult_depth: int | None = None,
        quantize_fn: Callable[[np.ndarray], Any] | None = None,
        dequantize_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        try:
            import concrete  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Concrete ML / Concrete Python is not installed. "
                "Install with: pip install concrete-ml"
            ) from exc

        self._circuit = circuit
        self._fhe_fn = fhe_fn
        self._mult_depth = mult_depth
        self._quantize_fn = quantize_fn
        self._dequantize_fn = dequantize_fn

    def encrypt(self, x: list[float]) -> Any:
        x_arr = np.asarray(x, dtype=np.float64)
        if self._quantize_fn is not None:
            return self._circuit.encrypt(self._quantize_fn(x_arr))
        if not (np.all(np.isfinite(x_arr)) and np.array_equal(x_arr, np.round(x_arr))):
            raise ValueError(
                "Concrete circuits take integer inputs; pass quantize_fn "
                "(e.g. model.quantize_input) for real-valued inputs"
            )
        return self._circuit.encrypt(x_arr.astype(np.int64))

    def decrypt(self, ciphertext: Any) -> list[float]:
        result = self._circuit.decrypt(ciphertext)
        if self._dequantize_fn is not None:
            result = self._dequantize_fn(result)
        return np.asarray(result, dtype=np.float64).ravel().tolist()

    def run_fhe_program(self, ciphertext: Any) -> Any:
        if self._fhe_fn is not None:
            return self._fhe_fn(self._circuit, ciphertext)
        return self._circuit.run(ciphertext)

    def get_noise_budget(self, ciphertext: Any) -> float:
        try:
            stats = self._circuit.statistics
            pbs_count = getattr(stats, "pbs_count", None) or getattr(
                stats, "global_p_error", None
            )
            if pbs_count is not None and isinstance(pbs_count, (int, float)):
                depth = self._effective_depth()
                if depth > 0:
                    return max(0.0, float(pbs_count) / depth * 100.0)
        except Exception:  # noqa: BLE001, S110 - metadata-only fallback; max_error and verdict unaffected
            pass
        depth = self._effective_depth()
        return max(0.0, 100.0 - depth * 10.0)

    def get_mult_depth_used(self, ciphertext: Any) -> int:
        return self._effective_depth()

    def get_scheme_name(self) -> str:
        return "Concrete-TFHE"

    def _effective_depth(self) -> int:
        if self._mult_depth is not None:
            return self._mult_depth
        try:
            stats = self._circuit.statistics
            for attr in ("multiplicative_depth", "depth", "circuit_depth"):
                val = getattr(stats, attr, None)
                if val is not None:
                    return int(val)
        except Exception:  # noqa: BLE001, S110 - metadata-only fallback; max_error and verdict unaffected
            pass
        return 1
