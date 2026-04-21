# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Adapter for Zama Concrete ML / Concrete Python.

Installation
------------
    pip install concrete-ml

Example
-------
    from concrete.ml.sklearn import LogisticRegression
    from fhe_oracle.adapters.concrete import ConcreteAdapter

    model = LogisticRegression(n_bits=8)
    model.fit(X_train, y_train)
    model.compile(X_train)

    adapter = ConcreteAdapter(model.fhe_circuit)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import FHEAdapter


class ConcreteAdapter(FHEAdapter):
    """FHEAdapter wrapping a compiled Concrete Python circuit."""

    def __init__(
        self,
        circuit: Any,
        fhe_fn: Callable[[Any, Any], Any] | None = None,
        mult_depth: int | None = None,
    ) -> None:
        try:
            import concrete  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Concrete ML / Concrete Python is not installed. "
                "Install with: pip install concrete-ml"
            )

        self._circuit = circuit
        self._fhe_fn = fhe_fn
        self._mult_depth = mult_depth

    def encrypt(self, x: list[float]) -> Any:
        x_arr = np.array(x, dtype=np.float64)
        return self._circuit.encrypt(x_arr)

    def decrypt(self, ciphertext: Any) -> list[float]:
        result = self._circuit.decrypt(ciphertext)
        if hasattr(result, "tolist"):
            return result.tolist()
        if isinstance(result, (int, float, np.integer, np.floating)):
            return [float(result)]
        return [float(v) for v in result]

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
        except Exception:
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
        except Exception:
            pass
        return 1
