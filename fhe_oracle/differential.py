# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differential testing between two FHE adapters.

Searches for inputs where two independent FHE adapters disagree with
each other, using only ``encrypt``/``run_fhe_program``/``decrypt`` --
no noise-budget API (several libraries' are broken/incompatible for
CKKS; output is always well-defined).
"""

from __future__ import annotations

from typing import Any, Optional

from .core import FHEOracle, OracleResult
from .fitness import DivergenceFitness


class CrossAdapterFitness:
    """Fitness = divergence between two FHE adapters' decrypted outputs.

    Parameters
    ----------
    adapter_a, adapter_b : FHEAdapter
        Two independently-implemented FHE backends running the SAME
        circuit (each adapter's ``run_fhe_program`` should implement
        equivalent logic). Both must implement ``.evaluate(x)``
        (``FHEAdapter`` provides a default: encrypt -> run -> decrypt).
    """

    def __init__(self, adapter_a: Any, adapter_b: Any) -> None:
        self._a = adapter_a
        self._b = adapter_b

    def score(self, x: list[float]) -> float:
        """Compare complete outputs; invalid evaluations raise an error."""
        return DivergenceFitness(self._a.evaluate, self._b.evaluate).score(x)


def differential_test(
    adapter_a: Any,
    adapter_b: Any,
    input_dim: int,
    input_bounds: Optional[list[tuple[float, float]] | tuple[float, float]] = None,
    n_trials: int = 500,
    threshold: float = 1e-2,
    seed: Optional[int] = None,
    **oracle_kwargs: Any,
) -> OracleResult:
    """Search for an input where ``adapter_a`` and ``adapter_b`` disagree.

    Convenience wrapper around :class:`~fhe_oracle.core.FHEOracle` with
    a :class:`CrossAdapterFitness`. ``"FAIL"`` means the two adapters'
    outputs diverge by >= ``threshold`` -- no reference plaintext
    function needed.

    Parameters
    ----------
    adapter_a, adapter_b : FHEAdapter
        The two backends to compare, running the same circuit.
    input_dim : int
        Dimensionality of the input space.
    input_bounds : list[tuple[float, float]], optional
        Per-dimension ``(low, high)`` box constraints.
    n_trials : int
        Fitness-evaluation budget. Default 500.
    threshold : float
        PASS/FAIL cut-off on max divergence between the two adapters.
    seed : int, optional
        Random seed for reproducibility.
    **oracle_kwargs
        Forwarded to :class:`~fhe_oracle.core.FHEOracle` (e.g.
        ``restarts``, ``separable``).
    """
    fitness = CrossAdapterFitness(adapter_a, adapter_b)
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,  # unused: fitness overrides scoring
        input_dim=input_dim,
        input_bounds=input_bounds,
        fitness=fitness,
        seed=seed,
        **oracle_kwargs,
    )
    return oracle.run(n_trials=n_trials, threshold=threshold)
