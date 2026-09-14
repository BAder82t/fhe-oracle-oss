# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitness functions for adversarial FHE testing.

Pure divergence fitness: ``|plain(x) - fhe(x)|``. Works without any
FHE-library-specific instrumentation and is the default strategy.

External plugins can register a noise-budget-aware fitness under the
name ``"noise_budget"`` in the :mod:`fhe_oracle.registry` ``fitness``
group; when registered, :class:`FHEOracle` will auto-dispatch to it
whenever an ``adapter`` is supplied.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class EvaluationError(ValueError):
    """An evaluation cannot support a precision verdict."""


def validated_outputs(plain, fhe) -> tuple[np.ndarray, np.ndarray]:
    """Require matching, nonempty, finite real outputs; never truncate."""
    try:
        p = np.atleast_1d(np.asarray(plain))
        f = np.atleast_1d(np.asarray(fhe))
        if np.iscomplexobj(p) or np.iscomplexobj(f):
            raise EvaluationError("outputs must be real-valued")
        p = p.astype(np.float64)
        f = f.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"invalid numeric output: {exc}") from exc
    if p.shape != f.shape:
        raise EvaluationError(f"output shape mismatch: {p.shape} != {f.shape}")
    if p.size == 0:
        raise EvaluationError("outputs must be nonempty")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(f)):
        raise EvaluationError("outputs must contain only finite values")
    return p, f


def absolute_error(plain, fhe) -> np.ndarray:
    p, f = validated_outputs(plain, fhe)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = np.abs(p - f)
    if not np.all(np.isfinite(diff)):
        raise EvaluationError("output divergence is not finite")
    return diff


def finite_score(value) -> float:
    score = float(value)
    if not np.isfinite(score):
        raise EvaluationError("fitness score must be finite")
    return score


def evaluate_outputs(plaintext_fn, fhe_fn, x) -> tuple[np.ndarray, np.ndarray]:
    try:
        plain, fhe = plaintext_fn(x), fhe_fn(x)
    except Exception as exc:
        raise EvaluationError(f"model evaluation failed: {exc}") from exc
    return validated_outputs(plain, fhe)


class DivergenceFitness:
    """Pure-divergence fitness: |plaintext_fn(x) - fhe_fn(x)|.

    Parameters
    ----------
    plaintext_fn : callable
        Reference plaintext implementation.
        ``plaintext_fn(x: list[float]) -> float | list[float]``.
    fhe_fn : callable
        FHE implementation under test. Same signature as plaintext_fn.
    output_reducer : callable, optional
        Applied to the absolute difference vector (e.g. ``np.max``,
        ``np.mean``). Default ``np.max`` — most sensitive to point
        precision bugs.
    """

    def __init__(
        self,
        plaintext_fn: Callable[[list[float]], float | list[float]],
        fhe_fn: Callable[[list[float]], float | list[float]],
        output_reducer: Callable[[np.ndarray], float] = np.max,
    ) -> None:
        self._plaintext_fn = plaintext_fn
        self._fhe_fn = fhe_fn
        self._reduce = output_reducer

    def score(self, x: Any) -> float:
        """Return divergence; invalid evaluations raise EvaluationError."""
        plain, fhe = evaluate_outputs(self._plaintext_fn, self._fhe_fn, x)
        return finite_score(self._reduce(absolute_error(plain, fhe)))


def _to_array(value) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)
