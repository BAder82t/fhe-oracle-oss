# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitness functions for adversarial FHE testing.

Pure divergence fitness: ``|plain(x) - fhe(x)|``. Works without any
FHE-library-specific instrumentation and is the default strategy for
the open-source edition.

Noise-guided (noise-budget-aware) fitness is part of
``fhe-oracle-pro`` and registers under the name ``"noise_budget"``
in the :mod:`fhe_oracle.registry` ``fitness`` group. When Pro is
installed, :class:`FHEOracle` auto-dispatches to it whenever an
``adapter`` is supplied; otherwise Core raises a pointer to the
commercial edition.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


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

    def score(self, x: list[float]) -> float:
        """Return the divergence at x. Exceptions return 0.0."""
        try:
            plain = _to_array(self._plaintext_fn(x))
            fhe = _to_array(self._fhe_fn(x))
        except Exception:
            return 0.0

        if plain.shape != fhe.shape:
            n = min(plain.size, fhe.size)
            plain = plain.ravel()[:n]
            fhe = fhe.ravel()[:n]

        diff = np.abs(plain - fhe)
        if diff.size == 0:
            return 0.0
        return float(self._reduce(diff))


def _to_array(value) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)
