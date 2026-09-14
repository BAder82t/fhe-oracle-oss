# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Property-based fitness functions for FHE Oracle.

``FHEOracle``'s fitness is pluggable -- these are ready-to-use fitness
objects that search for inputs violating an algebraic property (e.g.
additivity) instead of plaintext-vs-FHE divergence. Multi-argument
properties pack candidates into one search vector; see each class's
``dim``/``input_dim`` note.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .fitness import absolute_error, validated_outputs


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


class AdditivityFitness:
    """Search for a, b where ``fn(a + b) != fn(a) + fn(b)``.

    For testing FHE plumbing (encrypt/add/decrypt), not nonlinear
    model circuits -- those aren't additive and will "fail" everywhere.

    Parameters
    ----------
    fn : callable
        The function under test. ``fn(x: list[float]) -> float | list[float]``.
    dim : int
        Dimensionality of a single candidate (a or b). Use
        ``FHEOracle(..., input_dim=2 * dim, ...)`` -- the search vector
        packs ``[a, b]``.
    """

    def __init__(self, fn: Callable, dim: int) -> None:
        self._fn = fn
        self._dim = int(dim)

    def score(self, x: list[float]) -> float:
        arr = np.asarray(x, dtype=np.float64)
        a, b = arr[: self._dim], arr[self._dim : 2 * self._dim]
        out_a, out_b = validated_outputs(self._fn(a.tolist()), self._fn(b.tolist()))
        return float(absolute_error(self._fn((a + b).tolist()), out_a + out_b).max())


class ScalarLinearityFitness:
    """Search for x, c where ``fn(c * x) != c * fn(x)``.

    Parameters
    ----------
    fn : callable
        The function under test.
    dim : int
        Dimensionality of ``x``. Use ``FHEOracle(..., input_dim=dim+1, ...)``
        -- the search vector packs ``[x..., c]`` (``c`` is the last
        element).
    c_bounds : tuple[float, float]
        Clipping range applied to the scalar ``c`` component before
        evaluating. Default ``(-3.0, 3.0)``.
    """

    def __init__(
        self,
        fn: Callable,
        dim: int,
        c_bounds: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        self._fn = fn
        self._dim = int(dim)
        self._c_lo, self._c_hi = c_bounds

    def score(self, x: list[float]) -> float:
        arr = np.asarray(x, dtype=np.float64)
        x_part = arr[: self._dim]
        c = float(np.clip(arr[self._dim], self._c_lo, self._c_hi))
        out_scaled = self._fn((c * x_part).tolist())
        out_base = _to_array(self._fn(x_part.tolist()))
        diff = absolute_error(out_scaled, c * out_base)
        return float(np.max(diff))
