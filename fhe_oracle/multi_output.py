# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Multi-output fitness for vector-valued FHE circuits.

Extends scalar divergence with rank-aware objectives that target
decision-altering precision failures. Useful for classifiers that
output a vector of class probabilities: a small absolute error can
still flip the argmax (decision boundary inversion).

Modes
-----
1. ``MAX_ABSOLUTE`` -- ``max_i |p_i - f_i|`` (matches the existing
   default scalar fitness).
2. ``RANK_INVERSION`` -- 0 if argmax matches, otherwise positive
   penalty proportional to per-output error plus an inversion
   bonus. When argmax matches, returns inverse top-2 margin so the
   search prefers inputs that almost flip the prediction.
3. ``COMBINED`` -- ``MAX_ABSOLUTE + rank_weight * RANK_INVERSION``.

Example
-------
    from fhe_oracle.multi_output import MultiOutputFitness, \\
        MultiOutputMode

    fitness = MultiOutputFitness(
        plaintext_fn=clf.predict_proba,
        fhe_fn=fhe_clf.predict_proba,
        mode=MultiOutputMode.COMBINED,
        rank_weight=1.0,
    )
    score = fitness(x)
    report = fitness.detailed_report(x)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import numpy as np


class MultiOutputMode(Enum):
    """Fitness mode for vector-valued outputs."""

    MAX_ABSOLUTE = "max_absolute"
    RANK_INVERSION = "rank_inversion"
    COMBINED = "combined"


def _to_vector(value: Any) -> np.ndarray:
    """Coerce a callable's output to a 1-D float array."""
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    return arr


class MultiOutputFitness:
    """Fitness for vector-valued FHE circuits.

    Parameters
    ----------
    plaintext_fn : callable
        ``x -> array-like of shape (k,)``.
    fhe_fn : callable
        Same signature as ``plaintext_fn``.
    mode : MultiOutputMode, default COMBINED
        Which fitness to use.
    rank_weight : float, default 1.0
        Weight on the rank-inversion term in COMBINED mode.
    margin_bonus : float, default 0.5
        Reserved -- present for forward compatibility with future
        margin-aware variants. Currently unused.
    """

    def __init__(
        self,
        plaintext_fn: Callable,
        fhe_fn: Callable,
        mode: MultiOutputMode = MultiOutputMode.COMBINED,
        rank_weight: float = 1.0,
        margin_bonus: float = 0.5,
    ) -> None:
        self.plaintext_fn = plaintext_fn
        self.fhe_fn = fhe_fn
        self.mode = mode
        self.rank_weight = float(rank_weight)
        self.margin_bonus = float(margin_bonus)

    def __call__(self, x: Any) -> float:
        return self.score(x)

    def score(self, x: Any) -> float:
        """Compute fitness at ``x``. Exceptions return 0.0."""
        try:
            p = _to_vector(self.plaintext_fn(x))
            f = _to_vector(self.fhe_fn(x))
        except Exception:
            return 0.0
        n = min(p.size, f.size)
        if n == 0:
            return 0.0
        p = p[:n]
        f = f[:n]
        if self.mode == MultiOutputMode.MAX_ABSOLUTE:
            return self._max_absolute(p, f)
        if self.mode == MultiOutputMode.RANK_INVERSION:
            return self._rank_inversion(p, f)
        return self._combined(p, f)

    @staticmethod
    def _max_absolute(p: np.ndarray, f: np.ndarray) -> float:
        return float(np.max(np.abs(p - f)))

    @staticmethod
    def _rank_inversion(p: np.ndarray, f: np.ndarray) -> float:
        """Rank-inversion score.

        Tiered so a flip ALWAYS beats no-flip:
        - argmax mismatch (FLIPPED): ``100 + max|p-f|``. The +100
          guarantees flipped samples dominate any non-flip score.
        - argmax match (no flip): ``1 / (margin + 0.01)`` capped at
          ~100. Small margins still attract the search but cannot
          exceed an actual flip.

        The earlier (v0.4.0) version used unbounded ``1/(margin+1e-8)``
        which trapped CMA-ES at near-boundary non-flip inputs because
        the margin term grew unbounded faster than the inversion
        bonus.
        """
        if p.size == 1 or f.size == 1:
            # Scalar output -- no ranking, fall back to absolute.
            return float(np.abs(p - f).max())
        p_class = int(np.argmax(p))
        f_class = int(np.argmax(f))
        if p_class != f_class:
            return 100.0 + float(np.max(np.abs(p - f)))
        f_sorted = np.sort(f)[::-1]
        margin = float(f_sorted[0] - f_sorted[1])
        return 1.0 / (margin + 0.01)

    def _combined(self, p: np.ndarray, f: np.ndarray) -> float:
        return self._max_absolute(p, f) + self.rank_weight * self._rank_inversion(p, f)

    def detailed_report(self, x: Any) -> dict:
        """Diagnostic report at a specific input."""
        p = _to_vector(self.plaintext_fn(x))
        f = _to_vector(self.fhe_fn(x))
        n = min(p.size, f.size)
        p = p[:n]
        f = f[:n]
        report: dict = {
            "plaintext_output": p.tolist(),
            "fhe_output": f.tolist(),
            "per_output_error": np.abs(p - f).tolist(),
            "max_absolute_error": float(np.max(np.abs(p - f))) if n > 0 else 0.0,
        }
        if n >= 2:
            p_class = int(np.argmax(p))
            f_class = int(np.argmax(f))
            f_sorted_idx = np.argsort(f)[::-1]
            report.update(
                plaintext_class=p_class,
                fhe_class=f_class,
                decision_flipped=p_class != f_class,
                fhe_top2_margin=float(f[f_sorted_idx[0]] - f[f_sorted_idx[1]]),
                fhe_ranking=f_sorted_idx.tolist(),
            )
        else:
            report.update(
                plaintext_class=None,
                fhe_class=None,
                decision_flipped=False,
                fhe_top2_margin=None,
                fhe_ranking=None,
            )
        return report
