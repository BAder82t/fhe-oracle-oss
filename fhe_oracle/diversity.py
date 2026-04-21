# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Periodic diversity injection for CMA-ES.

Injects diverse candidates into the CMA-ES population at regular
intervals to prevent covariance collapse on plateau landscapes. The
warm-start patch (random-floor + warm restart of CMA-ES from the
best random sample) helps only at *initialisation* -- once CMA-ES
contracts, it never re-explores. Periodic injection re-introduces
diverse candidates throughout the search.

Strategies
----------
- ``CORNER``         -- coordinates pinned to box bounds with
                        probability ``corner_prob``.
- ``UNIFORM``        -- uniform random from the full input box.
- ``BEST_NEIGHBOR``  -- Gaussian perturbation of the running best,
                        clipped to the box.
- ``MIXED`` (default)-- rotates through the three above, one per
                        injection round.

Integration
-----------
The injector is consumed inside the CMA-ES ask-tell loop. Two
integration patterns:

1. Replace the worst ``inject_count`` entries of ``ask()``'s
   solutions before evaluation.
2. Call ``es.inject(injections, force=True)`` and let pycma merge
   them into the next ask().

The :class:`fhe_oracle.core.FHEOracle` integration uses pattern 1
because it preserves the population size and avoids double-
counting evaluations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class InjectionStrategy(Enum):
    """Diversity-injection strategies."""

    CORNER = "corner"
    UNIFORM = "uniform"
    BEST_NEIGHBOR = "best_neighbor"
    MIXED = "mixed"


class DiversityInjector:
    """Manages periodic injection of diverse candidates.

    Parameters
    ----------
    bounds : list of (lo, hi)
        Per-dimension input box (length d).
    inject_every : int, default 5
        Inject every N generations.
    inject_count : int, default 3
        Candidates injected per round. Should be < CMA-ES popsize.
    strategy : InjectionStrategy, default MIXED
        Which generator to use each round.
    corner_prob : float, default 0.7
        For CORNER strategy, probability each coordinate is at the
        box boundary (vs interior uniform).
    neighbor_sigma : float, default 0.3
        For BEST_NEIGHBOR, perturbation scale relative to box
        half-width.
    """

    def __init__(
        self,
        bounds: list,
        inject_every: int = 5,
        inject_count: int = 3,
        strategy: InjectionStrategy = InjectionStrategy.MIXED,
        corner_prob: float = 0.7,
        neighbor_sigma: float = 0.3,
    ) -> None:
        if inject_every < 1:
            raise ValueError("inject_every must be >= 1")
        if inject_count < 1:
            raise ValueError("inject_count must be >= 1")
        self.bounds = list(bounds)
        self.d = len(bounds)
        self.lo = np.array([b[0] for b in bounds], dtype=np.float64)
        self.hi = np.array([b[1] for b in bounds], dtype=np.float64)
        self.half_width = (self.hi - self.lo) / 2.0
        self.mid = (self.lo + self.hi) / 2.0
        self.inject_every = int(inject_every)
        self.inject_count = int(inject_count)
        self.strategy = strategy
        self.corner_prob = float(corner_prob)
        self.neighbor_sigma = float(neighbor_sigma)
        self._strategy_cycle = 0

    def should_inject(self, generation: int) -> bool:
        """Fire on multiples of ``inject_every`` (skipping gen 0)."""
        return generation > 0 and (generation % self.inject_every == 0)

    def generate_injections(
        self, best_so_far: Any, rng: np.random.RandomState
    ) -> list[np.ndarray]:
        """Generate ``inject_count`` diverse candidates.

        Parameters
        ----------
        best_so_far : array-like
            Best input found so far (used by BEST_NEIGHBOR).
        rng : np.random.RandomState
            RNG for reproducibility.
        """
        if self.strategy == InjectionStrategy.MIXED:
            cycle = [
                InjectionStrategy.CORNER,
                InjectionStrategy.UNIFORM,
                InjectionStrategy.BEST_NEIGHBOR,
            ]
            current = cycle[self._strategy_cycle % 3]
            self._strategy_cycle += 1
        else:
            current = self.strategy
        best_arr = np.asarray(best_so_far, dtype=np.float64).ravel()
        if best_arr.size != self.d:
            best_arr = self.mid.copy()

        out: list[np.ndarray] = []
        for _ in range(self.inject_count):
            if current == InjectionStrategy.CORNER:
                out.append(self._corner_sample(rng))
            elif current == InjectionStrategy.UNIFORM:
                out.append(self._uniform_sample(rng))
            else:  # BEST_NEIGHBOR
                out.append(self._neighbor_sample(best_arr, rng))
        return out

    def _corner_sample(self, rng: np.random.RandomState) -> np.ndarray:
        boundary_pick = rng.rand(self.d) < self.corner_prob
        side = rng.rand(self.d) < 0.5
        x = np.where(boundary_pick, np.where(side, self.hi, self.lo),
                     rng.uniform(self.lo, self.hi))
        return x.astype(np.float64)

    def _uniform_sample(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(self.lo, self.hi).astype(np.float64)

    def _neighbor_sample(
        self, best: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        perturbation = rng.randn(self.d) * self.half_width * self.neighbor_sigma
        return np.clip(best + perturbation, self.lo, self.hi).astype(np.float64)
