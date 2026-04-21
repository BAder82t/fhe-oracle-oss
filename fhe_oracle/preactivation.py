# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Preactivation-subspace adversarial search for affine front-ends.

For any model whose first layer is affine — ``z = W·x + b`` followed
by an arbitrary nonlinearity — the FHE divergence

    delta(x) = |plain(x) - fhe(x)|

depends on x **only through the preactivation z**. The effective
search dimension is therefore ``rank(W)`` (typically the head
dimension k), not the raw input dimension d. For a logistic-regression
classifier (k=1), the d=784 search space collapses to a 1-D line
search.

This module wraps :class:`fhe_oracle.core.FHEOracle` so the underlying
CMA-ES runs in R^k instead of R^d. Candidates are projected back to
x-space via the Moore-Penrose pseudoinverse, clipped to the input
bounds. A clip-distance penalty discourages the search from drifting
into infeasible z-regions.

Why this matters
----------------
At d=128 the published full-d CMA-ES advantage over uniform random is
~1.16x at budget B=2000. At d=784 the same approach would need
B=50d=39200 evaluations and thousands of CKKS calls per seed —
infeasible in this paper's compute envelope. Searching in 1-D
preactivation space at B=50 recovers the same advantage in seconds.

Usage
-----
    from fhe_oracle.preactivation import PreactivationOracle

    pre = PreactivationOracle(
        W=W,                 # (k, d) weight matrix
        b=b,                 # (k,) bias
        plaintext_fn=plain,
        fhe_fn=fhe,
        input_bounds=[(-3.0, 3.0)] * d,
    )
    results = pre.run(budget=50, seeds=range(1, 11))
    # results is a list of PreactivationResult dataclasses; each has
    # max_error, x, z, clip_distance, n_trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .core import FHEOracle
from .fitness import DivergenceFitness


@dataclass
class PreactivationResult:
    """Single-seed preactivation-search outcome."""

    seed: int
    max_error: float
    x: list[float]
    z: list[float]
    clip_distance: float
    n_trials: int
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class _PreactivationFitness:
    """Fitness in z-space with optional clip-distance penalty.

    F(z) = |plain(x) - fhe(x)| - clip_penalty * ||x_raw - x_clipped||

    where ``x_raw = W^+(z - b)`` and ``x_clipped`` is x_raw clipped to
    the input box. Exceptions yield 0.0 so a single bad evaluation
    doesn't crash CMA-ES.
    """

    def __init__(
        self,
        z_to_x: Callable[[np.ndarray], tuple[np.ndarray, float]],
        plaintext_fn: Callable,
        fhe_fn: Callable,
        clip_penalty: float,
        output_reducer: Callable[[np.ndarray], float] = np.max,
    ) -> None:
        self._z_to_x = z_to_x
        self._plain = plaintext_fn
        self._fhe = fhe_fn
        self._clip_penalty = float(clip_penalty)
        self._reduce = output_reducer

    def score(self, z) -> float:
        try:
            z_arr = np.asarray(z, dtype=np.float64).ravel()
            x, clip_dist = self._z_to_x(z_arr)
            plain = _to_array(self._plain(x))
            fhe = _to_array(self._fhe(x))
            n = min(plain.size, fhe.size)
            if n == 0:
                return 0.0
            diff = np.abs(plain.ravel()[:n] - fhe.ravel()[:n])
            div = float(self._reduce(diff)) if diff.size > 0 else 0.0
            return div - self._clip_penalty * float(clip_dist)
        except Exception:
            return 0.0


class PreactivationOracle:
    """Search worst-case FHE divergence in preactivation space.

    Parameters
    ----------
    W : array-like, shape (k, d)
        Weight matrix of the affine front-end.
    b : array-like, shape (k,)
        Bias vector of the affine front-end.
    plaintext_fn : callable
        Reference plaintext implementation, ``plain(x) -> scalar | array``.
    fhe_fn : callable
        FHE implementation under test, same signature as plaintext_fn.
    input_bounds : list of (low, high) per input dim
        Box constraints on x.
    clip_penalty : float, default 0.1
        Penalty coefficient on the L2 distance between the raw
        projected x and its bounds-clipped version. Discourages CMA-ES
        from drifting into z-regions whose preimage lies outside the
        input box. Set to 0.0 to disable.
    output_reducer : callable, default np.max
        Reduction over the absolute-difference vector.
    """

    def __init__(
        self,
        W,
        b,
        plaintext_fn: Callable,
        fhe_fn: Callable,
        input_bounds: list[tuple[float, float]],
        clip_penalty: float = 0.1,
        output_reducer: Callable[[np.ndarray], float] = np.max,
    ) -> None:
        self.W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        self.b = np.atleast_1d(np.asarray(b, dtype=np.float64)).astype(np.float64)
        self.k, self.d = self.W.shape
        if self.b.size != self.k:
            raise ValueError(
                f"bias has {self.b.size} entries, expected k={self.k}"
            )
        if len(input_bounds) != self.d:
            raise ValueError(
                f"input_bounds has length {len(input_bounds)}, expected d={self.d}"
            )
        self.W_pinv = np.linalg.pinv(self.W)        # shape (d, k)
        self.lb = np.array([float(lo) for lo, _ in input_bounds])
        self.ub = np.array([float(hi) for _, hi in input_bounds])
        self.input_bounds = list(input_bounds)
        self._plain = plaintext_fn
        self._fhe = fhe_fn
        self._clip_penalty = float(clip_penalty)
        self._reduce = output_reducer

    # ----- Geometry helpers -----------------------------------------

    def z_to_x(self, z) -> tuple[np.ndarray, float]:
        """Map z -> (x_clipped, ||x_raw - x_clipped||).

        x_raw = W^+ (z - b); x_clipped = clip(x_raw, lb, ub).
        """
        z_arr = np.asarray(z, dtype=np.float64).ravel()
        x_raw = self.W_pinv @ (z_arr - self.b)
        x_clipped = np.clip(x_raw, self.lb, self.ub)
        clip_dist = float(np.linalg.norm(x_raw - x_clipped))
        return x_clipped, clip_dist

    def z_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds on z induced by the input box.

        For each row k: z_k(x) = sum_j W[k,j] x_j + b[k]. With box
        constraints, the per-coordinate extremes are reached at the
        sign-aligned corner.
        """
        z_min = np.zeros(self.k)
        z_max = np.zeros(self.k)
        for i in range(self.k):
            w = self.W[i]
            pos_lo = np.where(w >= 0.0, w * self.lb, w * self.ub)
            pos_hi = np.where(w >= 0.0, w * self.ub, w * self.lb)
            z_min[i] = float(np.sum(pos_lo) + self.b[i])
            z_max[i] = float(np.sum(pos_hi) + self.b[i])
        return z_min, z_max

    # ----- Runs -----------------------------------------------------

    def _build_fitness(self) -> _PreactivationFitness:
        return _PreactivationFitness(
            z_to_x=self.z_to_x,
            plaintext_fn=self._plain,
            fhe_fn=self._fhe,
            clip_penalty=self._clip_penalty,
            output_reducer=self._reduce,
        )

    def measure_divergence_at(self, x) -> float:
        """Pure |plain(x) - fhe(x)| at a given x, reducer-aggregated."""
        try:
            plain = _to_array(self._plain(x))
            fhe = _to_array(self._fhe(x))
            n = min(plain.size, fhe.size)
            if n == 0:
                return 0.0
            diff = np.abs(plain.ravel()[:n] - fhe.ravel()[:n])
            return float(self._reduce(diff)) if diff.size > 0 else 0.0
        except Exception:
            return 0.0

    def run(
        self,
        budget: int = 50,
        seeds: Iterable[int] = range(1, 11),
        random_floor: float = 0.0,
        warm_start: bool = True,
        sigma0: float = 1.0,
        separable: bool = False,
        **oracle_kwargs: Any,
    ) -> list[PreactivationResult]:
        """Run search in z-space across seeds.

        Uses CMA-ES via :class:`FHEOracle` for k>=2; falls back to a
        grid + uniform-random sampler at k=1 because pycma's
        ``_stds_into_limits`` is unstable in 1-D under our bounds
        (raises ``ValueError: not yet initialized``).
        """
        if self.k == 1:
            return self._run_1d(budget=budget, seeds=seeds)

        z_lo, z_hi = self.z_bounds()
        z_bounds_list = list(zip(z_lo.tolist(), z_hi.tolist()))
        fitness = self._build_fitness()

        # Dummy plain/fhe in z-space for FHEOracle's post-run divergence
        # check; the search itself uses the custom fitness above. We
        # overwrite max_error/worst_input after the run so these are
        # only used to keep the inner oracle's invariants happy.
        _zero_fn = lambda _z: 0.0

        results: list[PreactivationResult] = []
        for seed in seeds:
            oracle = FHEOracle(
                plaintext_fn=_zero_fn,
                fhe_fn=_zero_fn,
                input_dim=self.k,
                input_bounds=z_bounds_list,
                fitness=fitness,
                sigma0=sigma0,
                seed=int(seed),
                random_floor=float(random_floor),
                warm_start=bool(warm_start),
                separable=bool(separable),
                **oracle_kwargs,
            )
            res = oracle.run(n_trials=int(budget), threshold=0.0)
            best_z = np.asarray(res.worst_input, dtype=np.float64).ravel()
            best_x, clip_dist = self.z_to_x(best_z)
            true_err = self.measure_divergence_at(best_x)
            results.append(PreactivationResult(
                seed=int(seed),
                max_error=true_err,
                x=best_x.tolist(),
                z=best_z.tolist(),
                clip_distance=clip_dist,
                n_trials=res.n_trials,
                elapsed_seconds=res.elapsed_seconds,
            ))
        return results

    def _run_1d(
        self,
        budget: int,
        seeds: Iterable[int],
    ) -> list[PreactivationResult]:
        """1-D search: dense grid + uniform-random refinement.

        Allocates 60% of the budget to a uniform grid over z-bounds and
        40% to random samples concentrated near the grid's best point.
        Each seed uses a different RNG offset for the random phase, so
        seed-variance is preserved.
        """
        import time

        z_lo, z_hi = self.z_bounds()
        lo, hi = float(z_lo[0]), float(z_hi[0])
        fitness = self._build_fitness()

        n_grid = max(8, int(round(0.6 * budget)))
        n_rand = max(1, budget - n_grid)
        grid = np.linspace(lo, hi, n_grid)

        results: list[PreactivationResult] = []
        for seed in seeds:
            t0 = time.perf_counter()
            best_score = -np.inf
            best_z = grid[0]
            n_evals = 0

            for z_val in grid:
                s = fitness.score(np.array([float(z_val)]))
                n_evals += 1
                if s > best_score:
                    best_score = s
                    best_z = float(z_val)

            # Random refinement near the grid's best.
            rng = np.random.default_rng(int(seed))
            span = (hi - lo) / max(1, n_grid - 1)
            for _ in range(n_rand):
                z_val = best_z + float(rng.normal(0.0, span))
                z_val = float(np.clip(z_val, lo, hi))
                s = fitness.score(np.array([z_val]))
                n_evals += 1
                if s > best_score:
                    best_score = s
                    best_z = z_val

            best_x, clip_dist = self.z_to_x(np.array([best_z]))
            true_err = self.measure_divergence_at(best_x)
            results.append(PreactivationResult(
                seed=int(seed),
                max_error=true_err,
                x=best_x.tolist(),
                z=[best_z],
                clip_distance=clip_dist,
                n_trials=n_evals,
                elapsed_seconds=time.perf_counter() - t0,
            ))
        return results


# --- helpers --------------------------------------------------------


def _to_array(value) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)
