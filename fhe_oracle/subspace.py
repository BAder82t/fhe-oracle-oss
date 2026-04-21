# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Random subspace embedding for high-dimensional FHE oracle search.

For circuits with ``d >> 100`` where preactivation factorisation is
unavailable (no affine front-end, or ``W, b`` not known), search in a
random ``k``-dimensional subspace of the input space instead of the
full ``R^d``.

v0.3.1 geometry overhaul
------------------------
The v0.3.0 implementation lost to uniform random on three of four
high-d benchmark circuits. Two root causes:

1. ``_z_bounds`` intersected per-axis input-box constraints, shrinking
   the reachable z-polytope to a sliver in high d. The new
   :meth:`_z_bounds_ball` uses a symmetric ball whose radius is the
   median (not minimum) per-axis reach, then steers CMA-ES toward
   in-bounds inputs via a clip-distance penalty (matching the strategy
   already used by :class:`PreactivationOracle`).

2. The search anchored exclusively at the box midpoint, while CKKS
   bugs concentrate near box corners (large operand magnitudes). The
   new :meth:`_generate_anchors` produces midpoint + corner-biased
   anchors and the search splits its budget across them.

Plus a sanity fallback: after each (projection, anchor) sub-search, if
the best found divergence still trails the pre-search random probe by
less than 10%, the remaining budget reverts to pure random sampling.

Theory
------
Johnson-Lindenstrauss guarantees that a random projection
``R in R^{d x k}`` with orthonormal columns preserves pairwise
distances between any ``n`` points in ``R^d`` up to ``(1 +/- eps)``
with probability ``>= 1 - delta``. In practice ``k=50`` works well
because we only need top-k *ranking* preserved; FHE divergence
landscapes tend to be low-rank; and multiple random projections can
be tried cheaply.

Geometry
--------
Each (projection, anchor) parameterises::

    x = anchor + R @ z,   z in R^k

with ``z`` drawn from a ball whose radius reaches the input-box
boundary along the median direction. CMA-ES minimises::

    -|plaintext_fn(clip(x)) - fhe_fn(clip(x))| + clip_penalty * ||x - clip(x)||

so out-of-bounds candidates are penalised but not hard-blocked.

Example
-------
    from fhe_oracle.subspace import SubspaceOracle

    oracle = SubspaceOracle(
        plaintext_fn=f, fhe_fn=f_tilde,
        bounds=[(-3, 3)] * 784,
        subspace_dim=50,
        n_projections=3,
        n_anchors=2,
        clip_penalty=0.1,
    )
    result = oracle.run(n_trials=500, seed=42)
    print(result.max_error, result.worst_input.shape)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class _ClipPenaltyFitness:
    """Fitness used by the inner CMA-ES on each (projection, anchor).

    Maps subspace coordinate ``z`` to the clipped input ``x``, evaluates
    true divergence at ``x``, and subtracts a clip-distance penalty so
    CMA-ES is steered toward in-bounds inputs without being hard-
    blocked from out-of-bounds exploration. Tracks the best in-bounds
    divergence seen so far in a shared ``tracker`` dict.
    """

    def __init__(
        self,
        plain_fn: Callable,
        fhe_fn: Callable,
        anchor: np.ndarray,
        R: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        clip_penalty: float,
        tracker: dict,
    ) -> None:
        self._plain = plain_fn
        self._fhe = fhe_fn
        self._anchor = anchor
        self._R = R
        self._lo = lo
        self._hi = hi
        self._k = float(clip_penalty)
        self._tracker = tracker

    def score(self, z: Any) -> float:
        z_arr = np.asarray(z, dtype=np.float64).ravel()
        x_raw = self._anchor + self._R @ z_arr
        x_clipped = np.clip(x_raw, self._lo, self._hi)
        clip_dist = float(np.linalg.norm(x_raw - x_clipped))
        try:
            p = self._plain(x_clipped)
            f = self._fhe(x_clipped)
        except Exception:
            return 0.0
        if np.isscalar(p) and np.isscalar(f):
            div = abs(float(p) - float(f))
        else:
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            div = float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
        if div > self._tracker["best_error"]:
            self._tracker["best_error"] = div
            self._tracker["best_x"] = x_clipped.copy()
        return div - self._k * clip_dist


class SubspaceOracle:
    """Random-subspace CMA-ES for high-dimensional inputs.

    Parameters
    ----------
    plaintext_fn, fhe_fn : callable
        Plaintext reference and FHE function under test. Both accept an
        array-like of length ``d``.
    bounds : list of (lo, hi)
        Per-dimension input box (length ``d``).
    subspace_dim : int, optional
        Subspace dimension ``k``. Defaults to ``min(50, d)``.
    n_projections : int, default 3
        Number of random subspaces to try.
    n_anchors : int, default 2
        Anchors per projection. The first is always the box midpoint;
        additional anchors are corner-biased random points.
    anchor : np.ndarray, optional
        Override the default midpoint anchor. When set, ``n_anchors``
        falls back to 1 (this anchor only).
    clip_penalty : float, default 0.1
        Coefficient on the clip-distance penalty. Larger values keep
        CMA-ES strictly in-bounds; smaller values allow exploring
        out-of-bounds and clipping back.
    probe_size : int, default 50
        Random probes drawn before the subspace search to estimate the
        baseline. Used to decide the random-fallback trigger.
    fallback_threshold : float, default 1.10
        After each (projection, anchor), if the best divergence is less
        than ``fallback_threshold * probe_max``, the search abandons
        further subspace exploration and burns the remaining budget on
        uniform random sampling.
    **oracle_kwargs
        Forwarded to the inner :class:`FHEOracle` (e.g. ``sigma0``,
        ``separable``, ``use_heuristic_seeds``).
    """

    def __init__(
        self,
        plaintext_fn: Callable,
        fhe_fn: Callable,
        bounds: list,
        subspace_dim: Optional[int] = None,
        n_projections: int = 3,
        n_anchors: int = 2,
        anchor: Optional[np.ndarray] = None,
        clip_penalty: float = 0.1,
        probe_size: int = 50,
        fallback_threshold: float = 1.10,
        **oracle_kwargs: Any,
    ) -> None:
        self.plaintext_fn = plaintext_fn
        self.fhe_fn = fhe_fn
        self.bounds = list(bounds)
        self.d = len(bounds)
        self.k = int(subspace_dim) if subspace_dim is not None else min(50, self.d)
        if self.k <= 0 or self.k > self.d:
            raise ValueError(
                f"subspace_dim must be in [1, d={self.d}]; got {self.k}"
            )
        if n_projections < 1:
            raise ValueError("n_projections must be >= 1")
        if n_anchors < 1:
            raise ValueError("n_anchors must be >= 1")
        self.n_projections = int(n_projections)
        self.n_anchors = int(n_anchors)
        self.clip_penalty = float(clip_penalty)
        self.probe_size = int(probe_size)
        self.fallback_threshold = float(fallback_threshold)
        self.lo = np.array([bd[0] for bd in bounds], dtype=np.float64)
        self.hi = np.array([bd[1] for bd in bounds], dtype=np.float64)
        if anchor is not None:
            self._user_anchor: Optional[np.ndarray] = (
                np.asarray(anchor, dtype=np.float64).ravel()
            )
            if self._user_anchor.size != self.d:
                raise ValueError(
                    f"anchor must have length d={self.d}, got {self._user_anchor.size}"
                )
            self.n_anchors = 1
        else:
            self._user_anchor = None
        # Backward-compat: legacy `anchor` attribute referencing the
        # default midpoint, used by older tests/callers.
        self.anchor = (
            self._user_anchor.copy()
            if self._user_anchor is not None
            else 0.5 * (self.lo + self.hi)
        )
        self._anchor_seed = 0xA0DC  # deterministic salt for anchor RNG
        # Stored across run() so tests can introspect the last probe.
        self.probe_max: float = 0.0
        self.oracle_kwargs = oracle_kwargs

    # --- geometry helpers ---------------------------------------------

    def _make_projection(self, rng: np.random.Generator) -> np.ndarray:
        """Random projection with orthonormal columns (d x k)."""
        A = rng.standard_normal(size=(self.d, self.k))
        Q, _ = np.linalg.qr(A)
        return Q[:, : self.k]

    def _z_to_x(self, z: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Map subspace coordinate z to clipped input x using the
        default (midpoint or user-supplied) anchor."""
        x = self.anchor + R @ z
        return np.clip(x, self.lo, self.hi)

    def _z_bounds_ball(
        self, R: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric ball-radius bounds per subspace axis.

        Each axis ``j`` gets ``[-r_j, r_j]`` where ``r_j`` is the median
        per-dimension reach ``half_width_i / |R[i, j]|``. The median
        keeps roughly half of projected inputs inside the box (the
        rest get a clip penalty), so CMA-ES has room to explore.

        Parameters
        ----------
        R : np.ndarray, optional
            Projection matrix (d x k). Required unless ``scale`` is
            supplied directly.
        scale : np.ndarray, optional
            Per-axis radius override (length k).
        """
        if scale is None:
            if R is None:
                raise ValueError("either R or scale must be supplied")
            half_widths = (self.hi - self.lo) / 2.0
            scales = np.empty(self.k, dtype=np.float64)
            for j in range(self.k):
                r_j = R[:, j]
                reaches = np.where(
                    np.abs(r_j) > 1e-12,
                    half_widths / np.where(np.abs(r_j) > 1e-12, np.abs(r_j), 1.0),
                    np.inf,
                )
                finite = reaches[np.isfinite(reaches)]
                scales[j] = float(np.median(finite)) if finite.size else 10.0
            scale = scales
        scale = np.asarray(scale, dtype=np.float64).ravel()
        if scale.size != self.k:
            raise ValueError(f"scale length must be k={self.k}, got {scale.size}")
        # Guard: positive width.
        scale = np.where(scale > 1e-9, scale, 10.0)
        return -scale, scale

    def _z_bounds_intersection(
        self, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy intersection-of-constraints z-bounds.

        Retained for benchmarking and the regression test that compares
        the new ball bounds against the (sliver-prone) old bounds.
        """
        z_lo = np.full(self.k, -np.inf)
        z_hi = np.full(self.k, np.inf)
        for j in range(self.k):
            r_j = R[:, j]
            for i in range(self.d):
                rij = r_j[i]
                if abs(rij) < 1e-12:
                    continue
                t_a = (self.lo[i] - self.anchor[i]) / rij
                t_b = (self.hi[i] - self.anchor[i]) / rij
                t_lo, t_hi = (t_a, t_b) if rij > 0 else (t_b, t_a)
                if t_lo > z_lo[j]:
                    z_lo[j] = t_lo
                if t_hi < z_hi[j]:
                    z_hi[j] = t_hi
        z_lo = np.where(np.isfinite(z_lo), z_lo, -10.0)
        z_hi = np.where(np.isfinite(z_hi), z_hi, 10.0)
        eps = 1e-6
        too_tight = (z_hi - z_lo) < eps
        z_lo = np.where(too_tight, z_lo - eps, z_lo)
        z_hi = np.where(too_tight, z_hi + eps, z_hi)
        return z_lo, z_hi

    def _z_bounds(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Default z-bounds: ball-radius (v0.3.1)."""
        return self._z_bounds_ball(R=R)

    def _generate_anchors(self, n_anchors: Optional[int] = None) -> list[np.ndarray]:
        """Anchor diversification.

        Returns the box midpoint plus ``n_anchors - 1`` corner anchors.
        Corner anchors place every coordinate on the box boundary
        (lo or hi, drawn uniformly) so ``||anchor||`` is maximised.
        CKKS divergence amplitudes concentrate at large operand
        magnitudes, so a true-corner anchor lets the subspace search
        explore the high-magnitude region from the start.
        """
        if self._user_anchor is not None:
            return [self._user_anchor.copy()]
        if n_anchors is None:
            n_anchors = self.n_anchors
        midpoint = 0.5 * (self.lo + self.hi)
        anchors = [midpoint]
        rng = np.random.RandomState(self._anchor_seed)
        for _ in range(max(0, n_anchors - 1)):
            mask = rng.randint(0, 2, size=self.d).astype(bool)
            a = np.where(mask, self.hi, self.lo).astype(np.float64)
            anchors.append(a)
        return anchors

    # --- run helpers --------------------------------------------------

    def _measure_divergence(self, x: np.ndarray) -> float:
        """True |plain(x) - fhe(x)|, reducer-max."""
        try:
            p = self.plaintext_fn(x)
            f = self.fhe_fn(x)
            if np.isscalar(p) and np.isscalar(f):
                return float(abs(p - f))
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            if n == 0:
                return 0.0
            return float(np.max(np.abs(p_arr[:n] - f_arr[:n])))
        except Exception:
            return 0.0

    def _random_probe(
        self, n: int, rng: np.random.Generator
    ) -> tuple[float, np.ndarray]:
        """Draw n uniform-random inputs, return (max_div, best_x)."""
        best = 0.0
        best_x = 0.5 * (self.lo + self.hi)
        for _ in range(int(n)):
            x = rng.uniform(self.lo, self.hi)
            d = self._measure_divergence(x)
            if d > best:
                best = d
                best_x = x.copy()
        return best, best_x

    # --- run ---------------------------------------------------------

    def run(self, n_trials: int = 500, seed: int = 42, threshold: float = 1e-2):
        """Search across (projection, anchor) pairs, return best result.

        Parameters
        ----------
        n_trials : int, default 500
            Total budget. The probe phase claims up to ``probe_size``
            evaluations; the remainder is split across the
            ``n_projections * n_anchors`` sub-searches. If a sub-search
            consistently fails to beat the probe baseline, the leftover
            budget is spent on uniform random sampling.
        seed : int, default 42
            Seed for the projection/anchor RNGs and inner CMA-ES seeds.
        threshold : float, default 1e-2
            PASS/FAIL cut-off copied onto the returned result.
        """
        from .core import FHEOracle, OracleResult

        rng = np.random.default_rng(int(seed))

        # --- probe phase ------------------------------------------------
        probe_n = min(self.probe_size, max(1, n_trials // 4))
        probe_max, probe_x = self._random_probe(probe_n, rng)
        self.probe_max = float(probe_max)
        budget_remaining = n_trials - probe_n

        # Initialise best with the probe winner so a degenerate sub-
        # search never returns a worse result than uniform random.
        best_error = float(probe_max)
        best_x = probe_x.copy()
        best_proj_idx = -1
        best_result: Optional[OracleResult] = None

        anchors = self._generate_anchors()
        n_subsearches = self.n_projections * len(anchors)
        budget_per = max(2, budget_remaining // max(1, n_subsearches)) if n_subsearches else 0

        evals_used_in_subsearches = 0
        fallback_taken = False

        for proj_idx in range(self.n_projections):
            R = self._make_projection(rng)
            z_lo, z_hi = self._z_bounds_ball(R=R)
            z_bounds_list = list(zip(z_lo.tolist(), z_hi.tolist()))

            for anchor_idx, anchor in enumerate(anchors):
                if budget_remaining <= 0:
                    break
                this_budget = min(budget_per, budget_remaining)
                tracker = {"best_error": -np.inf, "best_x": None}
                fitness = _ClipPenaltyFitness(
                    plain_fn=self.plaintext_fn,
                    fhe_fn=self.fhe_fn,
                    anchor=anchor,
                    R=R,
                    lo=self.lo,
                    hi=self.hi,
                    clip_penalty=self.clip_penalty,
                    tracker=tracker,
                )
                inner_seed = (
                    seed + proj_idx * len(anchors) + anchor_idx
                )
                oracle = FHEOracle(
                    plaintext_fn=lambda z: 0.0,
                    fitness=fitness,
                    input_dim=self.k,
                    input_bounds=z_bounds_list,
                    seed=inner_seed,
                    **self.oracle_kwargs,
                )
                result = oracle.run(n_trials=this_budget, threshold=threshold)
                budget_remaining -= this_budget
                evals_used_in_subsearches += this_budget

                # Record the best in-bounds divergence the fitness
                # witnessed (clip penalty is excluded from this number).
                if (
                    tracker["best_x"] is not None
                    and tracker["best_error"] > best_error
                ):
                    best_error = float(tracker["best_error"])
                    best_x = np.asarray(tracker["best_x"], dtype=np.float64)
                    best_proj_idx = proj_idx
                    best_result = result

                # Fallback trigger (spec: paragraph 1c). Once we have
                # invested >= 30% of the original budget across sub-
                # searches and still trail the probe by less than
                # ``fallback_threshold``, abandon the rest and burn the
                # remainder on uniform random sampling. 30% gives the
                # search ~2 (projection, anchor) attempts to find a
                # productive subspace before bailing -- enough to land
                # a corner anchor on hot-zone bugs but cheap to abort
                # on dense directional bugs where subspace is
                # fundamentally outmatched by uniform random.
                if (
                    evals_used_in_subsearches > n_trials * 0.3
                    and best_error < self.fallback_threshold * self.probe_max
                ):
                    fallback_taken = True
                    break
            if fallback_taken or budget_remaining <= 0:
                break

        # Fallback / leftover budget -> uniform random.
        if budget_remaining > 0:
            extra_max, extra_x = self._random_probe(budget_remaining, rng)
            if extra_max > best_error:
                best_error = float(extra_max)
                best_x = extra_x.copy()
            budget_remaining = 0

        # Build the result. If no inner CMA-ES result is available
        # (degenerate budgets), synthesise a minimal one.
        if best_result is None:
            best_result = OracleResult(
                verdict="FAIL" if best_error >= threshold else "PASS",
                max_error=float(best_error),
                worst_input=best_x.tolist(),
                threshold=float(threshold),
                n_trials=int(n_trials),
                elapsed_seconds=0.0,
                scheme="plaintext-diff",
            )
        else:
            best_result.worst_input = best_x.tolist()
            best_result.max_error = float(best_error)
            best_result.verdict = "FAIL" if best_error >= threshold else "PASS"

        best_result.strategy_used = "subspace"
        best_result.subspace_dim = self.k
        best_result.n_projections = self.n_projections
        best_result.n_anchors = self.n_anchors
        best_result.projection_index = best_proj_idx if best_proj_idx >= 0 else 0
        best_result.probe_max = float(self.probe_max)
        best_result.fallback_taken = bool(fallback_taken)
        return best_result
