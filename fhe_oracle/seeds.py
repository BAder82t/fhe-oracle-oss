# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-patented fallback seed generator for CMA-ES gen-0 injection.

Used by :class:`FHEOracle` when ``use_heuristic_seeds=True`` is set
but the patented heuristic seed generators (Multiplication Magnifier,
Depth Seeker, Near-Threshold Explorer -- PCT/IB2026/053378 Claim 5)
are not registered in the plugin registry (i.e.\\ ``fhe-oracle-pro``
is not installed).

The fallback mixes ``k // 2`` axis-aligned corner candidates with
``k // 2`` uniform random samples. Corners are where polynomial
approximations (Taylor, Chebyshev, minimax) diverge most; uniform
randoms provide coverage inside the box. No claim-5 scoring is
performed. This is intentionally weaker than the patented heuristics
but gives Core a reasonable seeded initialisation on narrow-corridor
circuits (e.g.\\ ckks_d5) where the CMA-ES default would otherwise
basin-trap at the box midpoint.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fallback_corner_seeds(
    rng: np.random.Generator,
    bounds: Sequence[tuple[float, float]],
    k: int = 10,
) -> list[list[float]]:
    """Return ``k`` seed inputs blending box corners and uniform random.

    Parameters
    ----------
    rng
        RNG stream for the random half of the seeds. Should be seeded
        independently from the CMA-ES internal RNG.
    bounds
        Per-dimension ``(low, high)`` pairs.
    k
        Total seeds to return. If ``k <= 0`` or ``len(bounds) == 0``,
        returns an empty list.

    Returns
    -------
    list[list[float]]
        ``min(k, 2**d + pool)`` seeds. The first ``k // 2`` are
        randomly chosen corners (without replacement if ``2**d >= k/2``,
        otherwise with repetition); the remaining are uniform samples.
    """
    if k <= 0:
        return []
    d = len(bounds)
    if d == 0:
        return []

    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)

    n_corner = k // 2
    n_random = k - n_corner

    corners: list[np.ndarray] = []
    if n_corner > 0:
        # Axis-aligned corners. At d > ~20 there are 2**d corners so
        # random subsetting is fine; at small d we may repeat, which
        # is harmless (CMA-ES dedupes).
        max_enum = 1 << min(d, 12)
        for _ in range(n_corner):
            mask = int(rng.integers(0, max_enum))
            corner = np.where(
                ((np.arange(d) < 12) & ((mask >> np.arange(d)) & 1 == 1)),
                highs,
                lows,
            )
            corners.append(corner)

    random_samples: list[np.ndarray] = []
    if n_random > 0:
        random_samples = [
            rng.uniform(lows, highs) for _ in range(n_random)
        ]

    return [s.tolist() for s in corners + random_samples]
