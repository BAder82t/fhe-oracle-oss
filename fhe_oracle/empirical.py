# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Empirical-distribution search for FHE divergence testing (A3).

Draws inputs by sampling (with replacement + Gaussian jitter) from a
user-supplied empirical dataset and measures divergence `δ(x)`. Used
alongside the CMA-ES oracle to catch failures that concentrate along
class-separating directions in the training distribution — the paper's
§6.7 WDBC failure mode where uniform-box CMA-ES loses 9/10 to
empirical random sampling.

Core library — no scikit-learn dependency. The caller supplies
`divergence_fn` (constructed from their plaintext + FHE callables)
and `data` (a numpy array of samples, typically standardised training
features).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class EmpiricalResult:
    """Outcome of an empirical-distribution search."""

    max_error: float
    worst_input: np.ndarray
    n_trials: int
    elapsed_s: float
    verdict: str               # "PASS" if max_error < threshold, else "FAIL"
    threshold: float
    hits: int                  # samples with δ ≥ threshold
    mu_hat: float              # hits / n_trials (empirical failure fraction)


class EmpiricalSearch:
    """Search for FHE divergence by sampling from an empirical distribution.

    Parameters
    ----------
    divergence_fn : callable
        `divergence_fn(x: np.ndarray) -> float` returning δ(x) at x.
    data : np.ndarray, shape (n_samples, d)
        Empirical dataset — e.g., standardised training features.
    threshold : float
        τ cutoff for FAIL verdict and hit counting. Default 1e-3.
    budget : int
        Number of evaluations (samples drawn with replacement + jitter).
        Default 500.
    jitter_std : float
        Gaussian jitter standard deviation added per feature. Default
        0.1 (per paper §6.7).
    seed : int
        RNG seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        divergence_fn: Callable[[np.ndarray], float],
        data: np.ndarray,
        threshold: float = 1e-3,
        budget: int = 500,
        jitter_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if jitter_std < 0:
            raise ValueError("jitter_std must be non-negative")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.ndim != 2:
            raise ValueError("data must be a 2-D array of shape (n, d)")
        if data_arr.shape[0] == 0:
            raise ValueError("data must contain at least one sample")

        self.divergence_fn = divergence_fn
        self.data = data_arr
        self.threshold = float(threshold)
        self.budget = int(budget)
        self.jitter_std = float(jitter_std)
        self.seed = int(seed)

    def run(self) -> EmpiricalResult:
        rng = np.random.default_rng(self.seed)
        best_err = -np.inf
        best_x: Optional[np.ndarray] = None
        hits = 0
        n, d = self.data.shape

        t0 = time.perf_counter()
        for _ in range(self.budget):
            idx = int(rng.integers(0, n))
            x = self.data[idx].copy()
            if self.jitter_std > 0:
                x = x + rng.normal(0.0, self.jitter_std, size=d)
            try:
                err = float(self.divergence_fn(x))
            except Exception:
                err = 0.0
            if err > best_err:
                best_err = err
                best_x = x.copy()
            if err >= self.threshold:
                hits += 1
        elapsed = time.perf_counter() - t0

        if best_x is None:
            best_x = np.zeros(d)
            best_err = 0.0

        verdict = "FAIL" if best_err >= self.threshold else "PASS"
        return EmpiricalResult(
            max_error=float(best_err),
            worst_input=best_x,
            n_trials=self.budget,
            elapsed_s=elapsed,
            verdict=verdict,
            threshold=self.threshold,
            hits=hits,
            mu_hat=hits / self.budget if self.budget > 0 else 0.0,
        )
