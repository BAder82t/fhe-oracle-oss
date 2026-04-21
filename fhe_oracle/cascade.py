# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Multi-fidelity (cascade) adversarial search.

Stage 1: search with a CHEAP fidelity (e.g. Taylor-3 sigmoid, 3 CKKS
levels, fast per-eval) for B_cheap evaluations. Track the K candidates
with the largest cheap-fidelity divergence.

Stage 2: re-evaluate those K candidates under the EXPENSIVE fidelity
(e.g. Cheb-15 sigmoid, 15 CKKS levels, slow per-eval) and return the
best.

Premise: low-degree and high-degree polynomial approximations to the
same nonlinearity (sigmoid) share peak-error structure — their
worst-case z values are correlated. If the rank correlation between
cheap and expensive divergence rankings is high, the cascade gets the
expensive fidelity's accuracy at a small fraction of its compute cost.

Public API
----------
    from fhe_oracle.cascade import CascadeSearch, evaluate_correlation

    cs = CascadeSearch(
        cheap_fhe_fn=taylor3_fn,
        expensive_fhe_fn=cheb15_fn,
        plaintext_fn=sigmoid_plain,
        input_bounds=bounds,
        top_k=20,
    )
    out = cs.run(budget_cheap=500, search_kind="cma", seeds=range(1, 11))

The cascade composes with :class:`~fhe_oracle.preactivation.PreactivationOracle`:
pass ``search_kind="preactivation"`` and the cheap/expensive functions
both go through preactivation projection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .core import FHEOracle
from .preactivation import PreactivationOracle


@dataclass
class CascadeResult:
    """Cascade-search outcome for one seed."""

    seed: int
    max_error_cheap_at_winner: float    # cheap divergence at the chosen x
    max_error_expensive: float          # expensive divergence at the chosen x
    x: list[float]
    n_evals_cheap: int
    n_evals_expensive: int
    elapsed_seconds: float
    extra: dict = field(default_factory=dict)


def evaluate_correlation(
    cheap_fhe_fn: Callable,
    expensive_fhe_fn: Callable,
    plaintext_fn: Callable,
    samples: list[Any],
) -> dict[str, float]:
    """Spearman rank correlation between cheap and expensive divergences.

    Evaluates ``samples`` under both fidelities and reports the
    Spearman coefficient. Used to decide whether the cascade premise
    holds before paying for a real run.

    Returns dict with keys: spearman, pearson, n_samples,
    cheap_mean, expensive_mean.
    """
    cheap = []
    expensive = []
    for x in samples:
        try:
            p = float(_to_scalar(plaintext_fn(x)))
            c = float(_to_scalar(cheap_fhe_fn(x)))
            e = float(_to_scalar(expensive_fhe_fn(x)))
        except Exception:
            continue
        cheap.append(abs(p - c))
        expensive.append(abs(p - e))

    if len(cheap) < 3:
        return {
            "spearman": 0.0, "pearson": 0.0, "n_samples": len(cheap),
            "cheap_mean": 0.0, "expensive_mean": 0.0,
        }

    arr_c = np.array(cheap)
    arr_e = np.array(expensive)
    # Spearman = Pearson on ranks
    rk_c = np.argsort(np.argsort(arr_c)).astype(float)
    rk_e = np.argsort(np.argsort(arr_e)).astype(float)
    if np.std(rk_c) == 0 or np.std(rk_e) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(rk_c, rk_e)[0, 1])
    if np.std(arr_c) == 0 or np.std(arr_e) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(arr_c, arr_e)[0, 1])
    return {
        "spearman": spearman,
        "pearson": pearson,
        "n_samples": len(cheap),
        "cheap_mean": float(arr_c.mean()),
        "expensive_mean": float(arr_e.mean()),
    }


class CascadeSearch:
    """Two-stage cheap-then-expensive adversarial search.

    Parameters
    ----------
    cheap_fhe_fn, expensive_fhe_fn : callables
        Two FHE-evaluation functions of the same plaintext function,
        differing in fidelity (e.g. Taylor-3 vs Cheb-15 sigmoid).
    plaintext_fn : callable
        Reference plaintext implementation.
    input_bounds : list of (low, high)
        Per-dim box constraints on x (raw input space when
        ``search_kind="cma"`` or ``search_kind="random"``; passed to
        the inner :class:`PreactivationOracle` when
        ``search_kind="preactivation"``).
    top_k : int
        Number of cheap-stage candidates re-evaluated under the
        expensive fidelity. Default 20.
    weights : (W, b) tuple, optional
        Required when ``search_kind="preactivation"``.
    """

    def __init__(
        self,
        cheap_fhe_fn: Callable,
        expensive_fhe_fn: Callable,
        plaintext_fn: Callable,
        input_bounds: list[tuple[float, float]],
        top_k: int = 20,
        weights: Optional[tuple[np.ndarray, np.ndarray]] = None,
        clip_penalty: float = 0.05,
    ) -> None:
        self._cheap = cheap_fhe_fn
        self._expensive = expensive_fhe_fn
        self._plain = plaintext_fn
        self._bounds = list(input_bounds)
        self._d = len(self._bounds)
        self._top_k = int(top_k)
        self._weights = weights
        self._clip_penalty = float(clip_penalty)

    # --------- Helpers ----------------------------------------------

    def _cheap_div(self, x) -> float:
        try:
            return abs(
                float(_to_scalar(self._plain(x)))
                - float(_to_scalar(self._cheap(x)))
            )
        except Exception:
            return 0.0

    def _expensive_div(self, x) -> float:
        try:
            return abs(
                float(_to_scalar(self._plain(x)))
                - float(_to_scalar(self._expensive(x)))
            )
        except Exception:
            return 0.0

    # --------- Cheap-stage search drivers ---------------------------

    def _cheap_random(self, budget: int, seed: int) -> list[tuple[float, np.ndarray]]:
        rng = np.random.default_rng(seed)
        lows = np.array([lo for lo, _ in self._bounds])
        highs = np.array([hi for _, hi in self._bounds])
        scored: list[tuple[float, np.ndarray]] = []
        for _ in range(budget):
            x = rng.uniform(lows, highs)
            s = self._cheap_div(x)
            scored.append((s, x.copy()))
        return scored

    def _cheap_cma(self, budget: int, seed: int) -> list[tuple[float, np.ndarray]]:
        scored: list[tuple[float, np.ndarray]] = []

        class _RecorderFitness:
            def __init__(self, parent: "CascadeSearch"):
                self._parent = parent

            def score(self, x):
                s = self._parent._cheap_div(x)
                scored.append((s, np.asarray(x, dtype=np.float64).copy()))
                return s

        oracle = FHEOracle(
            plaintext_fn=lambda _x: 0.0,
            fhe_fn=lambda _x: 0.0,
            input_dim=self._d,
            input_bounds=self._bounds,
            fitness=_RecorderFitness(self),
            seed=int(seed),
            random_floor=0.2,
            warm_start=True,
        )
        oracle.run(n_trials=int(budget), threshold=0.0)
        return scored

    def _cheap_preactivation(
        self, budget: int, seed: int
    ) -> list[tuple[float, np.ndarray]]:
        if self._weights is None:
            raise ValueError(
                "search_kind='preactivation' requires weights=(W,b)"
            )
        W, b = self._weights
        scored: list[tuple[float, np.ndarray]] = []

        def _wrapped_cheap(x):
            v = self._cheap(x)
            scored.append((self._cheap_div(x), np.asarray(x).copy()))
            return v

        pre = PreactivationOracle(
            W=W, b=b,
            plaintext_fn=self._plain,
            fhe_fn=_wrapped_cheap,
            input_bounds=self._bounds,
            clip_penalty=self._clip_penalty,
        )
        pre.run(budget=budget, seeds=[int(seed)])
        return scored

    # --------- Top-level run ----------------------------------------

    def run(
        self,
        budget_cheap: int = 500,
        seeds: Iterable[int] = range(1, 11),
        search_kind: str = "cma",
    ) -> list[CascadeResult]:
        """Execute cascade search across seeds.

        Parameters
        ----------
        budget_cheap : int
            Cheap-fidelity evaluation budget.
        seeds : iterable of int
        search_kind : {"cma", "preactivation", "random"}
        """
        results: list[CascadeResult] = []
        for seed in seeds:
            t0 = time.perf_counter()
            if search_kind == "cma":
                scored = self._cheap_cma(budget_cheap, seed)
            elif search_kind == "preactivation":
                scored = self._cheap_preactivation(budget_cheap, seed)
            elif search_kind == "random":
                scored = self._cheap_random(budget_cheap, seed)
            else:
                raise ValueError(f"unknown search_kind={search_kind!r}")

            # Sort by cheap divergence descending, dedupe roughly.
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: self._top_k]

            # Stage 2: expensive re-eval, pick the winner.
            best_exp = -np.inf
            best_x = None
            best_cheap_at_winner = 0.0
            for cheap_s, x in top:
                e = self._expensive_div(x)
                if e > best_exp:
                    best_exp = e
                    best_x = x.copy()
                    best_cheap_at_winner = cheap_s

            results.append(CascadeResult(
                seed=int(seed),
                max_error_cheap_at_winner=float(best_cheap_at_winner),
                max_error_expensive=float(max(0.0, best_exp)),
                x=(best_x.tolist() if best_x is not None else []),
                n_evals_cheap=len(scored),
                n_evals_expensive=len(top),
                elapsed_seconds=time.perf_counter() - t0,
            ))
        return results


# --- helpers --------------------------------------------------------


def _to_scalar(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    arr = np.asarray(value).ravel()
    return float(arr[0]) if arr.size > 0 else 0.0
