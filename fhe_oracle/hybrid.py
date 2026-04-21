# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Hybrid oracle + empirical search with union verdict (A3).

`run_hybrid` runs `FHEOracle.run()` (CMA-ES adversarial search over the
input box) and `EmpiricalSearch.run()` (sampling from a training
distribution) on the same circuit and combines them with a union
verdict: PASS iff BOTH pass; FAIL if either fails.

This is the paper's §6.7 methodological recommendation — neither
method dominates the other:
- Oracle finds inputs in the box corners that empirical never sees.
- Empirical finds class-separating-direction concentrations that
  uniform-box CMA-ES misses.

Union of two FAIL-sound legs is FAIL-sound. PASS is strictly stronger
than either leg alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .core import FHEOracle, OracleResult
from .empirical import EmpiricalResult, EmpiricalSearch


@dataclass(frozen=True)
class HybridResult:
    """Combined result of oracle + empirical search."""

    oracle_result: OracleResult
    empirical_result: Optional[EmpiricalResult]
    union_verdict: str          # "PASS" iff both legs PASS; "FAIL" if either FAILs
    max_error: float            # max across legs
    worst_input: Any            # input (list or ndarray) from the leg with larger max_error
    source: str                 # "oracle" | "empirical"


def _default_divergence_fn(
    plaintext_fn: Callable, fhe_fn: Callable
) -> Callable[[np.ndarray], float]:
    """Construct δ(x) = |plaintext_fn(x) − fhe_fn(x)| over scalar/vector outputs."""
    def _div(x: np.ndarray) -> float:
        try:
            p = plaintext_fn(x.tolist())
            f = fhe_fn(x.tolist())
        except Exception:
            return 0.0
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
        f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
        n = min(p_arr.size, f_arr.size)
        if n == 0:
            return 0.0
        return float(np.max(np.abs(p_arr[:n] - f_arr[:n])))
    return _div


def run_hybrid(
    *,
    plaintext_fn: Callable,
    fhe_fn: Callable,
    input_dim: int,
    input_bounds: Optional[list[tuple[float, float]] | tuple[float, float]] = None,
    threshold: float = 1e-3,
    # Oracle kwargs
    oracle_budget: int = 500,
    oracle_seed: int = 42,
    random_floor: float = 0.3,
    warm_start: bool = True,
    warm_sigma_scale: float = 0.2,
    restarts: int = 0,
    bipop: bool = False,
    # Empirical kwargs
    data: Optional[np.ndarray] = None,
    empirical_budget: int = 500,
    jitter_std: float = 0.1,
    empirical_seed: int = 43,
    divergence_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> HybridResult:
    """Run oracle + empirical search and combine with union verdict.

    When `data is None`, the empirical leg is skipped and the hybrid
    collapses to the oracle result (union_verdict = oracle verdict,
    empirical_result = None, source = "oracle").
    """
    oracle = FHEOracle(
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
        input_dim=input_dim,
        input_bounds=input_bounds,
        seed=oracle_seed,
        random_floor=random_floor,
        warm_start=warm_start,
        warm_sigma_scale=warm_sigma_scale,
        restarts=restarts,
        bipop=bipop,
    )
    oracle_result = oracle.run(n_trials=oracle_budget, threshold=threshold)

    empirical_result: Optional[EmpiricalResult] = None
    if data is not None:
        div_fn = divergence_fn or _default_divergence_fn(plaintext_fn, fhe_fn)
        emp = EmpiricalSearch(
            divergence_fn=div_fn,
            data=data,
            threshold=threshold,
            budget=empirical_budget,
            jitter_std=jitter_std,
            seed=empirical_seed,
        )
        empirical_result = emp.run()

    if empirical_result is None:
        return HybridResult(
            oracle_result=oracle_result,
            empirical_result=None,
            union_verdict=oracle_result.verdict,
            max_error=oracle_result.max_error,
            worst_input=oracle_result.worst_input,
            source="oracle",
        )

    # Union: FAIL if either FAILs; PASS only if both PASS.
    if oracle_result.verdict == "FAIL" or empirical_result.verdict == "FAIL":
        union_verdict = "FAIL"
    else:
        union_verdict = "PASS"

    if oracle_result.max_error >= empirical_result.max_error:
        max_err = oracle_result.max_error
        worst = oracle_result.worst_input
        source = "oracle"
    else:
        max_err = empirical_result.max_error
        worst = empirical_result.worst_input
        source = "empirical"

    return HybridResult(
        oracle_result=oracle_result,
        empirical_result=empirical_result,
        union_verdict=union_verdict,
        max_error=max_err,
        worst_input=worst,
        source=source,
    )
