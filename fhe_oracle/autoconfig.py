# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto-configuration probe for FHE Oracle.

Runs a cheap initial probe (default 50 uniform random evaluations) to
classify the divergence landscape into one of four regimes, then
dispatches to the strategy best suited for that regime:

1. ``FULL_DOMAIN_SATURATION`` -- divergence is high everywhere.
   Dispatch: pure random search (``random_floor=1.0``). CMA-ES has no
   concentrated region to exploit.
   Detection: >90% of probes exceed 0.5 * max(probes).

2. ``PLATEAU_THEN_CLIFF`` -- divergence is flat with rare spikes.
   Dispatch: warm-start search (``random_floor=0.3``, ``warm_start=True``).
   Detection: any of three complementary tests in
   :func:`_detect_plateau_cliff` -- relaxed CV (CV < 0.3 with
   max > 5*median), rank-based cliff (top-decile minimum > 5x median
   of bottom 90%), or gap test (p90/median < 2 AND max > 5*p90).
   Borderline cases (CV in (0.1, 0.5)) trigger a second probe pass.

3. ``PREACTIVATION_DOMINATED`` -- divergence correlates with |Wx+b|.
   Dispatch: :class:`PreactivationOracle` when ``W, b`` are supplied.
   Detection: Spearman(delta, |Wx+b|) > 0.7 across probes.

4. ``STANDARD`` -- divergence has structure but no extreme regime.
   Dispatch: default CMA-ES (pure divergence, no warm-start).
   Detection: none of the above.

In plain ``fhe_fn`` divergence mode (not preactivation) a boundary probe snaps
the best probes to vertices and, if a vertex beats every probe, climbs by
single-coordinate flips. A better pre-search witness is re-measured like core's.

Example
-------
    from fhe_oracle.autoconfig import AutoOracle

    oracle = AutoOracle(
        plaintext_fn=f, fhe_fn=f_tilde,
        bounds=[(-3, 3)] * d,
        W=W, b=b,                 # optional, enables preactivation
    )
    result = oracle.run(n_trials=500, seed=42)
    print(result.regime)          # 'standard', 'preactivation_dominated', ...
    print(result.strategy_used)   # 'cma_es', 'preactivation', ...

Every evaluation counts toward the ``n_trials`` budget: probes, structure diagnostic,
boundary probe, search and re-measurements (``adaptive=True`` may extend it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from scipy.stats import spearmanr

from .diagnostics import characterize_structure
from .evallog import EventSink, _event
from .fitness import DivergenceFitness

if TYPE_CHECKING:
    from .core import FHEOracle, OracleResult
    from .preactivation import PreactivationOracle

_Boundary = tuple[Optional[list[float]], float, int]  # (vertex, divergence, evaluations used)


_PROBE_SEED_SALT = 0xB0B3  # deterministic seed salt for probe RNG
_BOUNDARY_TOP_K = 3  # probe points snapped to vertices
_BOUNDARY_SHARE = 0.15  # max share of n_trials for the boundary probe
_RESERVE = 3  # one search evaluation plus two re-measurements (search and pre-search witness)
_RESERVE_1D_PREACT = 10  # PreactivationOracle's 1-D path spends max(budget, 9) + 1
_OTHER_OBJECTIVES = ("adapter", "fitness", "multi_output")  # kwargs that change what is scored


def _detect_plateau_cliff(divs: np.ndarray) -> bool:
    """Detect plateau-then-cliff landscapes via three complementary tests.

    Replaces the original ``std < 0.01 * mean AND max > 10 * median`` rule,
    which was too strict to trigger on Chebyshev-style TenSEAL circuits
    (CV ~= 0.4 with a narrow cliff).

    Tests
    -----
    A. Tight-plateau CV test: ``CV < 0.3 AND max > 5 * median``. Catches
       plateaus that are mildly noisy with a dramatic cliff.

    B. Dominant-plateau test: at least 80% of probes fall within
       ``[0, 1.5 * median]`` AND the max exceeds ``2 * median``.
       Workhorse for narrow cliffs at small probe counts; robust to
       zero-IQR plateaus where every plateau sample is identical.

    C. Gap test: ``p90 / median < 2`` (plateau tight in bulk) AND
       ``max > 5 * p90`` (cliff far above the plateau).

    Each test independently demands that BOTH a plateau and a cliff are
    visible -- heavy-tail (e.g. ``|Wx+b|^3``), saturated, and degenerate
    landscapes are rejected.
    """
    arr = np.asarray(divs, dtype=np.float64).ravel()
    n = arr.size
    if n < 5:
        return False

    mean_d = float(np.mean(arr))
    std_d = float(np.std(arr))
    med_d = float(np.median(arr))
    max_d = float(np.max(arr))
    p90 = float(np.percentile(arr, 90))

    if mean_d <= 0.0 or med_d <= 0.0:
        return False

    cv = std_d / mean_d if mean_d > 0.0 else 0.0

    # Test A: tight CV with dramatic cliff.
    if cv < 0.3 and max_d > 5.0 * med_d:
        return True

    # Test B: dominant-plateau fraction. >=80% of probes cluster in
    # [0, 1.5 * median] (the plateau) and the max sits clearly above
    # (cliff). Robust to zero-IQR plateaus where every plateau sample
    # is identical. Heavy-tail landscapes (|Wx+b|^3, exponential) fail
    # the 80% containment criterion.
    plateau_band = 1.5 * med_d
    plateau_frac = float(np.mean(arr <= plateau_band))
    if plateau_frac >= 0.80 and max_d > 2.0 * med_d:
        return True

    # Test C: bulk plateau + far cliff via percentiles.
    if p90 > 0.0:
        plateau_tight = (p90 / med_d) < 2.0
        cliff_far = max_d > 5.0 * p90
        if plateau_tight and cliff_far:
            return True

    return False


class Regime(Enum):
    """Landscape regime detected by :func:`classify_landscape`."""

    FULL_DOMAIN_SATURATION = "full_domain_saturation"
    PLATEAU_THEN_CLIFF = "plateau_then_cliff"
    DISTANT_DEFECT = "distant_defect"
    PREACTIVATION_DOMINATED = "preactivation_dominated"
    LOW_RANK_STRUCTURE = "low_rank_structure"
    STANDARD = "standard"


_LOW_RANK_MIN_DIM = 16
_LOW_RANK_STRUCTURE_SAMPLES = 100
_LOW_RANK_RANK_FRACTION = 0.5


_DISTANT_DEFECT_CENTER_PROBES = 20
_DISTANT_DEFECT_SIGMA = 1.0
_DISTANT_DEFECT_RATIO = 0.1


def _distant_defect_probe(score: Callable[[np.ndarray], float],
                          bounds: list[tuple[float, float]], rng: np.random.Generator,
                          n: int = _DISTANT_DEFECT_CENTER_PROBES) -> np.ndarray:
    """Sample n points in a Gaussian ball around box centre (radius
    matching default CMA-ES sigma0=1.0) and return per-sample
    divergences. Used for DISTANT_DEFECT detection.
    """
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    centre = (lo + hi) / 2.0
    centre_probes = centre + rng.normal(0.0, _DISTANT_DEFECT_SIGMA, size=(n, d))
    centre_probes = np.clip(centre_probes, lo, hi)
    return np.array([score(centre_probes[i]) for i in range(n)], dtype=np.float64)


def _detect_distant_defect(centre_divs: np.ndarray,
                           full_divs: np.ndarray) -> bool:
    """Defect concentrated far from box centre -- origin-start CMA-ES trap.

    Fires when the largest divergence in a Gaussian ball of radius
    sigma0=1.0 around the box centre is materially smaller than the
    largest divergence across the full bounds. In that landscape, a
    CMA-ES run starting at the midpoint with the default sigma0=1.0
    samples entirely inside the flat basin and receives no fitness
    gradient, so it collapses. Narrow-corridor CKKS defects (Taylor-3
    polynomial blow-up outside |z| <= 3) are the canonical case; see
    ``research/future-work/14-autoconfig-v2-regime-routing.md``.
    """
    max_centre = float(centre_divs.max()) if centre_divs.size > 0 else 0.0
    max_full = float(full_divs.max()) if full_divs.size > 0 else 0.0
    if max_full < 1e-8:
        return False
    return max_centre < _DISTANT_DEFECT_RATIO * max_full


@dataclass
class ProbeResult:
    """Outcome of a landscape probe.

    Attributes
    ----------
    regime : Regime
        Detected landscape regime.
    probe_divergences : np.ndarray
        Per-probe divergence values (length ``n_probes``).
    recommendation : dict
        Dispatch recipe (strategy name + kwargs).
    probe_points : np.ndarray, optional
        Uniform probe inputs, row-aligned with ``probe_divergences``.
    n_evals : int
        Evaluations charged to the budget, including the structure diagnostic.
    best_input, best_divergence
        Largest divergence over all charged evaluations and its input.
    """

    regime: Regime
    probe_divergences: np.ndarray
    recommendation: dict = field(default_factory=dict)
    probe_points: Optional[np.ndarray] = None
    n_evals: int = 0
    best_input: Optional[list[float]] = None
    best_divergence: float = -np.inf


def _divergence(plaintext_fn: Callable, fhe_fn: Callable, x: np.ndarray) -> float:
    """Reducer-max absolute divergence |plain(x) - fhe(x)|."""
    return DivergenceFitness(plaintext_fn, fhe_fn).score(x)


class _Renumbered:
    """``on_evaluation`` wrapper giving every event of a run one consecutive ``index``."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink
        self.count = 0

    def __call__(self, event: dict[str, Any]) -> None:
        self._sink({**event, "index": self.count})
        self.count += 1


def classify_landscape(
    plaintext_fn: Callable,
    fhe_fn: Callable,
    bounds: list[tuple[float, float]],
    n_probes: int = 50,
    W: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    seed: int = 0,
    second_pass_probes: int = 50,
    max_evals: Optional[int] = None,
    on_evaluation: Optional[EventSink] = None,
) -> ProbeResult:
    """Run ``n_probes`` uniform random evaluations and classify.

    Parameters
    ----------
    plaintext_fn, fhe_fn : callable
        Reference and FHE functions. Either scalar or array output.
    bounds : list of (lo, hi)
        Per-dimension input box.
    n_probes : int, default 50
        Number of random probes. Typical 30-200.
    W, b : array-like, optional
        Affine front-end weight/bias. When supplied enables the
        preactivation-dominance test.
    seed : int, default 0
        RNG seed for probe reproducibility.
    second_pass_probes : int, default 50
        Extra probes to draw when the first pass is borderline for
        plateau-cliff (CV in (0.1, 0.5) and ``max > 3 * median`` but
        no test fires). Set to ``0`` to disable.
    max_evals : int, optional
        Cap on charged evaluations. Later stages are skipped if they would exceed it,
        the structure diagnostic also if it needs over half; the first pass always runs.
    on_evaluation : callable, optional
        Receives one core-shaped event per charged evaluation (kind ``probe`` or ``structure``).

    Returns
    -------
    ProbeResult
    """
    if n_probes <= 0:
        raise ValueError("n_probes must be positive")

    rng = np.random.default_rng(int(seed) ^ _PROBE_SEED_SALT)
    d = len(bounds)
    lo = np.array([bd[0] for bd in bounds], dtype=np.float64)
    hi = np.array([bd[1] for bd in bounds], dtype=np.float64)
    n_evals = 0
    best_x: Optional[list[float]] = None
    best_s = -np.inf

    def div(x: np.ndarray, kind: str = "probe") -> float:
        nonlocal n_evals, best_x, best_s
        s = _divergence(plaintext_fn, fhe_fn, x)
        if on_evaluation is not None:
            on_evaluation(_event(n_evals, kind, x, score=s))
        n_evals += 1
        if s > best_s:
            best_x, best_s = np.asarray(x, dtype=np.float64).tolist(), s
        return s

    def fits(k: int) -> bool:
        return max_evals is None or n_evals + k <= max_evals

    probes = rng.uniform(lo, hi, size=(n_probes, d))
    divs = np.array([div(probes[i]) for i in range(n_probes)], dtype=np.float64)

    def done(regime: Regime, recommendation: dict) -> ProbeResult:
        return ProbeResult(
            regime=regime,
            probe_divergences=divs,
            recommendation=recommendation,
            probe_points=probes,
            n_evals=n_evals,
            best_input=best_x,
            best_divergence=best_s,
        )

    max_div = float(np.max(divs))
    med_div = float(np.median(divs))
    std_div = float(np.std(divs))
    mean_div = float(np.mean(divs))

    # 1. Full-domain saturation: divergence "high" (>50% of max) almost
    #    everywhere. Covers both genuinely saturated circuits (Concrete-ML
    #    4-bit MLPs) and degenerate flat circuits where every probe has
    #    the same divergence.
    if max_div > 0.0:
        high_frac = float(np.mean(divs > 0.5 * max_div))
    else:
        high_frac = 1.0
    if high_frac > 0.90:
        return done(Regime.FULL_DOMAIN_SATURATION, {
            "strategy": "random_only",
            "reason": (
                f"{high_frac:.0%} of probes exceed 50% of max divergence "
                f"-- no concentrated bug region"
            ),
            "random_floor": 1.0,
            "warm_start": False,
        })

    # 2. Plateau-then-cliff: detected by the helper's three-test ensemble.
    #    On borderline cases (CV in (0.1, 0.5) with a meaningful cliff
    #    signal) we draw an extra batch and re-run the tests on the
    #    combined sample -- this rescues circuits like Chebyshev TenSEAL
    #    where the cliff is too narrow for 50 probes to resolve.
    if _detect_plateau_cliff(divs):
        cv = std_div / mean_div if mean_div > 0.0 else 0.0
        return done(Regime.PLATEAU_THEN_CLIFF, {
            "strategy": "warm_start",
            "reason": (
                f"Plateau-cliff detected (CV={cv:.3f}, "
                f"max/med={(max_div / med_div) if med_div > 0 else float('inf'):.1f}x)"
            ),
            "random_floor": 0.3,
            "warm_start": True,
        })

    if (
        second_pass_probes > 0
        and mean_div > 0.0
        and med_div > 0.0
        and fits(int(second_pass_probes))
    ):
        cv = std_div / mean_div if mean_div > 0.0 else 0.0
        if 0.1 < cv < 0.5 and max_div > 3.0 * med_div:
            extra = rng.uniform(lo, hi, size=(int(second_pass_probes), d))
            extra_divs = np.array(
                [div(extra[i]) for i in range(extra.shape[0])], dtype=np.float64
            )
            probes = np.concatenate([probes, extra], axis=0)
            divs = np.concatenate([divs, extra_divs])
            max_div = float(np.max(divs))
            med_div = float(np.median(divs))
            std_div = float(np.std(divs))
            mean_div = float(np.mean(divs))
            if _detect_plateau_cliff(divs):
                c_cv = std_div / mean_div if mean_div > 0.0 else 0.0
                return done(Regime.PLATEAU_THEN_CLIFF, {
                    "strategy": "warm_start",
                    "reason": (
                        f"Plateau-cliff detected after second pass "
                        f"(n={divs.size}, CV={c_cv:.3f}, "
                        f"max/med={(max_div / med_div) if med_div > 0 else float('inf'):.1f}x)"
                    ),
                    "random_floor": 0.3,
                    "warm_start": True,
                })

    # 3. Preactivation-dominated: delta correlates with |Wx+b|. Only
    #    runs when W, b are supplied.
    if W is not None and b is not None:
        W_arr = np.atleast_2d(np.asarray(W, dtype=np.float64))
        b_arr = np.atleast_1d(np.asarray(b, dtype=np.float64)).astype(np.float64)
        # Use probes.shape[0], not n_probes — `probes` may have been
        # extended by the second-pass plateau-cliff branch above.
        preacts = np.array(
            [float(np.max(np.abs(W_arr @ probes[i] + b_arr)))
             for i in range(probes.shape[0])],
            dtype=np.float64,
        )
        assert preacts.size == divs.size, (
            f"preacts/divs length mismatch: {preacts.size} vs {divs.size}"
        )
        # Guard against degenerate constant inputs that make Spearman NaN.
        if np.std(preacts) > 0.0 and np.std(divs) > 0.0:
            _sr = spearmanr(divs, preacts)
            corr = float(_sr.statistic) if np.isfinite(_sr.statistic) else 0.0
            pval = float(_sr.pvalue) if np.isfinite(_sr.pvalue) else 1.0
            if corr > 0.7:
                return done(Regime.PREACTIVATION_DOMINATED, {
                    "strategy": "preactivation",
                    "reason": (
                        f"Spearman(delta, |Wx+b|) = {corr:.2f} "
                        f"(p={pval:.1e}) -- divergence factors through "
                        f"preactivation"
                    ),
                    "use_preactivation": True,
                    "preactivation_rank": int(W_arr.shape[0]),
                })

    # 4. Distant-defect: fitness concentrated far from box centre.
    #    CMA-ES with default sigma0=1.0 at the box midpoint would
    #    start inside a flat basin; dispatch to sigma0=auto +
    #    heuristic seeds so the search escapes the basin. Uses a
    #    dedicated centre-ball probe (20 evals in addition to the
    #    main probe batch) so the test is reliable across seeds and
    #    dimensions.
    if fits(_DISTANT_DEFECT_CENTER_PROBES):
        centre_divs = _distant_defect_probe(div, bounds, rng)
        if _detect_distant_defect(centre_divs, divs):
            return done(Regime.DISTANT_DEFECT, {
                "strategy": "robust_cma_es",
                "reason": (
                    "Divergence concentrated far from box centre "
                    "-- default CMA-ES would basin-trap; using "
                    "sigma0=auto + heuristic seeds"
                ),
                "sigma0": None,
                "use_heuristic_seeds": True,
                "heuristic_k": 10,
            })

    # 5. Low-rank structure in the divergence surface (measured via SVD,
    #    not dimension alone -- the d>100->SubspaceOracle heuristic
    #    below was reverted for firing on isotropic high-d circuits).
    #    Charged at up to 2*d evaluations per sample; skipped when it does not fit.
    structure_cost = _LOW_RANK_STRUCTURE_SAMPLES * 2 * d
    if d >= _LOW_RANK_MIN_DIM and fits(structure_cost) and (
        max_evals is None or structure_cost <= max_evals // 2
    ):
        def _delta(x: list[float]) -> float:
            return div(np.asarray(x, dtype=np.float64), "structure")

        structure = characterize_structure(
            _delta,
            d,
            bounds,
            n_samples=_LOW_RANK_STRUCTURE_SAMPLES,
            seed=int(seed) ^ _PROBE_SEED_SALT ^ 0x10AA,
        )
        if structure.effective_rank > 0 and structure.effective_rank <= int(
            d * _LOW_RANK_RANK_FRACTION
        ):
            return done(Regime.LOW_RANK_STRUCTURE, {
                "strategy": "separable_cma_es",
                "reason": (
                    f"characterize_structure found effective_rank="
                    f"{structure.effective_rank} of dim={d} "
                    f"-- diagonal-covariance search is likely to help"
                ),
                "separable": True,
                "effective_rank": structure.effective_rank,
            })

    # 6. Standard fall-through.
    return done(Regime.STANDARD, {
        "strategy": "cma_es",
        "reason": "No extreme regime detected -- standard CMA-ES search",
        "random_floor": 0.0,
        "warm_start": False,
    })


def _boundary_probe(
    plaintext_fn: Callable,
    fhe_fn: Callable,
    bounds: list[tuple[float, float]],
    probe: ProbeResult,
    budget: int,
    on_evaluation: Optional[EventSink] = None,
) -> tuple[Optional[list[float]], float, int]:
    """Snap top probe points to vertices; flip-climb only if a vertex beats the probes.

    Returns ``(best_vertex, divergence, evaluations_used)``.
    """
    pts, divs = probe.probe_points, probe.probe_divergences
    if budget <= 0 or pts is None or divs.size == 0:
        return None, -np.inf, 0
    lo = np.array([bd[0] for bd in bounds], dtype=np.float64)
    hi = np.array([bd[1] for bd in bounds], dtype=np.float64)
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    used = 0
    best_x: Optional[np.ndarray] = None
    best_s, best_rel = -np.inf, np.zeros_like(mid)
    tried: set[bytes] = set()

    def score(x: np.ndarray) -> float:
        nonlocal used
        s = _divergence(plaintext_fn, fhe_fn, x)
        if on_evaluation is not None:
            on_evaluation(_event(used, "boundary", x, score=s))
        used += 1
        return s

    for i in np.argsort(-divs, kind="stable")[:_BOUNDARY_TOP_K]:
        rel = np.divide(pts[i] - mid, half, out=np.zeros_like(mid), where=half > 0)
        v = np.where(rel >= 0.0, hi, lo)
        if used >= budget or v.tobytes() in tried:
            continue
        tried.add(v.tobytes())
        s = score(v)
        if s > best_s:
            best_x, best_s, best_rel = v, s, rel
    if best_x is None:
        return None, -np.inf, used

    mirror = np.where(best_x == hi, lo, hi)  # exact bounds; lo + hi - x can round outside
    if used < budget and mirror.tobytes() not in tried:
        s = score(mirror)
        if s > best_s:
            best_x, best_s = mirror, s

    # Interior worst case: the boundary lost to the probes, so stop here.
    if best_s <= float(np.max(divs)):
        return best_x.tolist(), best_s, used

    # Least-confident coordinates (nearest the centre) are flipped first.
    order = [j for j in np.argsort(np.abs(best_rel), kind="stable") if half[j] > 0]
    improved = True
    while improved and used < budget:
        improved = False
        for j in order:
            if used >= budget:
                break
            y = best_x.copy()
            y[j] = lo[j] if y[j] == hi[j] else hi[j]
            s = score(y)
            if s > best_s:
                best_x, best_s, improved = y, s, True
    return best_x.tolist(), best_s, used


class AutoOracle:
    """Auto-configuring FHE Oracle.

    Runs a probe phase to classify the landscape, then dispatches to
    the appropriate search strategy. The recommended entry point for
    users who do not know their circuit's landscape.

    Parameters
    ----------
    plaintext_fn, fhe_fn : callable
        Plaintext and FHE functions under test.
    bounds : list of (lo, hi)
        Per-dimension input box.
    W, b : array-like, optional
        Affine front-end weight matrix and bias. When supplied the
        probe tests preactivation dominance; if detected, dispatch uses
        :class:`PreactivationOracle`.
    n_probes : int, default 50
        First-pass probe evaluations; like all probes, counted in ``n_trials`` and
        ``result.n_trials``.
    **oracle_kwargs
        Passed through to the underlying :class:`FHEOracle` or
        :class:`PreactivationOracle` (e.g. ``sigma0``, ``separable``).

    Notes
    -----
    After :meth:`run`, ``self.probe_result`` exposes the full
    :class:`ProbeResult`, and ``self.last_oracle`` exposes the inner
    ``FHEOracle`` instance (e.g. for ``.shrink()``) -- ``None`` when
    dispatch used ``PreactivationOracle`` instead.

    ``result.n_trials`` counts every evaluation except re-measurements (probes, structure
    diagnostic, boundary probe, search), as FHEOracle does. A better pre-search witness
    replaces ``worst_input``. An ``on_evaluation`` kwarg receives all of them plus each
    re-measurement, with one consecutive ``index`` (kinds probe, structure, boundary,
    search, remeasure).
    """

    def __init__(
        self,
        plaintext_fn: Callable,
        fhe_fn: Callable,
        bounds: list[tuple[float, float]],
        W: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        n_probes: int = 50,
        **oracle_kwargs: Any,
    ) -> None:
        if n_probes <= 0:
            raise ValueError("n_probes must be positive")
        self.plaintext_fn = plaintext_fn
        self.fhe_fn = fhe_fn
        self.bounds = list(bounds)
        self.d = len(bounds)
        self.W = W
        self.b = b
        self.n_probes = int(n_probes)
        self.oracle_kwargs = oracle_kwargs
        self.probe_result: Optional[ProbeResult] = None
        self.last_oracle: Optional[Any] = None  # inner FHEOracle from the last run(); None for PreactivationOracle dispatch
        self._events: Optional[_Renumbered] = None  # this run's on_evaluation wrapper, if any

    def _attach_meta(self, result, regime: Regime, strategy: str):
        """Tag result with regime/strategy. Works for OracleResult and
        PreactivationResult alike (both non-frozen dataclasses)."""
        try:
            result.regime = regime.value
            result.strategy_used = strategy
        except (AttributeError, TypeError):
            # Frozen dataclass -- skip silently.
            pass
        return result

    def _finish(
        self,
        oracle: FHEOracle,
        regime: Regime,
        strategy: str,
        presearch: Optional[_Boundary],
        n_trials: int,
        threshold: float,
        run_kwargs: dict[str, Any],
    ) -> OracleResult:
        """Run the inner search, merge any pre-search witness, count probe evaluations, tag."""
        self.last_oracle = oracle
        result = oracle.run(n_trials=n_trials, threshold=threshold, **run_kwargs)
        if presearch is not None:
            result = self._merge_presearch(result, oracle, presearch)
        assert self.probe_result is not None
        result.n_trials += self.probe_result.n_evals
        return self._attach_meta(result, regime, strategy)

    def _merge_presearch(
        self, result: OracleResult, oracle: FHEOracle, boundary: _Boundary
    ) -> OracleResult:
        """Count boundary evaluations; a larger pre-search witness is re-measured like core's."""
        x_b, s_b, used = boundary
        result.n_trials += used
        cands: list[tuple[float, Optional[list[float]]]] = [(s_b, x_b)]
        if self.probe_result is not None:
            cands.append((self.probe_result.best_divergence, self.probe_result.best_input))
        s, x = max(((s, x) for s, x in cands if x is not None),
                   key=lambda c: c[0], default=(-np.inf, None))
        if x is None or not s > result.max_error:
            return result
        # Same rule as FHEOracle.run: the counted evaluation and its re-measurement both count.
        # _measure_divergence returns (error, noise_state, class_flip).
        measured = oracle._measure_divergence(list(x), s)
        if self._events is not None:
            self._events(_event(0, "remeasure", x, error=measured[0]))
        result.worst_input = list(x)
        result.remeasured_error = float(measured[0])
        result.noise_state = measured[1]
        result.search_max_error = float(s)
        result.max_error = max(float(s), float(measured[0]))
        if result.max_error >= result.threshold:
            result.verdict = "FAIL"
        return result

    def _preactivation_oracle(self) -> PreactivationOracle:
        """PreactivationOracle whose model calls reach this run's event log, if any."""
        from .preactivation import PreactivationOracle

        return PreactivationOracle(W=self.W, b=self.b, plaintext_fn=self.plaintext_fn,
                                   fhe_fn=self.fhe_fn, input_bounds=self.bounds,
                                   on_evaluation=self._events)

    def run(
        self,
        n_trials: int = 500,
        seed: int = 42,
        threshold: float = 1e-2,
        **run_kwargs: Any,
    ):
        """Probe, classify, probe the boundary, dispatch.

        Parameters
        ----------
        n_trials : int, default 500
            Total evaluation budget, including every probe and re-measurement.
            Minimum ``n_probes + 3``, or ``n_probes + 10`` when ``W`` has a single row.
        seed : int, default 42
            Seed for probe RNG and search.
        threshold : float, default 1e-2
            PASS/FAIL cut-off passed to the inner oracle.
        **run_kwargs
            Additional kwargs forwarded to the inner ``run()``.

        Returns
        -------
        result : OracleResult or PreactivationResult
            Augmented with ``.regime`` and ``.strategy_used`` attributes.
        """
        reserve = _RESERVE
        if self.W is not None and self.b is not None and np.atleast_2d(self.W).shape[0] == 1:
            reserve = _RESERVE_1D_PREACT
        if n_trials < self.n_probes + reserve:
            raise ValueError(
                f"n_trials ({n_trials}) must be at least n_probes + {reserve} "
                f"({self.n_probes + reserve})"
            )
        self.last_oracle = None  # reset -- stays None if this run dispatches to PreactivationOracle
        sink = self.oracle_kwargs.get("on_evaluation")
        self._events = _Renumbered(sink) if sink is not None else None
        okw = dict(self.oracle_kwargs)
        if self._events is not None:
            okw["on_evaluation"] = self._events

        self.probe_result = classify_landscape(
            self.plaintext_fn,
            self.fhe_fn,
            self.bounds,
            n_probes=self.n_probes,
            W=self.W,
            b=self.b,
            seed=seed,
            max_evals=min(max(self.n_probes, n_trials - self.n_probes), n_trials - reserve),
            on_evaluation=self._events,
        )

        regime = self.probe_result.regime
        used = self.probe_result.n_evals
        # The probe scores fhe_fn divergence, so it stays out of other search objectives.
        merge = regime != Regime.PREACTIVATION_DOMINATED and not any(
            self.oracle_kwargs.get(k) for k in _OTHER_OBJECTIVES
        )
        boundary: tuple[Optional[list[float]], float, int] = (None, -np.inf, 0)
        if merge:
            cap = min(
                math.ceil(_BOUNDARY_SHARE * n_trials),
                _BOUNDARY_TOP_K + 1 + 2 * self.d,
                max(0, (n_trials - used - _RESERVE) // 2),
            )
            boundary = _boundary_probe(
                self.plaintext_fn, self.fhe_fn, self.bounds, self.probe_result, cap,
                self._events,
            )
            used += boundary[2]
        # Held back: the inner run's re-measurement, plus one for a pre-search witness.
        remaining_budget = n_trials - used - (2 if merge else 1)
        presearch = boundary if merge else None

        if regime == Regime.FULL_DOMAIN_SATURATION:
            from .core import FHEOracle

            oracle = FHEOracle(
                plaintext_fn=self.plaintext_fn,
                fhe_fn=self.fhe_fn,
                input_dim=self.d,
                input_bounds=self.bounds,
                seed=seed,
                random_floor=1.0,
                **okw,
            )
            return self._finish(oracle, regime, "random_only", presearch,
                                remaining_budget, threshold, run_kwargs)

        if regime == Regime.PLATEAU_THEN_CLIFF:
            from .core import FHEOracle

            oracle = FHEOracle(
                plaintext_fn=self.plaintext_fn,
                fhe_fn=self.fhe_fn,
                input_dim=self.d,
                input_bounds=self.bounds,
                seed=seed,
                random_floor=0.3,
                warm_start=True,
                **okw,
            )
            return self._finish(oracle, regime, "warm_start", presearch,
                                remaining_budget, threshold, run_kwargs)

        if regime == Regime.DISTANT_DEFECT:
            from .core import FHEOracle

            # Merge recommendation into user oracle_kwargs; user wins
            # on any explicit override.
            kw: dict[str, Any] = {
                "sigma0": None,
                "use_heuristic_seeds": True,
                "heuristic_k": 10,
            }
            kw.update(okw)

            oracle = FHEOracle(
                plaintext_fn=self.plaintext_fn,
                fhe_fn=self.fhe_fn,
                input_dim=self.d,
                input_bounds=self.bounds,
                seed=seed,
                **kw,
            )
            return self._finish(oracle, regime, "robust_cma_es", presearch,
                                remaining_budget, threshold, run_kwargs)

        if regime == Regime.LOW_RANK_STRUCTURE:
            from .core import FHEOracle

            kw = {"separable": True}
            kw.update(okw)

            oracle = FHEOracle(
                plaintext_fn=self.plaintext_fn,
                fhe_fn=self.fhe_fn,
                input_dim=self.d,
                input_bounds=self.bounds,
                seed=seed,
                **kw,
            )
            return self._finish(oracle, regime, "separable_cma_es", presearch,
                                remaining_budget, threshold, run_kwargs)

        if regime == Regime.PREACTIVATION_DOMINATED:
            preact = self._preactivation_oracle()
            results = preact.run(budget=remaining_budget, seeds=[seed], threshold=threshold)
            results[0].n_trials += self.probe_result.n_evals
            return self._attach_meta(results[0], regime, "preactivation")

        # STANDARD -- dispatch to full CMA-ES. An earlier version routed
        # d > 100 to SubspaceOracle on the assumption that high-d circuits
        # benefit from random projection, but benchmark sweeps show this
        # regresses on spherical-defect circuits (e.g. lr_mock_d128 where
        # the noise amplification triggers on ||x||^2/d -- isotropic, not
        # low-rank). Subspace routing is now opt-in via the explicit
        # PREACTIVATION_DOMINATED regime (requires W, b) or direct
        # SubspaceOracle instantiation. See
        # research/future-work/14-autoconfig-v2-regime-routing.md.
        from .core import FHEOracle

        oracle = FHEOracle(
            plaintext_fn=self.plaintext_fn,
            fhe_fn=self.fhe_fn,
            input_dim=self.d,
            input_bounds=self.bounds,
            seed=seed,
            **okw,
        )
        return self._finish(oracle, regime, "cma_es", presearch,
                            remaining_budget, threshold, run_kwargs)
