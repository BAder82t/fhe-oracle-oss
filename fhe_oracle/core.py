# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FHEOracle: CMA-ES adversarial search for FHE precision bugs.

Finds inputs that maximise the divergence between a plaintext function
and its FHE-compiled counterpart. Designed for the case where random
sampling misses rare precision outliers (typical: ~1 in 10,000 to
1 in 1,000,000 inputs).

Public API
----------
    oracle = FHEOracle(plaintext_fn, fhe_fn, input_dim, input_bounds)
    result = oracle.run(n_trials=500, threshold=0.01)

    result.verdict      # "PASS" or "FAIL"
    result.max_error    # largest divergence found
    result.worst_input  # input vector that triggered max_error
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import registry
from .adaptive import AdaptiveBudget, AdaptiveConfig
from .diversity import DiversityInjector, InjectionStrategy
from .fitness import DivergenceFitness, EvaluationError, absolute_error, finite_score
from .guarantees import CoverageCertificate
from .multi_output import MultiOutputFitness, MultiOutputMode
from .seeds import fallback_corner_seeds


def _build_seeds(
    rng: np.random.Generator,
    bounds: list[tuple[float, float]],
    k: int,
    which: tuple[str, ...],
    tau: Optional[float],
) -> list[list[float]]:
    """Use the registered ``generate_seeds`` heuristic plugin if present,
    otherwise fall back to corner+random sampling.

    The plugin (registered via the ``fhe_oracle.heuristics`` entry-point
    group under name ``generate_seeds``) is a callable with signature
    ``(rng, bounds, k, tau, which) -> list[list[float]]``.
    """
    try:
        gen = registry.get_heuristic("generate_seeds")
    except KeyError:
        return fallback_corner_seeds(rng, bounds, k=k)
    return gen(rng, bounds, k=k, tau=tau, which=which)


@dataclass
class OracleResult:
    """Outcome of an adversarial oracle run.

    Attributes
    ----------
    verdict : str
        "FAIL" if a counted evaluation or the re-measurement met the
        threshold (or flipped the class in rank modes), else "PASS".
    max_error : float
        Largest divergence |plaintext_fn(x) - fhe_fn(x)| observed.
    worst_input : list[float]
        Input vector that produced max_error.
    threshold : float
        User-supplied tolerance.
    n_trials : int
        Number of fitness evaluations performed.
    elapsed_seconds : float
        Wall-clock search time.
    scheme : str
        FHE scheme name (from adapter), "fhe_fn" for a callable, or "custom-fitness".
    noise_state : dict[str, float | str]
        Noise-budget snapshot at worst_input when an adapter was used.
        Empty dict in pure-divergence mode. Holds a single ``"error"``
        string key instead if re-measurement raised.
    search_max_error : float, optional
        Largest error seen by a counted search evaluation when the fitness
        measures error directly; a value at or above threshold means FAIL.
    remeasured_error : float, optional
        Error from the final re-measurement at ``worst_input``.
    class_flip : bool, optional
        Multi-output rank modes only: an argmax flip was seen during
        search or at the re-measurement; a flip means FAIL.
    strategy_used, subspace_dim, n_projections, n_anchors,
    projection_index, probe_max, fallback_taken
        Set only by :class:`~fhe_oracle.subspace.SubspaceOracle`;
        ``None``/unset otherwise. See that class for their meaning.
    """

    verdict: str
    max_error: float
    worst_input: list[float]
    threshold: float
    n_trials: int
    elapsed_seconds: float
    scheme: str = "fhe_fn"
    noise_state: dict[str, float | str] = field(default_factory=dict)
    coverage_certificate: Optional["CoverageCertificate"] = None
    n_restarts_used: int = 0
    adaptive_stop_reason: Optional[str] = None
    adaptive_extensions_used: int = 0
    diversity_injections: int = 0
    strategy_used: Optional[str] = None
    subspace_dim: Optional[int] = None
    n_projections: Optional[int] = None
    n_anchors: Optional[int] = None
    projection_index: Optional[int] = None
    probe_max: Optional[float] = None
    fallback_taken: Optional[bool] = None
    search_max_error: Optional[float] = None
    remeasured_error: Optional[float] = None
    class_flip: Optional[bool] = None

    def __repr__(self) -> str:
        return (
            f"OracleResult(verdict={self.verdict!r}, "
            f"max_error={self.max_error:.6e}, "
            f"trials={self.n_trials}, "
            f"elapsed={self.elapsed_seconds:.2f}s)"
        )


@dataclass
class ShrinkResult:
    """Outcome of shrinking a FAIL witness toward a reference point.

    Attributes
    ----------
    original_input : list[float]
        The witness as found by ``run()`` (``result.worst_input``).
    shrunk_input : list[float]
        The minimised input: as close to the reference point as
        possible while divergence still meets ``threshold``.
    original_norm, shrunk_norm : float
        Euclidean distance from the reference point for each input.
    max_error : float
        Lowest confirming re-measurement at ``shrunk_input``; always meets
        ``threshold``. On noisy backends the point must clear the threshold
        by the measurement spread, else shrink retreats toward the original.
    threshold : float
        The PASS/FAIL threshold ``shrunk_input`` was constrained to
        keep meeting (copied from the source ``OracleResult``).
    n_evals : int
        Evaluations spent shrinking, including final verification.
    """

    original_input: list[float]
    shrunk_input: list[float]
    original_norm: float
    shrunk_norm: float
    max_error: float
    threshold: float
    n_evals: int

    def __repr__(self) -> str:
        reduction = (
            100.0 * (1.0 - self.shrunk_norm / self.original_norm)
            if self.original_norm > 0
            else 0.0
        )
        return (
            f"ShrinkResult(reduction={reduction:.1f}%, "
            f"max_error={self.max_error:.6e}, "
            f"evals={self.n_evals})"
        )


class FHEOracle:
    """Adversarial CMA-ES search for FHE precision bugs.

    Parameters
    ----------
    plaintext_fn : callable
        Reference implementation.
        ``plaintext_fn(x: list[float]) -> float | list[float]``.
    fhe_fn : callable, optional
        FHE implementation under test. Same signature as plaintext_fn.
        Supply this OR ``adapter``.
    input_dim : int
        Dimensionality of the input space.
    input_bounds : list[tuple[float, float]], optional
        Per-dimension ``(low, high)`` box constraints. If a single
        ``(low, high)`` tuple is given, it is broadcast. If None, the
        search is unconstrained.
    adapter : FHEAdapter, optional
        An instrumented FHE adapter enabling noise-guided search.
        When provided, fhe_fn is optional (the adapter runs the
        circuit).
    fitness : object, optional
        Custom fitness object with a ``score(x) -> float`` method.
        Overrides fhe_fn/adapter.
    sigma0 : float
        Initial CMA-ES step size. Default 1.0.
    x0 : list[float], optional
        Initial mean. Defaults to the midpoint of input_bounds or zeros.
    seed : int, optional
        Random seed for reproducibility.
    w_div : float, default 1.0
        Weight for pure divergence in the noise-guided fitness
        (applied only when ``adapter`` is supplied). The historical
        ``w_noise`` and ``w_depth`` shaping weights were removed in
        v0.5.1: paper §6.15 showed they are inert on tested CKKS
        circuits, and the Item 17 Lattigo correlation experiment
        (2026-04-28) found the level-proportional depth proxy
        empirically uncorrelated with decrypt-based precision at
        d ≥ 4 (Spearman ρ = 0.07). See
        ``research/future-work/17-results.md``.
    """

    def __init__(
        self,
        plaintext_fn: Callable[[list[float]], float | list[float]],
        fhe_fn: Optional[Callable[[list[float]], float | list[float]]] = None,
        input_dim: int = 0,
        input_bounds: Optional[list[tuple[float, float]] | tuple[float, float]] = None,
        adapter: Any = None,
        fitness: Any = None,
        sigma0: Optional[float] = 1.0,
        x0: Optional[list[float]] = None,
        seed: Optional[int] = None,
        use_heuristic_seeds: bool = False,
        heuristic_which: tuple[str, ...] = ("mm", "ds", "nt"),
        heuristic_k: int = 10,
        heuristic_tau: Optional[float] = None,
        random_floor: float = 0.0,
        warm_start: bool = True,
        warm_sigma_scale: float = 0.2,
        restarts: int = 0,
        bipop: bool = False,
        restart_popsize_factor: float = 2.0,
        stall_generations: int = 10,
        stall_tol: float = 1e-8,
        separable: bool = False,
        w_div: float = 1.0,
        adaptive: bool = False,
        adaptive_config: Optional[AdaptiveConfig] = None,
        diversity_injection: bool = False,
        inject_every: int = 5,
        inject_count: int = 3,
        inject_strategy: str = "mixed",
        multi_output: bool = False,
        multi_output_mode: str = "combined",
        rank_weight: float = 1.0,
        batch_fhe_fn: Optional[Callable[[list[list[float]]], Any]] = None,
        on_evaluation: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if fitness is None and fhe_fn is None and adapter is None:
            raise ValueError(
                "Provide one of: fhe_fn, adapter, or a custom fitness object."
            )
        if batch_fhe_fn is not None and (
            fhe_fn is None or fitness is not None or adapter is not None
        ):
            raise ValueError(
                "batch_fhe_fn needs fhe_fn and the built-in fitness "
                "(no adapter or custom fitness object)"
            )

        if not (0.0 <= random_floor <= 1.0):
            raise ValueError("random_floor must be in [0.0, 1.0]")
        if restarts < 0:
            raise ValueError("restarts must be non-negative")
        if restart_popsize_factor < 1.0:
            raise ValueError("restart_popsize_factor must be >= 1.0")
        if stall_generations < 1:
            raise ValueError("stall_generations must be >= 1")

        self._plaintext_fn = plaintext_fn
        self._fhe_fn = fhe_fn
        self._adapter = adapter
        self._input_dim = input_dim
        self._bounds = _normalise_bounds(input_bounds, input_dim)
        if sigma0 is None:
            # Auto-scale from bounds. CMA-ES guideline: sigma0 ~= range / 4
            # covers the search box within ~2 standard deviations. Required
            # for narrow-corridor defect circuits (e.g. CKKS Taylor-3) where
            # default sigma0=1.0 traps the optimiser in a flat-fitness basin.
            if self._bounds is None:
                raise ValueError("sigma0=None requires input_bounds to be set")
            ranges = [hi - lo for lo, hi in self._bounds]
            sigma0 = float(np.mean(ranges)) / 4.0
        self._sigma0 = float(sigma0)
        self._seed = seed
        self._use_heuristic_seeds = use_heuristic_seeds
        self._heuristic_which = tuple(heuristic_which)
        self._heuristic_k = int(heuristic_k)
        self._heuristic_tau = heuristic_tau
        self._random_floor = float(random_floor)
        self._warm_start = bool(warm_start)
        self._warm_sigma_scale = float(warm_sigma_scale)
        self._restarts = int(restarts)
        self._bipop = bool(bipop)
        self._restart_popsize_factor = float(restart_popsize_factor)
        self._stall_generations = int(stall_generations)
        self._stall_tol = float(stall_tol)
        self._separable = bool(separable)
        self.w_div = float(w_div)
        self._scheme = (
            adapter.get_scheme_name() if adapter is not None
            else ("fhe_fn" if fhe_fn is not None else "custom-fitness")
        )

        if fitness is not None:
            self._fitness = fitness
        elif multi_output:
            mode_map = {
                "max_absolute": MultiOutputMode.MAX_ABSOLUTE,
                "rank_inversion": MultiOutputMode.RANK_INVERSION,
                "combined": MultiOutputMode.COMBINED,
            }
            if multi_output_mode not in mode_map:
                raise ValueError(
                    f"multi_output_mode must be one of {list(mode_map)}; "
                    f"got {multi_output_mode!r}"
                )
            if fhe_fn is None:
                raise ValueError("multi_output=True requires fhe_fn")
            self._fitness = MultiOutputFitness(
                plaintext_fn=plaintext_fn,
                fhe_fn=fhe_fn,
                mode=mode_map[multi_output_mode],
                rank_weight=float(rank_weight),
            )
        elif adapter is not None:
            # Adapter is provided but no noise-budget fitness is
            # registered: fall back to pure divergence using the
            # adapter's evaluate_with_state path. Item 17 showed the
            # historical noise-shaping fitness was based on a proxy
            # uncorrelated with decrypt-based precision at d >= 4.
            try:
                noise_cls = registry.get_fitness("noise_budget")
                self._fitness = noise_cls(
                    plaintext_fn, adapter, weights=(self.w_div,)
                )
            except KeyError:
                self._fitness = DivergenceFitness(
                    plaintext_fn, lambda x: adapter.evaluate(x)
                )
        else:
            # Guaranteed non-None: the constructor's top-level check
            # rejects fitness=None, fhe_fn=None, adapter=None together,
            # and this branch is only reached when fitness and adapter
            # are both None.
            assert fhe_fn is not None
            self._fitness = DivergenceFitness(plaintext_fn, fhe_fn)

        self._batch_fhe_fn = batch_fhe_fn
        self._on_evaluation = on_evaluation
        self._n_events = 0

        # Adaptive + diversity configuration (default OFF -> existing
        # behaviour is bit-identical to v0.3.x).
        self._adaptive = bool(adaptive)
        self._adaptive_config = (
            adaptive_config if adaptive_config is not None else AdaptiveConfig()
        )
        self._diversity_injection = bool(diversity_injection)
        self._inject_every = int(inject_every)
        self._inject_count = int(inject_count)
        strategy_map = {
            "corner": InjectionStrategy.CORNER,
            "uniform": InjectionStrategy.UNIFORM,
            "best_neighbor": InjectionStrategy.BEST_NEIGHBOR,
            "mixed": InjectionStrategy.MIXED,
        }
        if inject_strategy not in strategy_map:
            raise ValueError(
                f"inject_strategy must be one of {list(strategy_map)}; "
                f"got {inject_strategy!r}"
            )
        self._inject_strategy = strategy_map[inject_strategy]

        if x0 is None:
            if self._bounds is not None:
                self._x0 = [(lo + hi) / 2.0 for lo, hi in self._bounds]
            else:
                self._x0 = [0.0] * input_dim
        else:
            if len(x0) != input_dim:
                raise ValueError("len(x0) must equal input_dim")
            self._x0 = list(x0)

    def run(
        self,
        n_trials: int = 500,
        threshold: float = 1e-2,
        verbose: bool = False,
    ) -> OracleResult:
        """Execute the adversarial search.

        Parameters
        ----------
        n_trials : int
            Maximum number of fitness evaluations. Default 500.
        threshold : float
            PASS/FAIL cut-off on max divergence. Default 1e-2.
        verbose : bool
            If True, print CMA-ES progress.

        Returns
        -------
        OracleResult
        """
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "The 'cma' package is required. Install with: pip install cma"
            ) from exc

        if n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        t0 = time.perf_counter()

        best_input = list(self._x0)
        best_score = -np.inf
        self._n_events = 0
        if isinstance(self._fitness, MultiOutputFitness):
            self._fitness.reset_observations()
        total_evals = 0
        certificate: Optional[CoverageCertificate] = None
        cma_x0 = list(self._x0)
        cma_sigma0 = self._sigma0
        adaptive_stop_reason: Optional[str] = None
        adaptive_extensions_used = 0
        diversity_injections = 0

        # --- Random floor phase (A1 + A4) ---
        b_rand = int(self._random_floor * n_trials)
        if b_rand > 0:
            if self._bounds is None:
                raise ValueError(
                    "random_floor > 0 requires input_bounds to be set."
                )
            rng_floor = np.random.default_rng(
                (self._seed if self._seed is not None else 0) ^ 0xC0FFEE
            )
            lows = np.array([lo for lo, _ in self._bounds])
            highs = np.array([hi for _, hi in self._bounds])
            hits = 0
            best_rand_x = None
            best_rand_score = -np.inf
            floor_xs = [rng_floor.uniform(lows, highs) for _ in range(b_rand)]
            floor_scores = self._score_batch([list(x) for x in floor_xs])
            for x, score in zip(floor_xs, floor_scores):
                total_evals += 1
                if score > best_rand_score:
                    best_rand_score = score
                    best_rand_x = x.copy()
                if score >= threshold:
                    hits += 1
            # Update global best tracker
            if best_rand_x is not None and best_rand_score > best_score:
                best_score = best_rand_score
                best_input = best_rand_x.tolist()
            # Build certificate
            certificate = CoverageCertificate(
                budget_rand=b_rand,
                threshold=float(threshold),
                hits=hits,
                mu_hat=hits / b_rand,
            )
            # Warm-start CMA-ES
            if self._warm_start and best_rand_x is not None:
                cma_x0 = best_rand_x.tolist()
                cma_sigma0 = self._sigma0 * self._warm_sigma_scale

        # --- CMA-ES phase ---
        b_cma = n_trials - b_rand
        n_restarts_used = 0
        if b_cma > 0 and self._restarts == 0:
            # Single-run path (A1-identity invariant; unchanged).
            options: dict[str, Any] = {
                "maxfevals": b_cma,
                "verbose": 1 if verbose else -9,
                "tolx": 1e-12,
                "tolfun": 1e-15,
            }
            if self._seed is not None:
                options["seed"] = _cma_seed(self._seed)
            if self._bounds is not None:
                lows_b = [lo for lo, _ in self._bounds]
                highs_b = [hi for _, hi in self._bounds]
                options["bounds"] = [lows_b, highs_b]
            if self._separable:
                options["CMA_diagonal"] = True

            if self._input_dim == 1:
                # pycma's bounded maxstd adjustment assumes dimension > 1.
                # Boundary transforms still keep sampled inputs in bounds.
                options["maxstd"] = np.inf
            es = cma.CMAEvolutionStrategy(cma_x0, cma_sigma0, options)

            if (
                self._use_heuristic_seeds
                and self._heuristic_k > 0
                and self._bounds is not None
            ):
                rng = np.random.default_rng(
                    (self._seed if self._seed is not None else 0) ^ 0xC0FFEE
                )
                seeds_injected = _build_seeds(
                    rng,
                    self._bounds,
                    k=self._heuristic_k,
                    tau=self._heuristic_tau,
                    which=self._heuristic_which,
                )
                if seeds_injected:
                    try:
                        es.inject(seeds_injected, force=True)
                    except TypeError:
                        es.inject(seeds_injected)

            # Adaptive + diversity wiring. Both default OFF -> the
            # loop below is bit-identical to the v0.3.x path.
            adaptive_budget = (
                AdaptiveBudget(
                    self._adaptive_config,
                    budget=b_cma,
                    threshold=float(threshold),
                    initial_sigma=float(cma_sigma0),
                )
                if self._adaptive
                else None
            )
            injector: Optional[DiversityInjector] = None
            inject_rng: Optional[np.random.RandomState] = None
            if self._diversity_injection and self._bounds is not None:
                injector = DiversityInjector(
                    bounds=list(self._bounds),
                    inject_every=self._inject_every,
                    inject_count=self._inject_count,
                    strategy=self._inject_strategy,
                )
                inject_seed = (self._seed if self._seed is not None else 0) ^ 0x1ED5
                inject_rng = np.random.RandomState(inject_seed & 0x7FFFFFFF)

            cma_evals = 0
            generation = 0
            while not es.stop():
                solutions = es.ask()

                # Diversity injection: replace the LAST inject_count
                # entries of the solutions list with diverse samples.
                # Keeps population size constant; CMA-ES sees them in
                # tell() as ordinary candidates.
                if (
                    injector is not None
                    and inject_rng is not None
                    and injector.should_inject(generation)
                ):
                    injections = injector.generate_injections(
                        np.asarray(best_input, dtype=np.float64), inject_rng
                    )
                    n_replace = min(len(injections), len(solutions))
                    for i in range(n_replace):
                        solutions[-(i + 1)] = list(injections[i])
                    diversity_injections += n_replace

                # Stay within n_trials; a partial generation is not told.
                # Adaptive mode may extend the budget, so it is exempt.
                truncated = (
                    adaptive_budget is None
                    and len(solutions) > b_cma - cma_evals
                )
                if truncated:
                    solutions = solutions[: b_cma - cma_evals]
                fitnesses = self._score_batch([list(s) for s in solutions])
                for sol, s in zip(solutions, fitnesses):
                    total_evals += 1
                    cma_evals += 1
                    if s > best_score:
                        best_score = s
                        best_input = list(sol)
                if truncated:
                    break

                if adaptive_budget is not None:
                    adaptive_budget.record(
                        eval_num=cma_evals,
                        max_error=max(0.0, best_score),
                        sigma=float(es.sigma),
                    )
                    if adaptive_budget.should_stop():
                        adaptive_stop_reason = "early_stop_fail_found"
                        break
                    if adaptive_budget.should_switch():
                        adaptive_stop_reason = "stall_switch_to_random"
                        adaptive_budget.mark_switched()
                        es.tell(solutions, [-f for f in fitnesses])
                        # Spend remaining CMA-ES budget on uniform random.
                        if self._bounds is not None:
                            lows_b = [lo for lo, _ in self._bounds]
                            highs_b = [hi for _, hi in self._bounds]
                            switch_seed = (
                                (self._seed if self._seed is not None else 0) ^ 0x5736
                            )
                            switch_rng = np.random.default_rng(switch_seed)
                            while cma_evals < b_cma:
                                x = switch_rng.uniform(lows_b, highs_b)
                                s = self._score(list(x))
                                total_evals += 1
                                cma_evals += 1
                                if s > best_score:
                                    best_score = s
                                    best_input = list(x)
                        break

                es.tell(solutions, [-f for f in fitnesses])
                generation += 1

                # Auto-extend on climbing trajectory.
                if (
                    adaptive_budget is not None
                    and cma_evals >= b_cma
                    and adaptive_budget.should_extend()
                ):
                    extra = adaptive_budget.extension_budget()
                    # pycma reads maxfevals lazily through stop(); count the
                    # extension only if pycma accepted it.
                    try:
                        es.opts.set({"maxfevals": b_cma + extra})
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        warnings.warn(
                            f"fhe-oracle: could not extend the CMA-ES budget: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        b_cma += extra
                        adaptive_extensions_used += 1

                if cma_evals >= b_cma:
                    break
        elif b_cma > 0 and self._restarts > 0:
            # IPOP/BIPOP restart loop.
            if self._bounds is None:
                raise ValueError("restarts > 0 requires input_bounds to be set.")
            lows_b = [lo for lo, _ in self._bounds]
            highs_b = [hi for _, hi in self._bounds]
            lows_arr = np.array(lows_b)
            highs_arr = np.array(highs_b)

            base_popsize = 4 + int(np.floor(3 * np.log(self._input_dim)))
            current_popsize = base_popsize

            rng_restart = np.random.default_rng(
                (self._seed if self._seed is not None else 0) ^ 0xDEADBEEF
            )

            cma_evals_total = 0
            prev_best_score = best_score
            run_index = 0

            while run_index <= self._restarts and cma_evals_total < b_cma:
                # Pick x0/sigma for this run
                if run_index == 0:
                    run_x0 = list(cma_x0)
                    run_sigma = float(cma_sigma0)
                else:
                    run_x0 = rng_restart.uniform(lows_arr, highs_arr).tolist()
                    run_sigma = float(self._sigma0)

                # Pick popsize (IPOP grows; BIPOP alternates small/large)
                if self._bipop and run_index > 0 and run_index % 2 == 0:
                    run_popsize = max(2, base_popsize // 2)
                else:
                    run_popsize = current_popsize

                run_budget = b_cma - cma_evals_total
                if run_budget <= 0:
                    break

                run_options: dict[str, Any] = {
                    "maxfevals": run_budget,
                    "verbose": 1 if verbose else -9,
                    "tolx": 1e-12,
                    "tolfun": 1e-15,
                    "popsize": run_popsize,
                    "bounds": [lows_b, highs_b],
                }
                if self._seed is not None:
                    run_options["seed"] = _cma_seed(self._seed + run_index + 1)
                if self._separable:
                    run_options["CMA_diagonal"] = True

                if self._input_dim == 1:
                    run_options["maxstd"] = np.inf
                es = cma.CMAEvolutionStrategy(run_x0, run_sigma, run_options)

                # Heuristic seed injection only on the first run.
                if (
                    run_index == 0
                    and self._use_heuristic_seeds
                    and self._heuristic_k > 0
                ):
                    rng_inj = np.random.default_rng(
                        (self._seed if self._seed is not None else 0) ^ 0xC0FFEE
                    )
                    # Guaranteed non-None: this branch is only reached
                    # when self._restarts > 0, which raises above if
                    # self._bounds is None.
                    assert self._bounds is not None
                    seeds_injected = _build_seeds(
                        rng_inj,
                        self._bounds,
                        k=self._heuristic_k,
                        tau=self._heuristic_tau,
                        which=self._heuristic_which,
                    )
                    if seeds_injected:
                        try:
                            es.inject(seeds_injected, force=True)
                        except TypeError:
                            es.inject(seeds_injected)

                stall_count = 0
                while not es.stop() and cma_evals_total < b_cma:
                    solutions = es.ask()
                    remaining = b_cma - cma_evals_total
                    truncated = len(solutions) > remaining
                    if truncated:
                        solutions = solutions[:remaining]
                    fitnesses = self._score_batch([list(s) for s in solutions])
                    for sol, s in zip(solutions, fitnesses):
                        total_evals += 1
                        cma_evals_total += 1
                        if s > best_score:
                            best_score = s
                            best_input = list(sol)
                    if truncated:
                        break
                    es.tell(solutions, [-f for f in fitnesses])

                    # Stall detection on GLOBAL best.
                    if best_score - prev_best_score < self._stall_tol:
                        stall_count += 1
                    else:
                        stall_count = 0
                        prev_best_score = best_score

                    if stall_count >= self._stall_generations:
                        break  # stalled → trigger restart

                    if cma_evals_total >= b_cma:
                        break

                if run_index > 0:
                    n_restarts_used += 1
                current_popsize = int(
                    current_popsize * self._restart_popsize_factor
                )
                run_index += 1

        elapsed = time.perf_counter() - t0

        remeasured, noise_state, flip = self._measure_divergence(
            best_input, best_score
        )
        if self._adapter is not None or self._fhe_fn is not None:
            self._emit("remeasure", best_input, error=remeasured)
        rank_mode = (
            isinstance(self._fitness, MultiOutputFitness)
            and self._fitness.mode != MultiOutputMode.MAX_ABSOLUTE
        )
        search_max: Optional[float] = None
        if isinstance(self._fitness, DivergenceFitness) and self._fitness._reduce is np.max:
            search_max = float(best_score)
        elif isinstance(self._fitness, MultiOutputFitness):
            search_max = self._fitness.max_abs_seen
            if rank_mode:
                flip = bool(flip or self._fitness.flip_seen)
        # A violation seen during search is a FAIL even when a noisy
        # re-measurement of the same input lands below threshold.
        max_error = remeasured
        if search_max is not None and not rank_mode:
            max_error = max(remeasured, search_max)
        failed = (
            max_error >= threshold
            or bool(flip)
            or (search_max is not None and search_max >= threshold)
        )
        verdict = "FAIL" if failed else "PASS"

        return OracleResult(
            verdict=verdict,
            max_error=max_error,
            worst_input=best_input,
            threshold=threshold,
            n_trials=total_evals,
            elapsed_seconds=elapsed,
            scheme=self._scheme,
            noise_state=noise_state,
            coverage_certificate=certificate,
            n_restarts_used=n_restarts_used,
            adaptive_stop_reason=adaptive_stop_reason,
            adaptive_extensions_used=adaptive_extensions_used,
            diversity_injections=diversity_injections,
            search_max_error=search_max,
            remeasured_error=remeasured,
            class_flip=flip if rank_mode else None,
        )

    def shrink(
        self,
        result: OracleResult,
        reference: Optional[list[float]] = None,
        max_evals: int = 200,
    ) -> ShrinkResult:
        """Shrink a FAIL witness toward ``reference`` while it still fails.

        Per-coordinate binary search, keeping ``fitness.score(x) >=
        threshold`` at every step. Beat a constrained-CMA-ES
        re-optimisation pass 10/10 seeds in testing -- no fancier
        search needed here. The final point is re-measured as ``run()``
        measures its verdict; on noisy backends it may retreat toward
        the original witness.

        Parameters
        ----------
        result : OracleResult
            A FAIL result from ``run()`` on this same oracle instance
            (same plaintext_fn/fhe_fn/fitness). ``result.worst_input``
            is the starting witness.
        reference : list[float], optional
            The point to shrink toward. Defaults to the midpoint of
            ``input_bounds`` (or zeros if unconstrained).
        max_evals : int
            Evaluation budget for the whole shrink pass, including
            verification. Must be positive.

        Returns
        -------
        ShrinkResult
        """
        if result.verdict != "FAIL":
            raise ValueError(
                "shrink() requires a FAIL result (a witness that meets "
                "the threshold); nothing to shrink for a PASS result."
            )

        if max_evals < 1:
            raise ValueError("max_evals must be positive")

        x = np.array(result.worst_input, dtype=np.float64)
        dim = x.size

        if reference is not None:
            if len(reference) != dim:
                raise ValueError(
                    "len(reference) must match the witness dimension"
                )
            ref = np.array(reference, dtype=np.float64)
        elif self._bounds is not None:
            ref = np.array([(lo + hi) / 2.0 for lo, hi in self._bounds])
        else:
            ref = np.zeros(dim)

        threshold = result.threshold
        original = x.copy()
        n_evals = 0

        def _still_fails(candidate: np.ndarray) -> bool:
            nonlocal n_evals
            n_evals += 1
            return self._score(candidate.tolist(), kind="shrink") >= threshold

        if self._seed is not None:
            order = np.random.default_rng(
                self._seed ^ 0x5117111C
            ).permutation(dim)
        else:
            order = np.arange(dim)

        # Reserve evaluations to confirm the final witness.
        search_budget = max_evals - min(30, max(3, max_evals // 5))
        per_coord_budget = max(1, search_budget // max(int(dim), 1))

        for i in order:
            if n_evals >= search_budget:
                break
            lo_f, hi_f = 0.0, 1.0
            best_f = 0.0
            candidate = x.copy()
            for _ in range(per_coord_budget):
                if n_evals >= search_budget:
                    break
                mid_f = (lo_f + hi_f) / 2.0
                candidate[i] = x[i] + mid_f * (ref[i] - x[i])
                if _still_fails(candidate):
                    best_f = mid_f
                    lo_f = mid_f
                else:
                    hi_f = mid_f
                if hi_f - lo_f < 1e-6:
                    break
            x[i] = x[i] + best_f * (ref[i] - x[i])

        def _measured(candidate: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            xs = candidate.tolist()
            no_model = self._adapter is None and self._fhe_fn is None
            fallback = self._score(xs, kind="shrink_verify") if no_model else 0.0
            error = self._measure_divergence(xs, fallback)[0]
            if not no_model:
                self._emit("shrink_verify", xs, error=error)
            return error

        def _confirmed(candidate: np.ndarray) -> Optional[float]:
            # Up to 5 re-measurements; an exact repeat means a deterministic
            # backend. Noisy points must clear threshold by the observed spread.
            values = [_measured(candidate)]
            while values[-1] >= threshold and len(values) < 5:
                if n_evals >= max_evals:
                    return None
                values.append(_measured(candidate))
                if values[-1] == values[0]:
                    return values[0]
            lo, hi = min(values), max(values)
            return lo if lo - threshold >= hi - lo else None

        # A boundary point accepted on one noisy evaluation often fails on
        # replay; step back toward the original witness until it is confirmed.
        confirmed = _confirmed(x)
        base, step = x.copy(), 0.01
        while confirmed is None and step <= 1.0 and n_evals < max_evals:
            x = base + step * (original - base)
            confirmed = _confirmed(x)
            step *= 2
        if confirmed is None:
            x = original.copy()
            max_error = float(result.max_error)
        else:
            max_error = confirmed

        return ShrinkResult(
            original_input=original.tolist(),
            shrunk_input=x.tolist(),
            original_norm=float(np.linalg.norm(original - ref)),
            shrunk_norm=float(np.linalg.norm(x - ref)),
            max_error=max_error,
            threshold=threshold,
            n_evals=n_evals,
        )

    def _score(self, x: list[float], kind: str = "search") -> float:
        score = finite_score(self._fitness.score(x))
        self._emit(kind, x, score=score)
        return score

    def _emit(self, kind: str, x: list[float], **values: float) -> None:
        if self._on_evaluation is None:
            return
        event: dict[str, Any] = {"index": self._n_events, "kind": kind,
                                 "x": [float(v) for v in x]}
        event.update({k: float(v) for k, v in values.items()})
        self._n_events += 1
        self._on_evaluation(event)

    def _score_batch(self, xs: list[list[float]]) -> list[float]:
        """Score candidates in order, via ``batch_fhe_fn`` when one was given."""
        if self._batch_fhe_fn is None or not xs:
            return [self._score(x) for x in xs]
        try:
            outputs = list(self._batch_fhe_fn(xs))
        except Exception as exc:
            raise EvaluationError(f"batch model evaluation failed: {exc}") from exc
        if len(outputs) != len(xs):
            raise EvaluationError(
                f"batch_fhe_fn returned {len(outputs)} outputs for {len(xs)} inputs"
            )
        scores = [
            finite_score(self._fitness.score_with_output(x, out))
            for x, out in zip(xs, outputs)
        ]
        for x, score in zip(xs, scores):
            self._emit("search", x, score=score)
        return scores

    def _measure_divergence(
        self, x: list[float], fallback_score: float
    ) -> tuple[float, dict[str, float | str], Optional[bool]]:
        """Re-evaluate x in pure-divergence terms and capture noise state.

        Falls back to ``fallback_score`` (the fitness score already
        computed for ``x`` during search) when neither an adapter nor
        ``fhe_fn`` is available to independently recompute divergence
        -- i.e. pure custom-fitness mode. Without this, the previous
        behaviour called the (None) ``fhe_fn``, the exception handler
        swallowed the resulting TypeError, and the run silently
        reported ``max_error=0.0`` -- a false PASS.
        """
        noise_state: dict[str, float | str] = {}
        if self._adapter is None and self._fhe_fn is None:
            return finite_score(fallback_score), noise_state, None
        try:
            if self._adapter is not None:
                ct_in = self._adapter.encrypt(x)
                budget_before = self._adapter.get_noise_budget(ct_in)
                ct_out = self._adapter.run_fhe_program(ct_in)
                budget_after = self._adapter.get_noise_budget(ct_out)
                depth_used = self._adapter.get_mult_depth_used(ct_out)
                fhe_val = self._adapter.decrypt(ct_out)
                noise_state = {
                    "budget_before": float(budget_before),
                    "budget_after": float(budget_after),
                    "depth_used": float(depth_used),
                }
            else:
                assert self._fhe_fn is not None  # guaranteed by the early return above
                fhe_val = self._fhe_fn(x)

            plain_val = self._plaintext_fn(x)
            max_error = float(absolute_error(plain_val, fhe_val).max())
        except Exception as exc:
            raise EvaluationError(f"final evaluation failed: {exc}") from exc
        flip: Optional[bool] = None
        if (
            isinstance(self._fitness, MultiOutputFitness)
            and self._fitness.mode != MultiOutputMode.MAX_ABSOLUTE
        ):
            p, f = np.ravel(plain_val), np.ravel(fhe_val)
            flip = p.size > 1 and int(np.argmax(p)) != int(np.argmax(f))
        return max_error, noise_state, flip


def _cma_seed(seed: int) -> int:
    # pycma seeds from the clock when given 0; map into its nonzero range.
    return seed % (2**32 - 1) or 2**32 - 1


def _normalise_bounds(
    bounds: Optional[list[tuple[float, float]] | tuple[float, float]],
    d: int,
) -> Optional[list[tuple[float, float]]]:
    if bounds is None:
        return None
    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(
        bounds[0], (list, tuple)
    ):
        return [(float(bounds[0]), float(bounds[1]))] * d
    out = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(out) != d:
        raise ValueError(
            f"input_bounds length {len(out)} does not match input_dim {d}"
        )
    return out


def _to_array(value) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)
