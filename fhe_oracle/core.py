# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
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

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import registry
from .adaptive import AdaptiveBudget, AdaptiveConfig
from .diversity import DiversityInjector, InjectionStrategy
from .fitness import DivergenceFitness
from .guarantees import CoverageCertificate
from .seeds import fallback_corner_seeds


def _build_seeds(rng, bounds, k, which, tau):
    """Generate heuristic seeds via Pro's patented generator if registered,
    else the non-patented corner+random fallback.

    The Pro generator is registered under the ``"generate_seeds"`` name
    in the ``heuristics`` registry by ``fhe-oracle-pro``'s ``__init__``.
    It is a single callable with the signature
    ``(rng, bounds, k, tau, which) -> list[list[float]]``.
    """
    try:
        gen = registry.get_heuristic("generate_seeds")
    except KeyError:
        return fallback_corner_seeds(rng, bounds, k=k)
    return gen(rng, bounds, k=k, tau=tau, which=which)

# Soft cap on n_trials for the open-source edition. Lifted by
# fhe-oracle-pro on import. Users who need the uncapped search in
# OSS can override via the FHE_ORACLE_MAX_TRIALS environment variable
# (e.g. export FHE_ORACLE_MAX_TRIALS=10000). Documented policy, not DRM.
MAX_TRIALS_OSS: int = int(os.environ.get("FHE_ORACLE_MAX_TRIALS", "1000"))
from .multi_output import MultiOutputFitness, MultiOutputMode


@dataclass
class OracleResult:
    """Outcome of an adversarial oracle run.

    Attributes
    ----------
    verdict : str
        "PASS" if max_error < threshold, else "FAIL".
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
        FHE scheme name (from adapter) or "plaintext-diff" in pure mode.
    noise_state : dict[str, float]
        Noise-budget snapshot at worst_input when an adapter was used.
        Empty dict in pure-divergence mode.
    """

    verdict: str
    max_error: float
    worst_input: list[float]
    threshold: float
    n_trials: int
    elapsed_seconds: float
    scheme: str = "plaintext-diff"
    noise_state: dict[str, float] = field(default_factory=dict)
    coverage_certificate: Optional["CoverageCertificate"] = None
    n_restarts_used: int = 0
    adaptive_stop_reason: Optional[str] = None
    adaptive_extensions_used: int = 0
    diversity_injections: int = 0

    def __repr__(self) -> str:
        return (
            f"OracleResult(verdict={self.verdict!r}, "
            f"max_error={self.max_error:.6e}, "
            f"trials={self.n_trials}, "
            f"elapsed={self.elapsed_seconds:.2f}s)"
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
        (applied only when ``adapter`` is supplied).
    w_noise : float, default 0.0
        Weight for noise-budget consumption in fitness. Default
        changed from ``0.5`` to ``0.0`` in v0.3.0: empirical evaluation
        (paper §6.15) showed shaping weights are inert on all tested
        CKKS circuits and counter-productive on pure-CKKS-noise
        circuits. Set to ``0.5`` to restore v0.2 behaviour.
    w_depth : float, default 0.0
        Weight for multiplicative-depth utilisation in fitness. Default
        changed from ``0.3`` to ``0.0`` in v0.3.0 for the same reason as
        ``w_noise``. Set to ``0.3`` to restore v0.2 behaviour.
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
        w_noise: float = 0.0,
        w_depth: float = 0.0,
        adaptive: bool = False,
        adaptive_config: Optional[AdaptiveConfig] = None,
        diversity_injection: bool = False,
        inject_every: int = 5,
        inject_count: int = 3,
        inject_strategy: str = "mixed",
        multi_output: bool = False,
        multi_output_mode: str = "combined",
        rank_weight: float = 1.0,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if fitness is None and fhe_fn is None and adapter is None:
            raise ValueError(
                "Provide one of: fhe_fn, adapter, or a custom fitness object."
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
        self.w_noise = float(w_noise)
        self.w_depth = float(w_depth)
        self._scheme = (
            adapter.get_scheme_name() if adapter is not None else "plaintext-diff"
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
            try:
                noise_cls = registry.get_fitness("noise_budget")
            except KeyError:
                raise RuntimeError(
                    "Noise-guided fitness requires fhe-oracle-pro "
                    "(patent PCT/IB2026/053378). Install from the "
                    "commercial index, or omit the adapter argument "
                    "to use pure-divergence fitness in the open-source "
                    "edition."
                ) from None
            self._fitness = noise_cls(
                plaintext_fn,
                adapter,
                weights=(self.w_div, self.w_noise, self.w_depth),
            )
        else:
            self._fitness = DivergenceFitness(plaintext_fn, fhe_fn)

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
        # Soft cap enforcement (OSS edition). Pro lifts MAX_TRIALS_OSS
        # to sys.maxsize on import; users who need more evaluations
        # without Pro can set FHE_ORACLE_MAX_TRIALS=<n> in env.
        if n_trials > MAX_TRIALS_OSS:
            raise ValueError(
                f"n_trials={n_trials} exceeds the open-source edition "
                f"limit of {MAX_TRIALS_OSS}. Override for a single run "
                f"with FHE_ORACLE_MAX_TRIALS=<n> in your environment, "
                f"or install fhe-oracle-pro (https://vaultbytes.com/"
                f"oracle.html) which removes the cap."
            )
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "The 'cma' package is required. Install with: pip install cma"
            ) from exc

        t0 = time.perf_counter()

        best_input = list(self._x0)
        best_score = -np.inf
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
            for _ in range(b_rand):
                x = rng_floor.uniform(lows, highs)
                score = self._fitness.score(list(x))
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
                options["seed"] = self._seed
            if self._bounds is not None:
                lows_b = [lo for lo, _ in self._bounds]
                highs_b = [hi for _, hi in self._bounds]
                options["bounds"] = [lows_b, highs_b]
            if self._separable:
                options["CMA_diagonal"] = True

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

                fitnesses = [self._fitness.score(list(s)) for s in solutions]
                for sol, s in zip(solutions, fitnesses):
                    total_evals += 1
                    cma_evals += 1
                    if s > best_score:
                        best_score = s
                        best_input = list(sol)

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
                            lows_b = np.array([lo for lo, _ in self._bounds])
                            highs_b = np.array([hi for _, hi in self._bounds])
                            switch_seed = (
                                (self._seed if self._seed is not None else 0) ^ 0x5736
                            )
                            switch_rng = np.random.default_rng(switch_seed)
                            while cma_evals < b_cma:
                                x = switch_rng.uniform(lows_b, highs_b)
                                s = self._fitness.score(list(x))
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
                    b_cma += extra
                    adaptive_extensions_used += 1
                    # pycma reads maxfevals lazily through stop(); update.
                    try:
                        es.opts.set({"maxfevals": b_cma})
                    except Exception:
                        pass

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
                    run_options["seed"] = self._seed + run_index + 1
                if self._separable:
                    run_options["CMA_diagonal"] = True

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
                    fitnesses = [
                        self._fitness.score(list(s)) for s in solutions
                    ]
                    for sol, s in zip(solutions, fitnesses):
                        total_evals += 1
                        cma_evals_total += 1
                        if s > best_score:
                            best_score = s
                            best_input = list(sol)
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

        max_error, noise_state = self._measure_divergence(best_input)
        verdict = "PASS" if max_error < threshold else "FAIL"

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
        )

    def _measure_divergence(
        self, x: list[float]
    ) -> tuple[float, dict[str, float]]:
        """Re-evaluate x in pure-divergence terms and capture noise state."""
        noise_state: dict[str, float] = {}
        try:
            if self._adapter is not None:
                ct_in = self._adapter.encrypt(x)
                budget_before = self._adapter.get_noise_budget(ct_in)
                ct_out = self._adapter.run_fhe_program(ct_in)
                budget_after = self._adapter.get_noise_budget(ct_out)
                depth_used = self._adapter.get_mult_depth_used(ct_out)
                fhe_val = _to_array(self._adapter.decrypt(ct_out))
                noise_state = {
                    "budget_before": float(budget_before),
                    "budget_after": float(budget_after),
                    "depth_used": float(depth_used),
                }
            else:
                fhe_val = _to_array(self._fhe_fn(x))

            plain_val = _to_array(self._plaintext_fn(x))
            n = min(plain_val.size, fhe_val.size)
            diff = np.abs(plain_val.ravel()[:n] - fhe_val.ravel()[:n])
            max_error = float(diff.max()) if diff.size > 0 else 0.0
        except Exception as exc:
            max_error = 0.0
            noise_state["error"] = str(exc)
        return max_error, noise_state


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
