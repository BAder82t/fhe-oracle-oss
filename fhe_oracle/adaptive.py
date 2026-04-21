# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Adaptive budget allocation for FHE Oracle.

Three independent behaviours combined into one decision helper:

1. **EARLY_STOP** -- if ``max_error >= threshold`` within the first
   ``early_stop_frac`` of the original budget, the witness is found
   and continuing wastes evaluations. Stop and return FAIL.

2. **AUTO_EXTEND** -- if at budget exhaustion the best error
   improved within the last ``climbing_window`` of evaluations, the
   trajectory is still climbing. Extend by ``extend_frac`` of the
   original budget, up to ``max_extensions`` times.

3. **STRATEGY_SWITCH** -- if CMA-ES's step size collapses below
   ``sigma_threshold * initial_sigma`` within the first
   ``switch_check_frac`` of the budget, the search is stuck on a
   plateau. Bail out and let the caller spend the remainder on
   uniform random sampling.

Caller-side integration uses the ask-tell loop:

.. code-block:: python

    ab = AdaptiveBudget(config, budget=500, threshold=0.01,
                        initial_sigma=0.5)
    for gen in range(...):
        candidates = es.ask()
        fitnesses = [f(c) for c in candidates]
        for c, s in zip(candidates, fitnesses):
            ab.record(eval_num, max_error_so_far, es.sigma)
        if ab.should_stop():
            break
        if ab.should_switch():
            break_to_random()
        es.tell(candidates, [-f for f in fitnesses])
        if ab.should_extend():
            budget += ab.extension_budget()
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveConfig:
    """Configuration knobs for :class:`AdaptiveBudget`."""

    early_stop: bool = True
    early_stop_frac: float = 0.2
    auto_extend: bool = True
    extend_frac: float = 0.5
    max_extensions: int = 2
    climbing_window: float = 0.2
    strategy_switch: bool = True
    sigma_threshold: float = 0.01
    switch_check_frac: float = 0.5


class AdaptiveBudget:
    """Decision helper for adaptive search behaviour.

    Records per-evaluation state via :meth:`record` and exposes
    boolean ``should_*`` predicates the caller queries each step.

    Parameters
    ----------
    config : AdaptiveConfig
        Behaviour toggles and thresholds.
    budget : int
        Original evaluation budget.
    threshold : float
        PASS/FAIL cut-off (also used by EARLY_STOP).
    initial_sigma : float
        CMA-ES initial step size (used by STRATEGY_SWITCH).
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        budget: int,
        threshold: float,
        initial_sigma: float,
    ) -> None:
        self.config = config
        self.original_budget = int(budget)
        self.current_budget = int(budget)
        self.threshold = float(threshold)
        self.initial_sigma = float(initial_sigma)
        self.extensions_used = 0
        self.best_error = 0.0
        self.best_error_at_eval = 0
        self.history: list[tuple[int, float]] = []
        self._current_sigma: float = float(initial_sigma)
        self._current_eval: int = 0
        self._switched: bool = False

    def record(self, eval_num: int, max_error: float, sigma: float) -> None:
        """Record an evaluation result."""
        self.history.append((int(eval_num), float(max_error)))
        if max_error > self.best_error:
            self.best_error = float(max_error)
            self.best_error_at_eval = int(eval_num)
        self._current_sigma = float(sigma)
        self._current_eval = int(eval_num)

    def should_stop(self) -> bool:
        """Early-stop predicate (definitive FAIL found in early window)."""
        if not self.config.early_stop:
            return False
        early_window = max(1, int(self.original_budget * self.config.early_stop_frac))
        if self._current_eval > early_window:
            return False
        return self.best_error >= self.threshold

    def should_switch(self) -> bool:
        """Strategy-switch predicate (CMA-ES sigma collapsed)."""
        if not self.config.strategy_switch or self._switched:
            return False
        switch_window = max(1, int(self.original_budget * self.config.switch_check_frac))
        if self._current_eval > switch_window:
            return False
        return self._current_sigma < self.config.sigma_threshold * self.initial_sigma

    def mark_switched(self) -> None:
        """Caller signals the strategy switch was taken (suppresses repeats)."""
        self._switched = True

    def should_extend(self) -> bool:
        """Extension predicate (still climbing at budget exhaustion)."""
        if not self.config.auto_extend:
            return False
        if self.extensions_used >= self.config.max_extensions:
            return False
        if self._current_eval < self.current_budget:
            return False
        window_start = int(self.current_budget * (1.0 - self.config.climbing_window))
        return self.best_error_at_eval >= window_start

    def extension_budget(self) -> int:
        """Mutate state to claim one extension; returns added budget."""
        ext = max(1, int(self.original_budget * self.config.extend_frac))
        self.extensions_used += 1
        self.current_budget += ext
        return ext

    @property
    def stop_reason(self) -> str:
        if self.should_stop():
            return "early_stop_fail_found"
        if self.should_switch():
            return "stall_switch_to_random"
        return "budget_exhausted"
