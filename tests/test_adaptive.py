# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for adaptive budget allocation (fhe_oracle.adaptive)."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.adaptive import AdaptiveBudget, AdaptiveConfig


# --- AdaptiveBudget unit tests ----------------------------------------


def test_should_stop_fires_on_early_fail():
    cfg = AdaptiveConfig(early_stop=True, early_stop_frac=0.2)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=10, max_error=1.0, sigma=0.9)
    assert ab.should_stop()


def test_should_stop_does_not_fire_after_window():
    cfg = AdaptiveConfig(early_stop=True, early_stop_frac=0.2)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=400, max_error=1.0, sigma=0.9)  # past 20% window
    assert not ab.should_stop()


def test_should_stop_disabled_by_config():
    cfg = AdaptiveConfig(early_stop=False)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=10, max_error=1.0, sigma=0.9)
    assert not ab.should_stop()


def test_should_switch_on_sigma_collapse():
    cfg = AdaptiveConfig(strategy_switch=True, sigma_threshold=0.01)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=50, max_error=0.001, sigma=0.005)
    assert ab.should_switch()


def test_should_switch_does_not_fire_when_sigma_healthy():
    cfg = AdaptiveConfig(strategy_switch=True, sigma_threshold=0.01)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=50, max_error=0.001, sigma=0.5)
    assert not ab.should_switch()


def test_should_switch_only_fires_once():
    cfg = AdaptiveConfig(strategy_switch=True, sigma_threshold=0.01)
    ab = AdaptiveBudget(cfg, budget=500, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=50, max_error=0.001, sigma=0.005)
    assert ab.should_switch()
    ab.mark_switched()
    assert not ab.should_switch()


def test_should_extend_when_climbing_at_exhaustion():
    cfg = AdaptiveConfig(auto_extend=True, climbing_window=0.2,
                         extend_frac=0.5, max_extensions=2)
    ab = AdaptiveBudget(cfg, budget=100, threshold=0.01, initial_sigma=1.0)
    # Best improved at eval 95 of 100 (within last 20%).
    ab.record(eval_num=95, max_error=0.5, sigma=0.5)
    ab.record(eval_num=100, max_error=0.5, sigma=0.5)
    assert ab.should_extend()


def test_extension_budget_increments_state():
    cfg = AdaptiveConfig(auto_extend=True, extend_frac=0.5, max_extensions=2)
    ab = AdaptiveBudget(cfg, budget=100, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=100, max_error=1.0, sigma=0.1)
    ab.best_error_at_eval = 95
    assert ab.should_extend()
    added = ab.extension_budget()
    assert added == 50
    assert ab.current_budget == 150
    assert ab.extensions_used == 1


def test_extension_capped():
    cfg = AdaptiveConfig(auto_extend=True, extend_frac=0.5, max_extensions=1)
    ab = AdaptiveBudget(cfg, budget=100, threshold=0.01, initial_sigma=1.0)
    ab.record(eval_num=100, max_error=1.0, sigma=0.1)
    ab.best_error_at_eval = 95
    ab.extension_budget()
    ab.record(eval_num=150, max_error=1.0, sigma=0.1)
    ab.best_error_at_eval = 145
    assert not ab.should_extend()


# --- FHEOracle integration -------------------------------------------


def test_early_stop_in_oracle_uses_few_evals():
    """With adaptive on and a definitive early FAIL, the oracle stops
    well before exhausting the nominal budget."""
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: 100.0,
        input_dim=4,
        input_bounds=[(-1.0, 1.0)] * 4,
        adaptive=True,
        seed=1,
    )
    result = oracle.run(n_trials=500, threshold=0.01)
    assert result.adaptive_stop_reason == "early_stop_fail_found"
    assert result.n_trials < 200, f"Used {result.n_trials} trials, expected early stop"


def test_adaptive_default_off_preserves_existing_path():
    """With adaptive=False the result has no adaptive metadata signal."""
    oracle = FHEOracle(
        plaintext_fn=lambda x: 0.0,
        fhe_fn=lambda x: float(np.sum(np.asarray(x) ** 2)) * 0.001,
        input_dim=4,
        input_bounds=[(-3.0, 3.0)] * 4,
        seed=1,
    )
    result = oracle.run(n_trials=100, threshold=1.0)
    assert result.adaptive_stop_reason is None
    assert result.adaptive_extensions_used == 0
    assert result.diversity_injections == 0


def test_adaptive_no_regression_on_standard():
    """Adaptive mode should match or beat vanilla on smooth landscapes
    when the threshold is unreachable (so EARLY_STOP cannot fire)."""
    rng = np.random.RandomState(42)
    w = rng.randn(8)

    def plain(x):
        z = float(w @ np.asarray(x))
        return 1.0 / (1.0 + np.exp(-z))

    def fhe(x):
        z = float(w @ np.asarray(x))
        return 0.5 + z / 4.0 - z ** 3 / 48.0

    # Threshold = 1e9 so the divergence never crosses it -> EARLY_STOP
    # never fires and the comparison measures search quality, not
    # the adaptive stop condition.
    vanilla, adaptive = [], []
    for seed in range(1, 6):
        v = FHEOracle(plaintext_fn=plain, fhe_fn=fhe,
                      input_dim=8, input_bounds=[(-3.0, 3.0)] * 8,
                      seed=seed).run(n_trials=200, threshold=1e9)
        a = FHEOracle(plaintext_fn=plain, fhe_fn=fhe,
                      input_dim=8, input_bounds=[(-3.0, 3.0)] * 8,
                      seed=seed, adaptive=True).run(n_trials=200, threshold=1e9)
        vanilla.append(v.max_error)
        adaptive.append(a.max_error)
    assert np.mean(adaptive) >= np.mean(vanilla) * 0.9
