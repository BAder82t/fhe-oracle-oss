# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AutoOracle whole-run budget accounting: every evaluation counts toward n_trials."""

from __future__ import annotations

import numpy as np
import pytest

import fhe_oracle.core as core_module
from fhe_oracle.autoconfig import (
    _RESERVE,
    _RESERVE_1D_PREACT,
    AutoOracle,
    classify_landscape,
)


def _counted(fn):
    def wrapped(x):
        wrapped.calls += 1
        return fn(x)
    wrapped.calls = 0
    return wrapped


def _count_both(plain, fhe):
    calls = {"plain": 0, "fhe": 0}

    def p(x):
        calls["plain"] += 1
        return plain(x)

    def f(x):
        calls["fhe"] += 1
        return fhe(x)

    return p, f, calls


def _taylor_gap(x):
    z = float(np.dot(np.linspace(-1.0, 1.0, len(x)), x)) + 0.1
    return abs(1.0 / (1.0 + np.exp(-z)) - (0.5 + z / 4.0 - z ** 3 / 48.0))


def _bump(x):
    c = np.linspace(-1.2, 1.2, len(x))
    return float(np.exp(-np.sum((np.asarray(x) - c) ** 2) / 4.5))


def _rank2_sin(d):
    W = np.random.default_rng(3).standard_normal((2, d))
    return lambda x: 0.01 * float(np.sum(np.sin(W @ np.asarray(x))))


_LANDSCAPE_FACTORIES = {
    "vertex": lambda d: _taylor_gap,
    "bump": lambda d: _bump,
    "saturated": lambda d: (lambda x: 2.0),
    "shell": lambda d: (lambda x: 5.0 if abs(x[0]) > 2.7 else 0.0),
    "cliff": lambda d: (lambda x: 100.0 if np.max(np.abs(x)) > 2.9 else 0.01),
    "rank2": _rank2_sin,
}


@pytest.mark.parametrize("d", [2, 8, 16, 32])
@pytest.mark.parametrize("landscape", sorted(_LANDSCAPE_FACTORIES))
@pytest.mark.parametrize("extra", [0, 1, 7, 120])
def test_total_model_evaluations_never_exceed_budget(d, landscape, extra):
    # The final re-measurement is charged inside n_trials, so the bound is n_trials itself.
    n_probes = 20
    n_trials = n_probes + _RESERVE + extra
    plain, fhe, calls = _count_both(lambda x: 0.0, _LANDSCAPE_FACTORIES[landscape](d))
    result = AutoOracle(plain, fhe, [(-3.0, 3.0)] * d, n_probes=n_probes).run(
        n_trials=n_trials, seed=1)
    assert calls["fhe"] <= n_trials
    assert calls["plain"] <= n_trials
    assert result.n_trials <= n_trials


@pytest.mark.parametrize("n_probes", [1, 20, 50])
def test_minimum_budget_rule(n_probes):
    oracle = AutoOracle(lambda x: 0.0, _bump, [(-3.0, 3.0)] * 4, n_probes=n_probes)
    with pytest.raises(ValueError):
        oracle.run(n_trials=n_probes + _RESERVE - 1)
    fhe = _counted(_bump)
    AutoOracle(lambda x: 0.0, fhe, [(-3.0, 3.0)] * 4, n_probes=n_probes).run(
        n_trials=n_probes + _RESERVE, seed=1)
    assert fhe.calls <= n_probes + _RESERVE


def test_classify_landscape_max_evals_caps_extra_stages():
    # CV~0.37, max=3.5x median, no plateau test fires: the second pass runs uncapped.
    pool = np.array(([1.0] * 30 + [1.6] * 12 + [2.0] * 7 + [3.5]) * 2)

    def run(max_evals):
        fhe = _counted(lambda x: float(pool[(fhe.calls - 1) % pool.size]))
        res = classify_landscape(lambda x: 0.0, fhe, [(-3.0, 3.0)] * 4,
                                 n_probes=50, seed=0, max_evals=max_evals)
        return res, fhe.calls

    capped, calls = run(60)
    assert calls <= 60
    assert capped.n_evals == calls
    uncapped, calls = run(None)
    assert uncapped.n_evals == calls >= 100
    assert uncapped.best_divergence == 3.5


def test_inner_search_gets_the_remaining_budget(monkeypatch):
    fhe = _counted(_bump)
    seen = {}
    real_run = core_module.FHEOracle.run

    def spy_run(self, n_trials=500, **kwargs):
        seen["calls_at_entry"] = fhe.calls
        seen["n_trials"] = n_trials
        return real_run(self, n_trials=n_trials, **kwargs)

    monkeypatch.setattr(core_module.FHEOracle, "run", spy_run)
    ao = AutoOracle(lambda x: 0.0, fhe, [(-3.0, 3.0)] * 8)
    ao.run(n_trials=200, seed=1)
    assert seen["calls_at_entry"] >= ao.probe_result.n_evals
    # Held back only for re-measurements: the inner run's, plus at most one more.
    left = 200 - seen["calls_at_entry"]
    assert left - (_RESERVE - 1) <= seen["n_trials"] <= left - 1
    assert 200 - (_RESERVE - 2) <= fhe.calls <= 200


def test_structure_diagnostic_is_charged_when_it_runs():
    d = 16
    plain, fhe, calls = _count_both(lambda x: 0.0, _rank2_sin(d))
    ao = AutoOracle(plain, fhe, [(-1.0, 1.0)] * d, n_probes=30)
    result = ao.run(n_trials=6600, seed=2)
    assert result.regime == "low_rank_structure"
    assert ao.probe_result.n_evals >= 100 * 2 * d
    assert calls["fhe"] <= 6600


def test_structure_diagnostic_skipped_when_unaffordable():
    d = 16
    plain, fhe, calls = _count_both(lambda x: 0.0, _rank2_sin(d))
    ao = AutoOracle(plain, fhe, [(-1.0, 1.0)] * d, n_probes=30)
    result = ao.run(n_trials=500, seed=2)
    assert result.regime != "low_rank_structure"
    assert calls["fhe"] <= 500


@pytest.mark.parametrize("k", [1, 2])
def test_preactivation_dispatch_within_budget_at_minimum(k):
    d, n_probes = 8, 30
    reserve = _RESERVE_1D_PREACT if k == 1 else _RESERVE
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(k, d)), rng.normal(size=k)

    def cubic(x):
        return float(np.max(np.abs(W @ np.asarray(x) + b)) ** 3 / 10.0)

    bounds = [(-3.0, 3.0)] * d
    for n_trials in (n_probes + reserve, n_probes + reserve + 3, 80):
        plain, fhe, calls = _count_both(lambda x: 0.0, cubic)
        result = AutoOracle(plain, fhe, bounds, W=W, b=b, n_probes=n_probes).run(
            n_trials=n_trials, seed=2)
        assert result.strategy_used == "preactivation"
        assert calls["fhe"] <= n_trials
        assert calls["plain"] <= n_trials
    with pytest.raises(ValueError):
        AutoOracle(lambda x: 0.0, cubic, bounds, W=W, b=b, n_probes=n_probes).run(
            n_trials=n_probes + reserve - 1)


def _remeasure_spy(monkeypatch, calls):
    # FHE calls made inside re-measurements, the only calls result.n_trials excludes.
    import fhe_oracle.preactivation as pre_module

    spent = {"n": 0}
    for cls, name in ((core_module.FHEOracle, "_measure_divergence"),
                      (pre_module.PreactivationOracle, "measure_divergence_at")):
        real = getattr(cls, name)

        def spy(self, *args, _real=real, **kwargs):
            before = calls["fhe"]
            try:
                return _real(self, *args, **kwargs)
            finally:
                spent["n"] += calls["fhe"] - before

        monkeypatch.setattr(cls, name, spy)
    return spent


def _plateau_then_cliff():
    vals = np.concatenate([np.full(47, 0.098), [0.30, 0.34, 0.36]])
    np.random.RandomState(7).shuffle(vals)
    state = {"i": 0}

    def fhe(x):
        state["i"] += 1
        return float(vals[(state["i"] - 1) % vals.size])

    return fhe


def _cubic_preact(k, d=8):
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(k, d)), rng.normal(size=k)
    return (lambda x: float(np.max(np.abs(W @ np.asarray(x) + b)) ** 3 / 10.0)), W, b


_REGIME_CASES = {
    # name: (fhe factory, bounds, n_probes, n_trials, extra AutoOracle kwargs, strategy)
    "saturated": (lambda: (lambda x: 2.0), [(-1.0, 1.0)] * 3, 20, 60, {}, "random_only"),
    "plateau": (_plateau_then_cliff, [(-3.0, 3.0)] * 5, 50, 100, {}, "warm_start"),
    "shell": (lambda: (lambda x: 5.0 if abs(x[0]) > 15.0 else 0.0), [(-20.0, 20.0)] * 2,
              30, 80, {}, "robust_cma_es"),
    "low_rank": (lambda: _rank2_sin(16), [(-1.0, 1.0)] * 16, 30, 6600, {}, "separable_cma_es"),
    "standard": (lambda: _bump, [(-3.0, 3.0)] * 8, 50, 200, {}, "cma_es"),
    "preact_k1": (lambda: _cubic_preact(1)[0], [(-3.0, 3.0)] * 8, 30, 80,
                  {"W": _cubic_preact(1)[1], "b": _cubic_preact(1)[2]}, "preactivation"),
    "preact_k2": (lambda: _cubic_preact(2)[0], [(-3.0, 3.0)] * 8, 30, 80,
                  {"W": _cubic_preact(2)[1], "b": _cubic_preact(2)[2]}, "preactivation"),
}


@pytest.mark.parametrize("case", sorted(_REGIME_CASES))
def test_result_n_trials_counts_every_evaluation_but_remeasurements(monkeypatch, case):
    make_fhe, bounds, n_probes, n_trials, kwargs, strategy = _REGIME_CASES[case]
    plain, fhe, calls = _count_both(lambda x: 0.0, make_fhe())
    spent = _remeasure_spy(monkeypatch, calls)
    result = AutoOracle(plain, fhe, bounds, n_probes=n_probes, **kwargs).run(
        n_trials=n_trials, seed=2 if case == "low_rank" else 1)
    assert result.strategy_used == strategy
    assert 1 <= spent["n"] <= 2
    assert calls["fhe"] == result.n_trials + spent["n"]
    assert calls["fhe"] <= n_trials


def test_check_report_counts_probe_evaluations(monkeypatch):
    from fhe_oracle.check import check

    plain, fhe, calls = _count_both(lambda x: 0.0, _bump)
    spent = _remeasure_spy(monkeypatch, calls)
    out = check(plain, fhe, [(-3.0, 3.0)] * 8, n_trials=200, seed=1, shrink=False)
    n = out.oracle_result.n_trials
    assert calls["fhe"] == n + spent["n"]
    assert n >= 200 - 2
    assert f"**Trials:** {n}" in out.report


def test_autooracle_rejects_budget_without_room_to_search():
    oracle = AutoOracle(lambda x: 0.0, lambda x: 1.0, [(-1.0, 1.0)] * 2, n_probes=50)
    with pytest.raises(ValueError):
        oracle.run(n_trials=51)
