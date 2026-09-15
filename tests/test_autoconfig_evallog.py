# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AutoOracle evaluation log: every counted evaluation reaches on_evaluation, in order."""

from __future__ import annotations

import numpy as np
import pytest

import fhe_oracle.core as core_module
import fhe_oracle.preactivation as pre_module
from fhe_oracle import JsonlEvaluationLog, read_log, replay
from fhe_oracle.autoconfig import AutoOracle, classify_landscape
from fhe_oracle.check import check

_RUN_KINDS = {"probe", "structure", "boundary", "search", "remeasure"}


def _counted(fn):
    def wrapped(x):
        wrapped.calls += 1
        return fn(x)
    wrapped.calls = 0
    return wrapped


def _remeasure_calls(monkeypatch, fhe):
    # FHE calls made inside re-measurements (core and preactivation).
    spent = {"n": 0}
    for cls, name in ((core_module.FHEOracle, "_measure_divergence"),
                      (pre_module.PreactivationOracle, "measure_divergence_at")):
        real = getattr(cls, name)

        def spy(self, *args, _real=real, **kwargs):
            before = fhe.calls
            try:
                return _real(self, *args, **kwargs)
            finally:
                spent["n"] += fhe.calls - before

        monkeypatch.setattr(cls, name, spy)
    return spent


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3(z):
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _bump(x):
    c = np.linspace(-1.2, 1.2, len(x))
    return float(np.exp(-np.sum((np.asarray(x) - c) ** 2) / 4.5))


def _rank2_sin(d):
    W = np.random.default_rng(3).standard_normal((2, d))
    return lambda x: 0.01 * float(np.sum(np.sin(W @ np.asarray(x))))


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


def _lr_vertex(d=8, seed=5):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, d)
    return (lambda x: float(_sigmoid(w @ np.asarray(x) + 0.1)),
            lambda x: float(_taylor3(w @ np.asarray(x) + 0.1)))


def _zero(x):
    return 0.0


def _case(name):
    # (plain, fhe, bounds, n_probes, n_trials, AutoOracle kwargs, strategy or None, seed)
    if name == "saturated":
        return _zero, (lambda x: 2.0), [(-1.0, 1.0)] * 3, 20, 60, {}, "random_only", 1
    if name == "plateau":
        return _zero, _plateau_then_cliff(), [(-3.0, 3.0)] * 5, 50, 100, {}, "warm_start", 1
    if name == "shell":
        return (_zero, (lambda x: 5.0 if abs(x[0]) > 15.0 else 0.0), [(-20.0, 20.0)] * 2,
                30, 80, {}, "robust_cma_es", 1)
    if name == "low_rank":
        return _zero, _rank2_sin(16), [(-1.0, 1.0)] * 16, 30, 6600, {}, "separable_cma_es", 2
    if name == "standard":
        return _zero, _bump, [(-3.0, 3.0)] * 8, 50, 200, {}, "cma_es", 1
    if name == "vertex":
        plain, fhe = _lr_vertex()
        return plain, fhe, [(-3.0, 3.0)] * 8, 50, 200, {}, None, 2
    k = 1 if name == "preact_k1" else 2
    fhe, W, b = _cubic_preact(k)
    return _zero, fhe, [(-3.0, 3.0)] * 8, 30, 80, {"W": W, "b": b}, "preactivation", 1


_CASES = ["saturated", "plateau", "shell", "low_rank", "standard", "vertex",
          "preact_k1", "preact_k2"]


@pytest.mark.parametrize("name", _CASES)
def test_events_cover_every_counted_evaluation(monkeypatch, name):
    plain, fhe_fn, bounds, n_probes, n_trials, kwargs, strategy, seed = _case(name)
    fhe = _counted(fhe_fn)
    spent = _remeasure_calls(monkeypatch, fhe)
    events = []
    ao = AutoOracle(plain, fhe, bounds, n_probes=n_probes, on_evaluation=events.append,
                    **kwargs)
    result = ao.run(n_trials=n_trials, seed=seed)
    if strategy is not None:
        assert result.strategy_used == strategy
    kinds = [e["kind"] for e in events]
    assert set(kinds) <= _RUN_KINDS
    assert [e["index"] for e in events] == list(range(len(events)))
    assert len(events) == fhe.calls
    assert len(kinds) - kinds.count("remeasure") == result.n_trials
    assert kinds.count("remeasure") == spent["n"]
    last = [e for e in events if e["kind"] == "remeasure"][-1]
    assert last["x"] == [float(v) for v in result.worst_input]
    assert kinds.count("probe") + kinds.count("structure") == ao.probe_result.n_evals
    assert ("structure" in kinds) == (name == "low_rank")
    assert ("boundary" in kinds) == (strategy != "preactivation")
    assert all(("error" in e) == (e["kind"] == "remeasure") for e in events)
    assert all(("score" in e) != ("error" in e) for e in events)
    if name != "plateau":  # the plateau mock is stateful, so it cannot be replayed
        rows = replay(events, plain, fhe_fn, kinds=tuple(_RUN_KINDS))
        assert len(rows) == len(events)
        assert all(r["logged"] == r["replayed"] for r in rows)


def _noisy_vertex(d=6, threshold=0.01):
    # True error 0.9*threshold at every vertex; the first visit to a vertex reads 1.1*threshold.
    visits: dict[tuple, int] = {}

    def fhe(x):
        xa = np.asarray(x, dtype=float)
        err = 0.9 * threshold * (np.mean(np.abs(xa)) / 3.0) ** 4
        if np.all(np.abs(xa) == 3.0):
            key = tuple(xa.tolist())
            visits[key] = visits.get(key, 0) + 1
            if visits[key] == 1:
                err = 1.1 * threshold
        return err

    return fhe, [(-3.0, 3.0)] * d


def test_presearch_witness_remeasurement_is_logged_last():
    fhe, bounds = _noisy_vertex()
    events = []
    result = AutoOracle(_zero, fhe, bounds, on_evaluation=events.append).run(
        n_trials=200, seed=1, threshold=0.01)
    remeasures = [e for e in events if e["kind"] == "remeasure"]
    assert len(remeasures) == 2
    assert events[-1] is remeasures[-1]
    assert events[-1]["x"] == result.worst_input
    assert events[-1]["error"] == result.remeasured_error
    assert len(events) == result.n_trials + 2


def test_jsonl_log_end_to_end(tmp_path, monkeypatch):
    fhe = _counted(_bump)
    spent = _remeasure_calls(monkeypatch, fhe)
    path = tmp_path / "auto.jsonl"
    with JsonlEvaluationLog(path) as log:
        result = AutoOracle(_zero, fhe, [(-3.0, 3.0)] * 8, on_evaluation=log).run(
            n_trials=200, seed=1)
    events = read_log(path)
    assert len(path.read_text().splitlines()) == result.n_trials + spent["n"] == fhe.calls
    assert [e["index"] for e in events] == list(range(len(events)))
    assert events[-1]["kind"] == "remeasure"
    assert all(r["logged"] == r["replayed"] for r in replay(events, _zero, _bump))


def test_jsonl_log_via_check_without_shrink(tmp_path, monkeypatch):
    plain, fhe_fn = _lr_vertex()
    fhe = _counted(fhe_fn)
    spent = _remeasure_calls(monkeypatch, fhe)
    path = tmp_path / "check.jsonl"
    with JsonlEvaluationLog(path) as log:
        out = check(plain, fhe, [(-3.0, 3.0)] * 8, n_trials=200, seed=2, shrink=False,
                    on_evaluation=log)
    lines = path.read_text().splitlines()
    assert len(lines) == out.oracle_result.n_trials + spent["n"] == fhe.calls
    events = read_log(path)
    assert [e["index"] for e in events] == list(range(len(events)))


def test_jsonl_log_via_check_with_shrink(tmp_path):
    plain, fhe_fn = _lr_vertex()
    fhe = _counted(fhe_fn)
    path = tmp_path / "check_shrink.jsonl"
    with JsonlEvaluationLog(path) as log:
        out = check(plain, fhe, [(-3.0, 3.0)] * 8, n_trials=200, seed=2,
                    shrink_max_evals=60, on_evaluation=log)
    assert out.shrink_result is not None
    events = read_log(path)
    kinds = [e["kind"] for e in events]
    run_len = out.oracle_result.n_trials + kinds.count("remeasure")
    assert set(kinds[:run_len]) <= _RUN_KINDS
    assert set(kinds[run_len:]) <= {"shrink", "shrink_verify"}
    assert len(kinds) - run_len == out.shrink_result.n_evals
    assert [e["index"] for e in events] == list(range(len(events)))
    assert len(events) == fhe.calls


@pytest.mark.parametrize("name", ["standard", "vertex", "preact_k2"])
def test_same_seed_gives_identical_event_list(name):
    def run():
        plain, fhe, bounds, n_probes, n_trials, kwargs, _, seed = _case(name)
        events = []
        AutoOracle(plain, fhe, bounds, n_probes=n_probes, on_evaluation=events.append,
                   **kwargs).run(n_trials=n_trials, seed=seed)
        return events

    first, second = run(), run()
    assert first and first == second


def test_classify_landscape_emits_probe_events():
    fhe = _counted(_bump)
    events = []
    res = classify_landscape(_zero, fhe, [(-3.0, 3.0)] * 4, n_probes=30, seed=0,
                             on_evaluation=events.append)
    assert len(events) == res.n_evals == fhe.calls
    assert [e["index"] for e in events] == list(range(len(events)))
    assert {e["kind"] for e in events} == {"probe"}
    assert events[0]["x"] == [float(v) for v in res.probe_points[0]]
    assert events[0]["score"] == res.probe_divergences[0]
