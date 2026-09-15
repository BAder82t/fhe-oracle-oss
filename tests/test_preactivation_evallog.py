# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PreactivationOracle evaluation log: every model evaluation reaches on_evaluation, in x-space."""

from __future__ import annotations

import numpy as np
import pytest

import fhe_oracle.core as core_module
from fhe_oracle import JsonlEvaluationLog, read_log, replay
from fhe_oracle.fitness import EvaluationError
from fhe_oracle.preactivation import PreactivationOracle

D = 6
BOUNDS = [(-2.0, 1.5)] * D  # asymmetric, so the pseudoinverse preimage is often clipped


class _Counted:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


def _oracle(k, **kwargs):
    rng = np.random.default_rng(4)
    W, b = rng.normal(size=(k, D)), rng.normal(size=k)

    def plain(x):
        return 1.0 / (1.0 + np.exp(-(W @ np.asarray(x) + b)))

    def fhe(x):
        z = W @ np.asarray(x) + b
        return 0.5 + z / 4.0 - z ** 3 / 48.0

    plain_c, fhe_c = _Counted(plain), _Counted(fhe)
    pre = PreactivationOracle(W=W, b=b, plaintext_fn=plain_c, fhe_fn=fhe_c,
                              input_bounds=BOUNDS, **kwargs)
    return pre, plain_c, fhe_c


@pytest.mark.parametrize("k", [1, 2])
def test_events_cover_every_model_evaluation(k):
    events = []
    pre, plain, fhe = _oracle(k, on_evaluation=events.append)
    results = pre.run(budget=40, seeds=[1, 2])
    assert len(events) == plain.calls == fhe.calls
    assert [e["index"] for e in events] == list(range(len(events)))
    expected = [kind for r in results for kind in ["search"] * r.n_trials + ["remeasure"]]
    assert [e["kind"] for e in events] == expected
    lo, hi = np.array(BOUNDS).T
    for e in events:
        x = np.asarray(e["x"])
        assert x.shape == (D,) and np.all((lo <= x) & (x <= hi))
        assert ("error" in e) == (e["kind"] == "remeasure")
        assert ("score" in e) != ("error" in e)
    remeasures = [e for e in events if e["kind"] == "remeasure"]
    assert [e["x"] for e in remeasures] == [r.x for r in results]
    assert [e["error"] for e in remeasures] == [r.max_error for r in results]
    rows = replay(events, plain.fn, fhe.fn, kinds=("search", "remeasure"))
    assert len(rows) == len(events)
    assert all(r["logged"] == r["replayed"] for r in rows)
    events.clear()
    pre.run(budget=40, seeds=[1])
    assert [e["index"] for e in events] == list(range(len(events)))  # restarts each run


@pytest.mark.parametrize("k", [1, 2])
def test_jsonl_log_replays(tmp_path, k):
    path = tmp_path / "preact.jsonl"
    with JsonlEvaluationLog(path) as log:
        pre, _, fhe = _oracle(k, on_evaluation=log)
        result = pre.run(budget=30, seeds=[3], threshold=0.05)[0]
    events = read_log(path)
    assert len(events) == fhe.calls == result.n_trials + 1
    assert events[-1]["kind"] == "remeasure" and events[-1]["x"] == result.x
    rows = replay(events, pre._plain.fn, fhe.fn)
    assert len(rows) == 1
    assert rows[0]["logged"] == rows[0]["replayed"] == result.max_error


@pytest.mark.parametrize("k", [1, 2])
def test_no_sink_logs_nothing_and_search_is_unchanged(monkeypatch, k):
    sinks = []
    real_init = core_module.FHEOracle.__init__

    def spy_init(self, *args, **kwargs):
        sinks.append(kwargs.get("on_evaluation"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(core_module.FHEOracle, "__init__", spy_init)
    events = []
    logged, logged_plain, logged_fhe = _oracle(k, on_evaluation=events.append)
    bare, bare_plain, bare_fhe = _oracle(k)
    with_sink = logged.run(budget=40, seeds=[1, 2])
    without = bare.run(budget=40, seeds=[1, 2])
    assert events and len(events) == logged_fhe.calls
    assert (bare_plain.calls, bare_fhe.calls) == (logged_plain.calls, logged_fhe.calls)
    for r1, r2 in zip(with_sink, without):
        assert (r1.x, r1.z, r1.max_error, r1.clip_distance, r1.n_trials) == (
            r2.x, r2.z, r2.max_error, r2.clip_distance, r2.n_trials)
    # The inner z-space FHEOracle never logs, with or without a sink.
    assert len(sinks) == (4 if k == 2 else 0)
    assert all(s is None for s in sinks)


@pytest.mark.parametrize("k", [1, 2])
def test_run_rejects_on_evaluation(k):
    events = []
    pre, _, fhe = _oracle(k)
    with pytest.raises(TypeError, match="on_evaluation"):
        pre.run(budget=20, seeds=[1], on_evaluation=events.append)
    assert events == [] and fhe.calls == 0


@pytest.mark.parametrize("k", [1, 2])
def test_failed_evaluation_is_not_logged(k):
    events = []
    pre, _, fhe = _oracle(k, on_evaluation=events.append)
    good = fhe.fn

    def flaky(x):
        return np.nan * good(x) if fhe.calls == 5 else good(x)

    fhe.fn = flaky
    with pytest.raises(EvaluationError):
        pre.run(budget=40, seeds=[1])
    assert fhe.calls == 5 and len(events) == 4
