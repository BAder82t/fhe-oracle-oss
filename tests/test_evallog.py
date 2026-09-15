# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluation log: every counted evaluation is recorded and replayable."""

from __future__ import annotations

import hashlib

import numpy as np

from fhe_oracle import FHEOracle, JsonlEvaluationLog, read_log, replay


def _plain(x):
    return float(np.sum(np.asarray(x) ** 2))


def _fhe(x):
    v = _plain(x)
    return v + 1e-3 * v * (10.0 if v > 1.5 else 1.0)


def _oracle(**kwargs):
    return FHEOracle(_plain, _fhe, input_dim=3, input_bounds=[(-1.0, 1.0)] * 3,
                     seed=4, **kwargs)


def test_log_records_search_and_remeasure():
    events = []
    result = _oracle(on_evaluation=events.append).run(n_trials=41, threshold=0.01)
    search = [e for e in events if e["kind"] == "search"]
    assert len(search) == result.n_trials
    assert [e["index"] for e in events] == list(range(len(events)))
    assert events[-1]["kind"] == "remeasure"
    assert events[-1]["x"] == result.worst_input
    assert events[-1]["error"] == result.remeasured_error
    assert max(e["score"] for e in search) == result.search_max_error


def test_batch_mode_logs_the_same_events():
    seq, bat = [], []
    _oracle(on_evaluation=seq.append).run(n_trials=41, threshold=0.01)
    _oracle(on_evaluation=bat.append,
            batch_fhe_fn=lambda xs: [_fhe(x) for x in xs]).run(n_trials=41, threshold=0.01)
    assert seq == bat


def test_shrink_events_are_logged():
    events = []
    oracle = _oracle(on_evaluation=events.append)
    result = oracle.run(n_trials=41, threshold=0.01)
    start = len(events)
    shrunk = oracle.shrink(result, max_evals=40)
    kinds = [e["kind"] for e in events[start:]]
    assert set(kinds) <= {"shrink", "shrink_verify"}
    assert len(kinds) == shrunk.n_evals


def test_jsonl_log_roundtrip_hash_and_replay(tmp_path):
    path = tmp_path / "run.jsonl"
    with JsonlEvaluationLog(path) as log:
        result = _oracle(on_evaluation=log).run(n_trials=41, threshold=0.01)
        digest = log.sha256()
    events = read_log(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert events[-1]["error"] == result.remeasured_error
    (row,) = replay(events, _plain, _fhe)
    assert row["replayed"] == row["logged"] == result.remeasured_error
