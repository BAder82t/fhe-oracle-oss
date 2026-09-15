# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AutoOracle boundary probe: vertex snapping, flip climbing and witness merge."""

from __future__ import annotations

import numpy as np
import pytest

import fhe_oracle.core as core_module
from fhe_oracle.autoconfig import AutoOracle, classify_landscape
from fhe_oracle.check import check


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3(z):
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _lr_landscape(d=8, seed=5):
    # Taylor-3 sigmoid on w.x+b: error grows with |z|, so the supremum is a vertex.
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, d)
    b = 0.1
    bounds = [(-3.0, 3.0)] * d
    z_ext = [b + 3.0 * np.abs(w).sum(), b - 3.0 * np.abs(w).sum()]
    sup = max(abs(_sigmoid(z) - _taylor3(z)) for z in z_ext)
    return (lambda x: float(_sigmoid(w @ np.asarray(x) + b)),
            lambda x: float(_taylor3(w @ np.asarray(x) + b)), bounds, sup)


def _bump_landscape(d=8):
    # Smooth bump centred inside the box: supremum 1.0 at an interior point.
    c = np.linspace(-1.2, 1.2, d)
    return (lambda x: 0.0,
            lambda x: float(np.exp(-np.sum((np.asarray(x) - c) ** 2) / 4.5)),
            [(-3.0, 3.0)] * d)


def _counted(fn):
    def wrapped(x):
        wrapped.calls += 1
        return fn(x)
    wrapped.calls = 0
    return wrapped


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_autooracle_reaches_vertex_supremum(seed):
    plain, fhe, bounds, sup = _lr_landscape()
    result = AutoOracle(plain, fhe, bounds).run(n_trials=200, seed=seed)
    assert result.max_error >= sup * (1 - 1e-9)
    assert all(abs(v) == 3.0 for v in result.worst_input)


def test_check_reaches_vertex_supremum():
    plain, fhe, bounds, sup = _lr_landscape(d=6, seed=11)
    out = check(plain, fhe, bounds, n_trials=200, seed=4, shrink=False)
    assert out.oracle_result.max_error >= sup * (1 - 1e-9)


def test_vertex_landscape_within_budget():
    plain, fhe, bounds, _ = _lr_landscape()
    fhe = _counted(fhe)
    result = AutoOracle(plain, fhe, bounds).run(n_trials=200, seed=1)
    assert fhe.calls <= 200
    assert result.n_trials <= 200


def test_probe_points_align_with_divergences():
    batch = [1.0] * 30 + [1.6] * 12 + [2.0] * 7 + [3.5]
    pool = np.array(batch * 2)
    fhe = _counted(lambda x: float(pool[(fhe.calls - 1) % pool.size]))
    res = classify_landscape(lambda x: 0.0, fhe, [(-3.0, 3.0)] * 4, n_probes=50, seed=0)
    assert res.probe_points.shape == (res.probe_divergences.size, 4)
    assert res.probe_divergences.size == 100


def test_boundary_probe_is_cheap_and_recycled_on_interior_landscape(monkeypatch):
    plain, fhe, bounds = _bump_landscape()
    fhe = _counted(fhe)
    seen = {}
    real_run = core_module.FHEOracle.run

    def spy_run(self, n_trials=500, **kwargs):
        seen["calls_at_entry"] = fhe.calls
        seen["n_trials"] = n_trials
        return real_run(self, n_trials=n_trials, **kwargs)

    monkeypatch.setattr(core_module.FHEOracle, "run", spy_run)
    ao = AutoOracle(plain, fhe, bounds)
    result = ao.run(n_trials=200, seed=1)
    assert seen["calls_at_entry"] - ao.probe_result.n_evals <= 4
    # Two calls are held back: the inner re-measurement and a possible witness re-measurement.
    assert seen["n_trials"] == 200 - seen["calls_at_entry"] - 2
    assert not all(abs(v) == 3.0 for v in result.worst_input)


def test_boundary_vertices_stay_inside_asymmetric_bounds():
    # 0.1 + 0.7 - 0.7 != 0.1 in floating point; vertices must be exact bounds.
    d = 6
    signs = np.array([1.0, -1.0] * 3)
    oob = []

    def fhe(x):
        xa = np.asarray(x)
        if np.any(xa < 0.1) or np.any(xa > 0.7):
            oob.append(xa)
        return float(_taylor3(12.0 * signs @ (xa - 0.4)))

    def plain(x):
        return float(_sigmoid(12.0 * signs @ (np.asarray(x) - 0.4)))

    result = AutoOracle(plain, fhe, [(0.1, 0.7)] * d).run(n_trials=200, seed=1)
    assert not oob
    assert all(v in (0.1, 0.7) for v in result.worst_input)


def _noisy_vertex_landscape(d=6, threshold=0.01):
    # True error 0.9*threshold at every vertex; the first visit to a vertex reads 1.1*threshold.
    visits: dict[tuple, int] = {}
    calls: list[tuple[tuple, float]] = []

    def fhe(x):
        xa = np.asarray(x, dtype=float)
        err = 0.9 * threshold * (np.mean(np.abs(xa)) / 3.0) ** 4
        if np.all(np.abs(xa) == 3.0):
            key = tuple(xa.tolist())
            visits[key] = visits.get(key, 0) + 1
            if visits[key] == 1:
                err = 1.1 * threshold
        calls.append((tuple(xa.tolist()), err))
        return err

    return (lambda x: 0.0), fhe, [(-3.0, 3.0)] * d, calls


def test_boundary_witness_is_remeasured_like_the_search_witness():
    plain, fhe, bounds, calls = _noisy_vertex_landscape()
    result = AutoOracle(plain, fhe, bounds).run(n_trials=200, seed=1, threshold=0.01)
    assert all(abs(v) == 3.0 for v in result.worst_input)
    last_x, last_err = calls[-1]
    assert last_x == tuple(result.worst_input)
    assert result.remeasured_error == pytest.approx(last_err)
    assert result.remeasured_error < 0.01
    # A counted evaluation met the threshold, so the core verdict rule says FAIL.
    assert result.search_max_error == pytest.approx(0.011)
    assert result.max_error == pytest.approx(0.011)
    assert result.verdict == "FAIL"
    # Two re-measurements (search witness, then the vertex); everything else is in n_trials.
    assert len(calls) == result.n_trials + 2


def test_boundary_witness_matches_a_logged_evaluation():
    plain, fhe, bounds, _ = _lr_landscape()
    log = []

    def logged(x):
        y = fhe(x)
        log.append((tuple(np.asarray(x, dtype=float).tolist()), abs(plain(x) - y)))
        return y

    result = AutoOracle(plain, logged, bounds).run(n_trials=200, seed=2, threshold=0.01)
    x = tuple(result.worst_input)
    assert any(lx == x and abs(e - result.max_error) <= 1e-12 * result.max_error
               for lx, e in log)
    assert result.verdict == "FAIL"


def _spy_boundary(monkeypatch):
    import fhe_oracle.autoconfig as autoconfig

    calls = []
    real = autoconfig._boundary_probe

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(autoconfig, "_boundary_probe", spy)
    return calls


def test_boundary_probe_skipped_for_preactivation(monkeypatch):
    calls = _spy_boundary(monkeypatch)
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(1, 10)), rng.normal(size=1)

    def cubic(x):
        return float(abs((W @ np.asarray(x) + b)[0]) ** 3 / 10.0)

    result = AutoOracle(lambda x: 0.0, cubic, [(-3.0, 3.0)] * 10, W=W, b=b,
                        n_probes=30).run(n_trials=80, seed=2)
    assert result.strategy_used == "preactivation"
    assert not calls


class _FakeAdapter:
    def __init__(self, fn):
        self._fn = fn

    def get_scheme_name(self):
        return "fake"

    def evaluate(self, x):
        return self._fn(x)

    def encrypt(self, x):
        return list(x)

    def run_fhe_program(self, ct):
        return self._fn(ct)

    def decrypt(self, ct):
        return ct

    def get_noise_budget(self, ct):
        return 0.0

    def get_mult_depth_used(self, ct):
        return 0


def _scores(x):
    s = 0.3 * float(np.tanh(np.sum(x)))
    return [0.5 + s, 0.5 - s]


@pytest.mark.parametrize("kwargs", [
    {"adapter": "fake"},
    {"multi_output": True, "multi_output_mode": "rank_inversion"},
    {"fitness": "custom"},
])
def test_boundary_probe_skipped_when_search_objective_differs(monkeypatch, kwargs):
    # The probe scores fhe_fn divergence; other objectives must not be overridden by it.
    calls = _spy_boundary(monkeypatch)
    plain, fhe, bounds, _ = _lr_landscape()
    if "multi_output" in kwargs:
        plain, fhe = _scores, (lambda x: [v + 1e-3 for v in _scores(x)])
    kw = dict(kwargs)
    if kw.get("adapter") == "fake":
        kw["adapter"] = _FakeAdapter(fhe)
    if kw.get("fitness") == "custom":
        kw["fitness"] = type("F", (), {"score": staticmethod(lambda x: float(np.sum(x)))})()
    AutoOracle(plain, fhe, bounds, **kw).run(n_trials=200, seed=1)
    assert not calls
