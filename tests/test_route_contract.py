# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract for every public search route: verdict, in-bounds witness, bounded evaluations."""

from __future__ import annotations

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.autoconfig import AutoOracle
from fhe_oracle.cascade import CascadeSearch
from fhe_oracle.check import check
from fhe_oracle.empirical import EmpiricalSearch
from fhe_oracle.fitness import DivergenceFitness
from fhe_oracle.hybrid import run_hybrid
from fhe_oracle.preactivation import PreactivationOracle
from fhe_oracle.subspace import SubspaceOracle

D = 4
BOUNDS = [(-3.0, 3.0)] * D
N = 120  # search budget
M = 80   # empirical budget
K = 8    # cascade top_k
S = 60   # check() shrink budget
W = {
    1: np.array([[1.0, 0.5, -0.3, 0.2]]),
    2: np.array([[1.0, 0.5, -0.3, 0.2], [0.2, -0.4, 0.6, 0.1]]),
}
B = {1: np.array([0.1]), 2: np.array([0.1, -0.2])}
THRESHOLDS = {"FAIL": 0.05, "PASS": 100.0}
DATA = np.full((20, D), 2.9)  # near the upper corner, so jitter leaves the box


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _taylor3(z):
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _taylor5(z):
    return _taylor3(z) + z ** 5 / 480.0


def _affine(k, act):
    def f(x):
        out = act(W[k] @ np.asarray(x, dtype=np.float64) + B[k])
        return float(out[0]) if k == 1 else out
    return f


class _Counted:
    """Model callable that counts its evaluations."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.fn(x)


def _fhe_oracle(thr):
    fhe = _Counted(_affine(1, _taylor3))
    r = FHEOracle(_affine(1, _sigmoid), fhe, input_dim=D, input_bounds=BOUNDS,
                  seed=0).run(n_trials=N, threshold=thr)
    return r, [("fhe", fhe.calls, N + 1)]  # + final re-measurement


def _auto_oracle(k):
    def run(thr):
        fhe = _Counted(_affine(1, _taylor3))
        kw = {} if k is None else {"W": W[k], "b": B[k]}
        r = AutoOracle(_affine(1, _sigmoid), fhe, BOUNDS, **kw).run(
            n_trials=N, seed=0, threshold=thr)
        return r, [("fhe", fhe.calls, N)]  # n_trials includes probes and re-measurement
    return run


def _check(k):
    def run(thr):
        fhe = _Counted(_affine(1, _taylor3))
        kw = {} if k is None else {"W": W[k], "b": B[k]}
        out = check(_affine(1, _sigmoid), fhe, BOUNDS, n_trials=N, threshold=thr,
                    seed=0, shrink_max_evals=S, **kw)
        return out.oracle_result, [("fhe", fhe.calls, N + S)]  # + shrink pass on FAIL
    return run


def _hybrid(thr):
    fhe = _Counted(_affine(1, _taylor3))
    r = run_hybrid(plaintext_fn=_affine(1, _sigmoid), fhe_fn=fhe, input_dim=D,
                   input_bounds=BOUNDS, threshold=thr, oracle_budget=N, oracle_seed=0,
                   data=DATA, empirical_budget=M, jitter_std=0.3)
    return r, [("fhe", fhe.calls, N + 1 + M)]  # + oracle re-measurement


def _cascade(kind, k=1):
    def run(thr):
        cheap = _Counted(_affine(k, _taylor3))
        expensive = _Counted(_affine(k, _taylor5))
        cs = CascadeSearch(cheap_fhe_fn=cheap, expensive_fhe_fn=expensive,
                           plaintext_fn=_affine(k, _sigmoid), input_bounds=BOUNDS,
                           top_k=K, weights=(W[k], B[k]))
        r = cs.run(budget_cheap=N, seeds=[1], search_kind=kind, threshold=thr)[0]
        extra = 1 if kind == "preactivation" else 0  # PreactivationOracle re-measures
        return r, [("cheap", cheap.calls, N + extra), ("expensive", expensive.calls, K)]
    return run


def _preactivation(k):
    def run(thr):
        fhe = _Counted(_affine(k, _taylor3))
        pre = PreactivationOracle(W=W[k], b=B[k], plaintext_fn=_affine(k, _sigmoid),
                                  fhe_fn=fhe, input_bounds=BOUNDS)
        r = pre.run(budget=N, seeds=[1], threshold=thr)[0]
        return r, [("fhe", fhe.calls, N + 1)]  # + witness re-measurement
    return run


def _subspace(thr):
    fhe = _Counted(_affine(1, _taylor3))
    r = SubspaceOracle(_affine(1, _sigmoid), fhe, BOUNDS, subspace_dim=2).run(
        n_trials=N, seed=0, threshold=thr)
    return r, [("fhe", fhe.calls, N)]


def _empirical(thr):
    fhe = _Counted(_affine(1, _taylor3))
    div = DivergenceFitness(_affine(1, _sigmoid), fhe).score
    r = EmpiricalSearch(div, DATA, threshold=thr, budget=M, jitter_std=0.3,
                        seed=1, bounds=BOUNDS).run()
    return r, [("fhe", fhe.calls, M)]


ROUTES = [
    pytest.param(_fhe_oracle, id="FHEOracle"),
    pytest.param(_auto_oracle(None), id="AutoOracle"),
    pytest.param(_auto_oracle(1), id="AutoOracle-preactivation"),
    pytest.param(_check(None), id="check"),
    pytest.param(_check(1), id="check-preactivation"),
    pytest.param(_hybrid, id="run_hybrid"),
    pytest.param(_cascade("cma"), id="CascadeSearch-cma"),
    pytest.param(_cascade("random"), id="CascadeSearch-random"),
    pytest.param(_cascade("preactivation", 1), id="CascadeSearch-preactivation-k1"),
    pytest.param(_cascade("preactivation", 2), id="CascadeSearch-preactivation-k2"),
    pytest.param(_preactivation(1), id="PreactivationOracle-k1"),
    pytest.param(_preactivation(2), id="PreactivationOracle-k2"),
    pytest.param(_subspace, id="SubspaceOracle"),
    pytest.param(_empirical, id="EmpiricalSearch"),
]


@pytest.mark.parametrize("expected", ["FAIL", "PASS"])
@pytest.mark.parametrize("route", ROUTES)
def test_route_contract(route, expected):
    thr = THRESHOLDS[expected]
    result, budgets = route(thr)

    assert result.verdict in ("PASS", "FAIL")
    assert result.verdict == ("PASS" if result.max_error < thr else "FAIL")
    assert result.verdict == expected

    x = np.asarray(result.worst_input, dtype=np.float64)
    lo, hi = np.array(BOUNDS).T
    assert x.shape == (D,)
    assert np.all((lo <= x) & (x <= hi)), f"witness outside bounds: {x}"

    for label, used, allowed in budgets:
        assert used <= allowed, f"{label}: {used} evaluations > budget {allowed}"


def test_preactivation_mock_dispatches_to_preactivation():
    # The AutoOracle-preactivation route case is only meaningful if this mock takes that route.
    r = AutoOracle(_affine(1, _sigmoid), _affine(1, _taylor3), BOUNDS,
                   W=W[1], b=B[1]).run(n_trials=N, seed=0, threshold=0.05)
    assert r.regime == "preactivation_dominated"
