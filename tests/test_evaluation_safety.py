"""Regression tests: an invalid evaluation must never become a PASS."""
from types import SimpleNamespace

import numpy as np
import pytest

from fhe_oracle import EvaluationError, FHEOracle, check
from fhe_oracle.cli import main
from fhe_oracle.fitness import DivergenceFitness
from fhe_oracle.empirical import EmpiricalSearch
from fhe_oracle.multi_output import MultiOutputFitness
from fhe_oracle.preactivation import PreactivationOracle
from fhe_oracle.subspace import SubspaceOracle
from fhe_oracle.cascade import CascadeSearch


def broken(x):
    raise RuntimeError("backend unavailable")


def oracle(plain=lambda x: 1., fhe=lambda x: 1., **kwargs):
    return FHEOracle(plain, fhe, input_dim=2,
                     input_bounds=[(-1., 1.)] * 2, seed=42, **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"random_floor": 0.5}, {"restarts": 1},
                                   {"multi_output": True}])
def test_backend_failure_aborts_search(kwargs):
    with pytest.raises(EvaluationError, match="backend unavailable"):
        oracle(fhe=broken, **kwargs).run(n_trials=30)


@pytest.mark.parametrize("plain,fhe", [
    ([1., 999.], [1.]),
    ([[1., 2.]], [1., 2.]),
    ([], []),
    ([np.nan], [0.]),
    ([0.], [np.inf]),
    ([1e308], [-1e308]),
    ([1.+2.j], [1.]),
])
@pytest.mark.parametrize("route", ["direct", "auto", "multi"])
def test_invalid_outputs_never_pass(plain, fhe, route):
    with pytest.raises(EvaluationError):
        if route == "auto":
            check(lambda x: plain, lambda x: fhe, [(-1., 1.)] * 2,
                  n_trials=80)
        else:
            oracle(lambda x: plain, lambda x: fhe,
                   multi_output=route == "multi").run(n_trials=30)


def test_failure_after_valid_candidates_is_not_discarded():
    calls = 0

    def sometimes_broken(x):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("third candidate failed")
        return 1.

    with pytest.raises(EvaluationError, match="third candidate"):
        oracle(fhe=sometimes_broken).run(n_trials=30)
    assert calls == 3


@pytest.mark.parametrize("fhe", [broken, lambda x: [1., 2.], lambda x: np.nan])
def test_final_verification_rejects_invalid_backend(fhe):
    # A custom fitness bypasses model scoring; final verification must
    # independently enforce the output contract.
    with pytest.raises(EvaluationError, match="final evaluation"):
        oracle(fhe=fhe, fitness=SimpleNamespace(score=lambda x: 0.)).run(n_trials=30)


def test_invalid_custom_fitness_is_rejected():
    with pytest.raises(EvaluationError, match="finite"):
        oracle(fitness=SimpleNamespace(score=lambda x: np.nan)).run(n_trials=30)


def test_scalar_and_singleton_vector_remain_compatible():
    assert DivergenceFitness(lambda x: 1., lambda x: [1.]).score([0.]) == 0.


@pytest.mark.parametrize("restarts", [0, 1])
def test_one_dimensional_bounded_search(restarts):
    visited = []

    def fhe(x):
        visited.append(x[0])
        return x[0] + 0.1

    result = FHEOracle(lambda x: x[0], fhe, input_dim=1,
                       input_bounds=[(-1., 1.)], seed=42,
                       restarts=restarts).run(n_trials=40, threshold=0.01)
    assert result.verdict == "FAIL"
    assert visited and all(-1. <= x <= 1. for x in visited)


def test_empirical_failure_aborts():
    with pytest.raises(RuntimeError, match="backend unavailable"):
        EmpiricalSearch(broken, np.zeros((2, 2)), budget=3).run()


def test_multi_output_report_rejects_missing_values():
    with pytest.raises(EvaluationError, match="shape mismatch"):
        MultiOutputFitness(lambda x: [1., 2.], lambda x: [1.]).detailed_report([0.])


def test_preactivation_failure_aborts():
    search = PreactivationOracle(np.eye(2), np.zeros(2), lambda x: 1., broken,
                                 [(-1., 1.)] * 2)
    with pytest.raises(EvaluationError, match="backend unavailable"):
        search.run(budget=30, seeds=[42])


def test_subspace_failure_aborts():
    search = SubspaceOracle(lambda x: 1., broken, bounds=[(-1., 1.)] * 4, subspace_dim=2)
    with pytest.raises(EvaluationError, match="backend unavailable"):
        search.run(n_trials=30)


def test_cascade_expensive_failure_aborts():
    search = CascadeSearch(lambda x: 1., broken, lambda x: 1., [(-1., 1.)] * 2)
    with pytest.raises(EvaluationError, match="backend unavailable"):
        search._expensive_div([0., 0.])


def test_cli_evaluation_error_exit(tmp_path, capsys):
    model = tmp_path / "broken.py"
    model.write_text("def plaintext_fn(x): return 1.\n"
                     "def fhe_fn(x): raise RuntimeError('backend unavailable')\n"
                     "input_bounds = [(-1., 1.)] * 2\n")
    assert main(["check", str(model), "--n-trials", "80"]) == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "PASS" not in captured.out
