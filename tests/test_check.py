# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.check."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fhe_oracle.check import CheckResult, check
from fhe_oracle.diagnostics import OperationStep


def _square(x):
    return float(np.sum(np.asarray(x) ** 2))


def _hot_zone_bug(x):
    v = _square(x)
    return v + (0.5 if v > 8.0 else 1e-5 * v)


def test_check_pass_when_fhe_matches():
    result = check(_square, _square, input_bounds=[(-2.0, 2.0)] * 3, seed=0)
    assert isinstance(result, CheckResult)
    assert result.oracle_result.verdict == "PASS"
    assert result.shrink_result is None
    assert result.localized_fault is None
    assert "## Diagnostics" not in result.report


def test_check_fail_auto_shrinks():
    result = check(
        _square, _hot_zone_bug, input_bounds=[(-3.0, 3.0)] * 4,
        n_trials=300, threshold=1e-3, seed=1,
    )
    assert result.oracle_result.verdict == "FAIL"
    assert result.shrink_result is not None
    assert result.shrink_result.shrunk_norm <= result.shrink_result.original_norm
    assert "## Diagnostics" in result.report
    assert "shrunk_input" in result.report


def test_check_shrink_disabled():
    result = check(
        _square, _hot_zone_bug, input_bounds=[(-3.0, 3.0)] * 4,
        n_trials=300, threshold=1e-3, seed=1, shrink=False,
    )
    assert result.oracle_result.verdict == "FAIL"
    assert result.shrink_result is None


def test_check_no_localization_for_untraced_fn():
    result = check(
        _square, _hot_zone_bug, input_bounds=[(-3.0, 3.0)] * 4,
        n_trials=300, threshold=1e-3, seed=1,
    )
    assert result.localized_fault is None


def test_check_localizes_when_fn_is_traceable():
    class TraceableBug:
        def __call__(self, x):
            return _hot_zone_bug(x)

        def trace(self, x):
            v = _square(x)
            return [
                OperationStep(
                    name="square",
                    plaintext_value=v,
                    fhe_value=v,
                    step_error=0.0,
                    cumulative_error=0.0,
                ),
                OperationStep(
                    name="hotzone_add",
                    plaintext_value=v,
                    fhe_value=_hot_zone_bug(x),
                    step_error=abs(_hot_zone_bug(x) - v),
                    cumulative_error=abs(_hot_zone_bug(x) - v),
                ),
            ]

    result = check(
        _square, TraceableBug(), input_bounds=[(-3.0, 3.0)] * 4,
        n_trials=300, threshold=1e-3, seed=1,
    )
    assert result.oracle_result.verdict == "FAIL"
    assert result.localized_fault is not None
    assert result.localized_fault.name == "hotzone_add"
    assert "localized_fault" in result.report


def test_check_json_format():
    result = check(
        _square, _hot_zone_bug, input_bounds=[(-3.0, 3.0)] * 4,
        n_trials=300, threshold=1e-3, seed=1, report_format="json",
    )
    payload = json.loads(result.report)
    assert payload["verdict"] == "FAIL"
    assert "diagnostics" in payload


def test_check_forwards_autooracle_kwargs():
    # n_probes is an AutoOracle kwarg, not a check() kwarg -- confirm
    # it's forwarded rather than raising a TypeError.
    result = check(
        _square, _square, input_bounds=[(-2.0, 2.0)] * 3, seed=0, n_probes=10,
    )
    assert result.oracle_result.verdict == "PASS"


def test_check_rejects_undersized_budget():
    with pytest.raises(ValueError):
        check(_square, _square, input_bounds=[(-2.0, 2.0)] * 3, n_trials=10)


_W = np.array([[1.0, 0.5, -0.3, 0.2]])
_B = np.array([0.1])


def _lr_plain(x):
    return float(1.0 / (1.0 + np.exp(-(_W @ np.asarray(x) + _B)[0])))


def _lr_taylor3(x):
    z = float((_W @ np.asarray(x) + _B)[0])
    return 0.5 + z / 4.0 - z ** 3 / 48.0


@pytest.mark.parametrize("threshold,verdict", [(0.05, "FAIL"), (100.0, "PASS")])
def test_check_preactivation_dispatch_reports_verdict(threshold, verdict):
    result = check(_lr_plain, _lr_taylor3, [(-3.0, 3.0)] * 4, n_trials=200,
                   threshold=threshold, W=_W, b=_B)
    r = result.oracle_result
    assert r.regime == "preactivation_dominated"
    assert r.verdict == verdict
    assert r.threshold == threshold
    assert list(r.worst_input) == list(r.x)
    assert result.shrink_result is None
    assert f"**Verdict:** {verdict}" in result.report


def test_check_preactivation_dispatch_json_report():
    result = check(_lr_plain, _lr_taylor3, [(-3.0, 3.0)] * 4, n_trials=200,
                   threshold=0.05, W=_W, b=_B, report_format="json")
    payload = json.loads(result.report)
    assert payload["verdict"] == "FAIL"
    assert len(payload["worst_input"]) == 4
