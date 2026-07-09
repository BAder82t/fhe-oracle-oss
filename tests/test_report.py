# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.report."""

from __future__ import annotations

import json

from fhe_oracle.core import OracleResult
from fhe_oracle.report import to_json, to_markdown


def _make_result(verdict="FAIL"):
    return OracleResult(
        verdict=verdict,
        max_error=0.5,
        worst_input=[1.0, 2.0],
        threshold=0.01,
        n_trials=100,
        elapsed_seconds=1.23,
    )


def test_to_markdown_basic_fields():
    md = to_markdown(_make_result())
    assert "**Verdict:** FAIL" in md
    assert "1.000000, 2.000000" in md


def test_to_markdown_without_diagnostics_has_no_diagnostics_section():
    md = to_markdown(_make_result())
    assert "## Diagnostics" not in md


def test_to_markdown_with_diagnostics_on_fail():
    md = to_markdown(
        _make_result("FAIL"),
        diagnostics={"localized_fault": "op_2", "structure": "rank=3"},
    )
    assert "## Diagnostics" in md
    assert "**localized_fault:** op_2" in md
    assert "**structure:** rank=3" in md


def test_to_markdown_diagnostics_suppressed_on_pass():
    # nothing to diagnose on a PASS -- diagnostics section is omitted
    # even if the caller supplies a dict.
    md = to_markdown(_make_result("PASS"), diagnostics={"foo": "bar"})
    assert "## Diagnostics" not in md


def test_to_markdown_empty_diagnostics_dict_is_treated_as_absent():
    md = to_markdown(_make_result("FAIL"), diagnostics={})
    assert "## Diagnostics" not in md


def test_to_json_empty_diagnostics_dict_is_treated_as_absent():
    payload = json.loads(to_json(_make_result("FAIL"), diagnostics={}))
    assert "diagnostics" not in payload


def test_to_json_without_diagnostics():
    payload = json.loads(to_json(_make_result()))
    assert "diagnostics" not in payload
    assert payload["verdict"] == "FAIL"


def test_to_json_with_diagnostics():
    payload = json.loads(
        to_json(_make_result(), diagnostics={"localized_fault": "op_2"})
    )
    assert payload["diagnostics"] == {"localized_fault": "op_2"}
