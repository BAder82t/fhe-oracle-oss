# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for fhe_oracle.cli."""

from __future__ import annotations

import json

import pytest

from fhe_oracle.cli import main

_PASS_MODEL = """
import numpy as np

def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))

fhe_fn = plaintext_fn
input_bounds = [(-2.0, 2.0)] * 3
"""

_FAIL_MODEL = """
import numpy as np

def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))

def fhe_fn(x):
    v = float(np.sum(np.asarray(x) ** 2))
    return plaintext_fn(x) + (0.5 if v > 8.0 else 1e-5 * v)

input_bounds = [(-3.0, 3.0)] * 4
n_trials = 300
threshold = 1e-3
seed = 1
"""


def _write_model(tmp_path, content, name="model.py"):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_cli_check_pass_exits_zero(tmp_path, capsys):
    model = _write_model(tmp_path, _PASS_MODEL)
    code = main(["check", str(model)])
    assert code == 0
    out = capsys.readouterr().out
    assert "**Verdict:** PASS" in out


def test_cli_check_fail_exits_one(tmp_path, capsys):
    model = _write_model(tmp_path, _FAIL_MODEL)
    code = main(["check", str(model)])
    assert code == 1
    out = capsys.readouterr().out
    assert "**Verdict:** FAIL" in out
    assert "## Diagnostics" in out


def test_cli_module_level_config_used_when_no_flags(tmp_path, capsys):
    # _FAIL_MODEL sets threshold=1e-3 and seed=1 at module level; no
    # CLI flags override them -- confirm they're actually picked up.
    model = _write_model(tmp_path, _FAIL_MODEL)
    code = main(["check", str(model)])
    out = capsys.readouterr().out
    assert "**Threshold:** 1.000000e-03" in out
    assert code == 1


def test_cli_flags_override_module_level_config(tmp_path, capsys):
    model = _write_model(tmp_path, _FAIL_MODEL)
    code = main(["check", str(model), "--threshold", "10.0"])
    out = capsys.readouterr().out
    # threshold=10 is way above max_error -> PASS despite module default FAIL.
    assert "**Verdict:** PASS" in out
    assert code == 0


def test_cli_no_shrink_flag_omits_diagnostics(tmp_path, capsys):
    model = _write_model(tmp_path, _FAIL_MODEL)
    code = main(["check", str(model), "--no-shrink"])
    out = capsys.readouterr().out
    assert code == 1
    assert "## Diagnostics" not in out


def test_cli_json_format(tmp_path, capsys):
    model = _write_model(tmp_path, _FAIL_MODEL)
    code = main(["check", str(model), "--format", "json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["verdict"] == "FAIL"
    assert code == 1


def test_cli_missing_file_exits_two(capsys):
    code = main(["check", "/nonexistent/path/model.py"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_missing_required_attrs_exits_two(tmp_path, capsys):
    model = _write_model(tmp_path, "x = 1\n", name="bad.py")
    code = main(["check", str(model)])
    assert code == 2
    err = capsys.readouterr().err
    assert "plaintext_fn" in err
    assert "fhe_fn" in err
    assert "input_bounds" in err


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
