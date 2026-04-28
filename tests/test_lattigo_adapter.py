# Copyright (C) 2026 Bader Alissaei
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Lattigo subprocess adapter.

Two layers:

1. End-to-end smoke tests against the real Go binary (skipped if the
   binary is not built — build with `cd benchmarks/lattigo_probe && go
   build .`).

2. Mock-based unit tests of the failure paths (always run): malformed
   CSV, non-zero exit, timeout, NaN/Inf rejection, allowlist
   enforcement, timeout cap, binary path validation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fhe_oracle.adapters.lattigo import (
    LattigoProbe,
    LattigoProbeError,
    LattigoPrecisionRow,
    _checked_float,
    _parse_csv,
)


_BINARY = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "lattigo_probe"
    / "lattigo_probe"
)


# ----------------------------------------------------------------------
# End-to-end smoke tests (skipped without binary)
# ----------------------------------------------------------------------


_skip_no_binary = pytest.mark.skipif(
    not _BINARY.exists(),
    reason=f"Lattigo probe not built at {_BINARY}; run `go build .` in benchmarks/lattigo_probe/",
)


@_skip_no_binary
def test_probe_smoke_three_inputs():
    probe = LattigoProbe()
    rows = probe.precision_per_input(
        inputs=[[1.0], [2.0], [3.0]],
        w=[0.5],
        b=0.5,
    )
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert isinstance(row, LattigoPrecisionRow)
        assert row.idx == i
        assert 20.0 < row.mean_bits < 50.0


@_skip_no_binary
def test_probe_decrypts_within_tolerance():
    probe = LattigoProbe()
    rows = probe.precision_per_input(
        inputs=[[1.0], [2.0]],
        w=[0.5],
        b=0.5,
    )
    expected = [(0.5 * 1.0 + 0.5) ** 2, (0.5 * 2.0 + 0.5) ** 2]
    for row, want in zip(rows, expected):
        assert row.plaintext_value == pytest.approx(want, abs=1e-9)
        assert row.decrypted_real == pytest.approx(want, abs=1e-4)


@_skip_no_binary
def test_probe_empty_inputs():
    probe = LattigoProbe()
    rows = probe.precision_per_input(inputs=[], w=[0.5], b=0.0)
    assert rows == []


# ----------------------------------------------------------------------
# Constructor validation (no binary needed)
# ----------------------------------------------------------------------


def test_missing_binary_raises():
    with pytest.raises(LattigoProbeError, match="not found"):
        LattigoProbe(binary="/nonexistent/lattigo_probe")


@_skip_no_binary
def test_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout_s must be in"):
        LattigoProbe(timeout_s=0.0)


@_skip_no_binary
def test_timeout_must_be_below_cap():
    with pytest.raises(ValueError, match="timeout_s must be in"):
        LattigoProbe(timeout_s=10_000.0)


@_skip_no_binary
def test_timeout_rejects_infinity():
    with pytest.raises(ValueError, match="timeout_s must be in"):
        LattigoProbe(timeout_s=float("inf"))


# ----------------------------------------------------------------------
# Allowlist enforcement
# ----------------------------------------------------------------------


@_skip_no_binary
def test_unknown_circuit_rejected():
    probe = LattigoProbe()
    with pytest.raises(LattigoProbeError, match="Unknown circuit"):
        probe.precision_per_input(
            inputs=[[1.0]], w=[0.5], circuit="not_a_real_circuit"
        )


@_skip_no_binary
def test_unknown_params_rejected():
    probe = LattigoProbe()
    with pytest.raises(LattigoProbeError, match="Unknown param set"):
        probe.precision_per_input(
            inputs=[[1.0]], w=[0.5], params="not_a_real_paramset"
        )


@_skip_no_binary
def test_too_many_inputs_rejected():
    probe = LattigoProbe()
    too_many = [[0.0]] * 10_001
    with pytest.raises(LattigoProbeError, match="exceeds maximum"):
        probe.precision_per_input(inputs=too_many, w=[0.5])


# ----------------------------------------------------------------------
# Mock-based subprocess failure paths (always run)
# ----------------------------------------------------------------------


def _mock_probe(monkeypatch, returncode=0, stdout=b"", stderr=b""):
    """Build a LattigoProbe that bypasses the binary existence check."""
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    probe = LattigoProbe(binary="/fake/binary")
    mock_result = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
    return probe, mock_result


def test_nonzero_exit_raises_with_stderr_snippet(monkeypatch):
    probe, mock_result = _mock_probe(
        monkeypatch, returncode=2, stderr=b"go runtime panic"
    )
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="exit 2") as exc:
            probe.precision_per_input([[1.0]], [0.5])
        assert "go runtime panic" in str(exc.value)


def test_nonzero_exit_truncates_long_stderr(monkeypatch):
    huge = b"X" * 5000
    probe, mock_result = _mock_probe(monkeypatch, returncode=1, stderr=huge)
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="truncated") as exc:
            probe.precision_per_input([[1.0]], [0.5])
        # Truncated output should be smaller than the original
        assert len(str(exc.value)) < 3000


def test_timeout_raises(monkeypatch):
    probe, _ = _mock_probe(monkeypatch)
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="probe", timeout=120.0),
    ):
        with pytest.raises(LattigoProbeError, match="timed out after 120"):
            probe.precision_per_input([[1.0]], [0.5])


def test_malformed_csv_wrong_column_count(monkeypatch):
    probe, mock_result = _mock_probe(
        monkeypatch, returncode=0, stdout=b"0,1.0,2.0\n"
    )
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="got 3 cols"):
            probe.precision_per_input([[1.0]], [0.5])


def test_malformed_csv_non_numeric(monkeypatch):
    probe, mock_result = _mock_probe(
        monkeypatch,
        returncode=0,
        stdout=b"0,not_a_number,1,1,0,1,1\n",
    )
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="parse"):
            probe.precision_per_input([[1.0]], [0.5])


def test_csv_rejects_nan(monkeypatch):
    probe, mock_result = _mock_probe(
        monkeypatch, returncode=0, stdout=b"0,nan,1,1,0,1,1\n"
    )
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="Non-finite"):
            probe.precision_per_input([[1.0]], [0.5])


def test_csv_rejects_inf(monkeypatch):
    probe, mock_result = _mock_probe(
        monkeypatch, returncode=0, stdout=b"0,inf,1,1,0,1,1\n"
    )
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(LattigoProbeError, match="Non-finite"):
            probe.precision_per_input([[1.0]], [0.5])


def test_invalid_utf8_does_not_crash(monkeypatch):
    # Non-UTF-8 bytes followed by valid CSV should not raise
    # UnicodeDecodeError (errors='replace' is configured).
    probe, mock_result = _mock_probe(
        monkeypatch,
        returncode=0,
        stdout=b"0,30.0,30.0,30.0,0.0,1.0,1.0\n\xff\xfe",
    )
    with patch("subprocess.run", return_value=mock_result):
        # Either it parses one row and ignores the trash, or raises a
        # LattigoProbeError. The non-acceptable outcome is an
        # UnhandledExceptionError leaking out.
        try:
            probe.precision_per_input([[1.0]], [0.5])
        except LattigoProbeError:
            pass


# ----------------------------------------------------------------------
# Helper-level unit tests
# ----------------------------------------------------------------------


def test_checked_float_accepts_finite():
    assert _checked_float("1.5", "x") == 1.5
    assert _checked_float("-0.0", "x") == 0.0


def test_checked_float_rejects_nan():
    with pytest.raises(LattigoProbeError, match="Non-finite"):
        _checked_float("nan", "x")


def test_checked_float_rejects_inf():
    with pytest.raises(LattigoProbeError, match="Non-finite"):
        _checked_float("inf", "x")
    with pytest.raises(LattigoProbeError, match="Non-finite"):
        _checked_float("-inf", "x")


def test_parse_csv_skips_blank_rows():
    blob = "\n0,30.0,30.0,30.0,0.0,1.0,1.0\n\n"
    rows = list(_parse_csv(blob))
    assert len(rows) == 1
    assert rows[0].idx == 0
