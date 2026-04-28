# Copyright (C) 2026 Bader Alissaei
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end smoke test for the Lattigo subprocess adapter.

Skipped when the Go binary is not built. Build with:

    cd benchmarks/lattigo_probe && go build .
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fhe_oracle.adapters.lattigo import (
    LattigoProbe,
    LattigoProbeError,
    LattigoPrecisionRow,
)


_BINARY = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "lattigo_probe"
    / "lattigo_probe"
)

pytestmark = pytest.mark.skipif(
    not _BINARY.exists(),
    reason=f"Lattigo probe not built at {_BINARY}; run `go build .` in benchmarks/lattigo_probe/",
)


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
        # CKKS at LogScale=40 should give ~25-35 bits on a depth-2 circuit.
        assert 20.0 < row.mean_bits < 50.0


def test_probe_decrypts_within_tolerance():
    probe = LattigoProbe()
    rows = probe.precision_per_input(
        inputs=[[1.0], [2.0]],
        w=[0.5],
        b=0.5,
    )
    # plaintext (w*x + b)^2 = (0.5*x + 0.5)^2
    expected = [(0.5 * 1.0 + 0.5) ** 2, (0.5 * 2.0 + 0.5) ** 2]
    for row, want in zip(rows, expected):
        assert row.plaintext_value == pytest.approx(want, abs=1e-9)
        assert row.decrypted_real == pytest.approx(want, abs=1e-4)


def test_probe_empty_inputs():
    probe = LattigoProbe()
    rows = probe.precision_per_input(inputs=[], w=[0.5], b=0.0)
    assert rows == []


def test_missing_binary_raises():
    with pytest.raises(LattigoProbeError, match="not found"):
        LattigoProbe(binary="/nonexistent/lattigo_probe")
