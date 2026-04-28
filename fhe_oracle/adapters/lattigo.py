# Copyright (C) 2026 Bader Alissaei
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lattigo CKKS subprocess adapter.

Calls out to the Go binary at ``benchmarks/lattigo_probe/lattigo_probe``
to obtain decrypt-based precision statistics from Lattigo v6's
``GetPrecisionStats``-equivalent path. Used by the Item 17 measurement
probe and (future) by Item 07 bootstrapping-aware search.

This adapter does **not** implement the Adapter protocol used by Core
(``encrypt`` / ``noise_budget`` / ``decrypt``) — Lattigo does not
expose a blind noise-budget API. It produces precision statistics
batched across many inputs in a single subprocess call instead.

Example
-------
    from fhe_oracle.adapters.lattigo import LattigoProbe

    probe = LattigoProbe()
    rows = probe.precision_per_input(
        inputs=[[1.0], [2.0], [3.0]],
        w=[0.5],
        b=0.5,
    )
    # rows[i] = LattigoPrecisionRow(idx=i, mean_bits=..., ...)
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


_DEFAULT_BINARY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "lattigo_probe"
    / "lattigo_probe"
)


@dataclass(frozen=True)
class LattigoPrecisionRow:
    """One row of the probe's CSV output."""

    idx: int
    mean_bits: float
    min_bits: float
    max_bits: float
    std_bits: float
    plaintext_value: float
    decrypted_real: float


class LattigoProbeError(RuntimeError):
    """Raised on subprocess / parse failure."""


class LattigoProbe:
    """Subprocess wrapper around the Go Lattigo precision probe.

    Parameters
    ----------
    binary : str | Path | None
        Path to the compiled ``lattigo_probe`` Go binary. Defaults to
        the in-tree build at ``benchmarks/lattigo_probe/lattigo_probe``
        (build with ``go build`` in that directory).
    timeout_s : float
        Subprocess timeout.
    """

    def __init__(
        self,
        binary: Optional[os.PathLike] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.binary = Path(binary) if binary is not None else _DEFAULT_BINARY
        if not self.binary.exists():
            raise LattigoProbeError(
                f"Lattigo probe binary not found at {self.binary}. "
                f"Build with: cd benchmarks/lattigo_probe && go build ."
            )
        self.timeout_s = timeout_s

    def precision_per_input(
        self,
        inputs: Sequence[Sequence[float]],
        w: Sequence[float],
        b: float = 0.0,
        circuit: str = "wxb_squared",
        params: str = "n14_logq200",
    ) -> list[LattigoPrecisionRow]:
        """Run the probe and return one precision row per input.

        Parameters
        ----------
        inputs : sequence of input vectors
        w : weight vector (must match input dim)
        b : scalar bias
        circuit : circuit identifier (currently only ``wxb_squared``)
        params : Lattigo param set name

        Returns
        -------
        list of LattigoPrecisionRow, one per input, in input order.
        """
        if not inputs:
            return []

        job = {
            "circuit": circuit,
            "params": params,
            "inputs": [list(map(float, x)) for x in inputs],
            "w": list(map(float, w)),
            "b": float(b),
        }
        payload = json.dumps(job).encode("utf-8")

        try:
            res = subprocess.run(
                [str(self.binary)],
                input=payload,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LattigoProbeError(
                f"Lattigo probe timed out after {self.timeout_s}s"
            ) from exc

        if res.returncode != 0:
            raise LattigoProbeError(
                f"Lattigo probe failed (exit {res.returncode}): "
                f"{res.stderr.decode('utf-8', errors='replace')}"
            )

        return list(_parse_csv(res.stdout.decode("utf-8")))


def _parse_csv(blob: str) -> list[LattigoPrecisionRow]:
    rows: list[LattigoPrecisionRow] = []
    reader = csv.reader(io.StringIO(blob))
    for raw in reader:
        if not raw:
            continue
        if len(raw) != 7:
            raise LattigoProbeError(
                f"Unexpected probe CSV row (got {len(raw)} cols): {raw!r}"
            )
        rows.append(
            LattigoPrecisionRow(
                idx=int(raw[0]),
                mean_bits=float(raw[1]),
                min_bits=float(raw[2]),
                max_bits=float(raw[3]),
                std_bits=float(raw[4]),
                plaintext_value=float(raw[5]),
                decrypted_real=float(raw[6]),
            )
        )
    return rows
