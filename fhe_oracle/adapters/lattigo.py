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
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence


# Default location only valid in a source checkout where benchmarks/
# sits two levels above fhe_oracle/adapters/. PyPI installs exclude
# benchmarks/ from the wheel, so users on a pip install must build
# the Go binary separately and pass `binary=...` explicitly.
_DEFAULT_BINARY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "lattigo_probe"
    / "lattigo_probe"
)

_ALLOWED_CIRCUITS = frozenset({"wxb_squared"})
_ALLOWED_PARAMS = frozenset({"n14_logq200"})

_MAX_INPUTS = 10_000
_MAX_TIMEOUT_S = 600.0
_MAX_STDERR_BYTES = 2048


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
        (build with ``go build`` in that directory). The default is
        only valid in a source checkout — pip-installed users must
        pass an explicit path.
    timeout_s : float
        Subprocess timeout. Must be in (0, 600].
    """

    def __init__(
        self,
        binary: Optional[os.PathLike] = None,
        timeout_s: float = 120.0,
    ) -> None:
        if binary is not None:
            self.binary = Path(binary).resolve()
        else:
            self.binary = _DEFAULT_BINARY
        if not self.binary.is_file():
            raise LattigoProbeError(
                f"Lattigo probe binary not found or not a file at {self.binary}. "
                f"Build with: cd benchmarks/lattigo_probe && go build ."
            )
        if not (0 < timeout_s <= _MAX_TIMEOUT_S):
            raise ValueError(
                f"timeout_s must be in (0, {_MAX_TIMEOUT_S}], got {timeout_s}"
            )
        self.timeout_s = float(timeout_s)

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
        circuit : circuit identifier (must be in ``_ALLOWED_CIRCUITS``)
        params : Lattigo param set name (must be in ``_ALLOWED_PARAMS``)

        Returns
        -------
        list of LattigoPrecisionRow, one per input, in input order.
        """
        if not inputs:
            return []
        if len(inputs) > _MAX_INPUTS:
            raise LattigoProbeError(
                f"Input count {len(inputs)} exceeds maximum {_MAX_INPUTS}"
            )
        if circuit not in _ALLOWED_CIRCUITS:
            raise LattigoProbeError(
                f"Unknown circuit {circuit!r}. "
                f"Allowed: {sorted(_ALLOWED_CIRCUITS)}"
            )
        if params not in _ALLOWED_PARAMS:
            raise LattigoProbeError(
                f"Unknown param set {params!r}. "
                f"Allowed: {sorted(_ALLOWED_PARAMS)}"
            )

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
            stderr_snippet = res.stderr[:_MAX_STDERR_BYTES].decode(
                "utf-8", errors="replace"
            )
            if len(res.stderr) > _MAX_STDERR_BYTES:
                stderr_snippet += "... (truncated)"
            raise LattigoProbeError(
                f"Lattigo probe failed (exit {res.returncode}): {stderr_snippet}"
            )

        try:
            stdout_text = res.stdout.decode("utf-8", errors="replace")
            return list(_parse_csv(stdout_text))
        except (ValueError, IndexError) as exc:
            raise LattigoProbeError(
                f"Failed to parse Lattigo probe output: {exc}"
            ) from exc


def _checked_float(s: str, field: str) -> float:
    v = float(s)
    if not math.isfinite(v):
        raise LattigoProbeError(
            f"Non-finite value {v!r} in CSV field {field!r}"
        )
    return v


def _parse_csv(blob: str) -> Iterator[LattigoPrecisionRow]:
    reader = csv.reader(io.StringIO(blob))
    for raw in reader:
        if not raw:
            continue
        if len(raw) != 7:
            raise LattigoProbeError(
                f"Unexpected probe CSV row (got {len(raw)} cols): {raw!r}"
            )
        yield LattigoPrecisionRow(
            idx=int(raw[0]),
            mean_bits=_checked_float(raw[1], "mean_bits"),
            min_bits=_checked_float(raw[2], "min_bits"),
            max_bits=_checked_float(raw[3], "max_bits"),
            std_bits=_checked_float(raw[4], "std_bits"),
            plaintext_value=_checked_float(raw[5], "plaintext_value"),
            decrypted_real=_checked_float(raw[6], "decrypted_real"),
        )
