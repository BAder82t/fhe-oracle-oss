# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON Lines log of oracle evaluations, for audit and replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .fitness import absolute_error, evaluate_outputs

EventSink = Callable[[dict[str, Any]], None]


def _event(index: int, kind: str, x: Any, **values: float) -> dict[str, Any]:
    """Evaluation event in core's ``on_evaluation`` shape."""
    event: dict[str, Any] = {"index": index, "kind": kind,
                             "x": [float(v) for v in np.ravel(np.asarray(x, dtype=np.float64))]}
    event.update({k: float(v) for k, v in values.items()})
    return event


class JsonlEvaluationLog:
    """``on_evaluation`` callback that writes one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")

    def __call__(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, sort_keys=True) + "\n")

    def sha256(self) -> str:
        """Digest of everything written so far."""
        self._fh.flush()
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlEvaluationLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_log(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay(
    events: Iterable[dict[str, Any]],
    plaintext_fn: Callable,
    fhe_fn: Callable,
    kinds: tuple[str, ...] = ("remeasure",),
) -> list[dict[str, Any]]:
    """Re-evaluate logged inputs; returns the logged and replayed error of each."""
    rows = []
    for event in events:
        if event["kind"] not in kinds:
            continue
        error = float(absolute_error(*evaluate_outputs(plaintext_fn, fhe_fn, event["x"])).max())
        logged = event.get("error", event.get("score"))
        rows.append({"index": event["index"], "kind": event["kind"], "x": event["x"],
                     "logged": logged, "replayed": error})
    return rows
