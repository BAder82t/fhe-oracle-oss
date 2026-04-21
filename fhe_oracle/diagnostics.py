# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-evaluation fitness-component logger for change-point analysis.

Wraps the S0 ablation fitness (divergence + noise + depth) with a
lightweight logger that records every component value produced during
a CMA-ES run. The logged trajectories feed C5's change-point analysis
(`benchmarks/analysis/changepoint_analysis.py`).

The noise/depth mock proxies and component aggregation here match
`benchmarks/ablation_heuristics.py::run_one_cell::CfgFitness` exactly,
including the `min(1.0, ...)` clamp on both shaping terms. Any change
to CfgFitness must be mirrored here or the cross-validation in
`benchmarks/component_logging_runs.py` will drift.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ComponentLog:
    """Per-evaluation log of fitness-component values."""

    evaluations: list[dict] = field(default_factory=list)

    def record(
        self,
        x: np.ndarray,
        divergence: float,
        noise_term: float,
        depth_term: float,
        fitness: float,
    ) -> None:
        self.evaluations.append(
            {
                "eval_index": len(self.evaluations),
                "divergence": float(divergence),
                "noise_term": float(noise_term),
                "depth_term": float(depth_term),
                "fitness": float(fitness),
                "x_norm": float(np.linalg.norm(x)),
            }
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self.evaluations:
            return {}
        keys = self.evaluations[0].keys()
        return {
            k: np.array([e[k] for e in self.evaluations]) for k in keys
        }

    def to_csv(self, path: str) -> None:
        if not self.evaluations:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(self.evaluations[0].keys())
            )
            writer.writeheader()
            writer.writerows(self.evaluations)


class InstrumentedFitness:
    """Fitness wrapper that logs each component value per evaluation.

    Mirrors the S0 ``CfgFitness`` aggregation:

        divergence = max |plaintext - fhe| over matched outputs
        noise_term = min(1.0, ||x||     / (sqrt(d) * 3))
        depth_term = min(1.0, max|x_i|  / 3)
        fitness    = w_div * div + w_noise * noise + w_depth * depth

    When plaintext_fn / fhe_fn raise, divergence is recorded as 0.0
    and noise/depth are still computed from ``x`` (matches S0 fallback).
    """

    def __init__(
        self,
        plaintext_fn: Callable,
        fhe_fn: Callable,
        dim: int,
        w_div: float = 1.0,
        w_noise: float = 0.5,
        w_depth: float = 0.3,
        log: Optional[ComponentLog] = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.plaintext_fn = plaintext_fn
        self.fhe_fn = fhe_fn
        self.dim = int(dim)
        self.w_div = float(w_div)
        self.w_noise = float(w_noise)
        self.w_depth = float(w_depth)
        self.log = log if log is not None else ComponentLog()

    def score(self, x) -> float:
        arr = np.asarray(x, dtype=np.float64)
        try:
            p = self.plaintext_fn(x)
            f = self.fhe_fn(x)
            p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
            f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
            n = min(p_arr.size, f_arr.size)
            divergence = (
                float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
            )
        except Exception:
            divergence = 0.0
        noise_term = min(
            1.0, float(np.linalg.norm(arr) / (np.sqrt(self.dim) * 3.0))
        )
        depth_term = min(1.0, float(np.max(np.abs(arr)) / 3.0))
        fitness = (
            self.w_div * divergence
            + self.w_noise * noise_term
            + self.w_depth * depth_term
        )
        self.log.record(arr, divergence, noise_term, depth_term, fitness)
        return float(fitness)

    __call__ = score


# ---------------------------------------------------------------------------
# C4: Per-operation trace diagnostic
#
# Post-hoc debugging utility. Given a worst-case input `x` found by the
# oracle, trace through the circuit operation-by-operation and report
# where error accumulates. This is a software artifact, NOT a search
# improvement — S0 proved per-operation noise as a fitness term adds no
# search signal (shaping terms are inert). The tool is for practitioners
# who want to localise which operation in their circuit produces the
# dominant error at a flagged input.
#
# Caveat: TenSEAL traces decrypt every intermediate ciphertext. That
# costs multiple decryptions per trace call (acceptable for a one-shot
# debug tool, not a production measurement).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationStep:
    """One step in the operation trace."""

    name: str
    plaintext_value: float
    fhe_value: float
    step_error: float
    cumulative_error: float
    noise_budget: Optional[float] = None


@dataclass(frozen=True)
class OperationTrace:
    """Per-operation error trace for a single input."""

    input_x: np.ndarray
    total_divergence: float
    plaintext_output: float
    fhe_output: float
    operations: list[OperationStep]

    def to_csv(self, path: str) -> None:
        fieldnames = [
            "step_index",
            "name",
            "plaintext_value",
            "fhe_value",
            "step_error",
            "cumulative_error",
            "noise_budget",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for idx, op in enumerate(self.operations):
                writer.writerow(
                    {
                        "step_index": idx,
                        "name": op.name,
                        "plaintext_value": op.plaintext_value,
                        "fhe_value": op.fhe_value,
                        "step_error": op.step_error,
                        "cumulative_error": op.cumulative_error,
                        "noise_budget": (
                            "" if op.noise_budget is None else op.noise_budget
                        ),
                    }
                )

    def summary(self) -> str:
        lines = [
            "OperationTrace",
            f"  input_x          = {np.asarray(self.input_x).tolist()}",
            f"  plaintext_output = {self.plaintext_output:.6e}",
            f"  fhe_output       = {self.fhe_output:.6e}",
            f"  total_divergence = {self.total_divergence:.6e}",
            f"  operations       ({len(self.operations)} steps)",
        ]
        header = (
            f"    {'idx':>3}  {'name':<14s}  "
            f"{'plain':>14s}  {'fhe':>14s}  "
            f"{'step_err':>12s}  {'cum_err':>12s}  "
            f"{'noise':>10s}"
        )
        lines.append(header)
        for idx, op in enumerate(self.operations):
            nb = "-" if op.noise_budget is None else f"{op.noise_budget:.3e}"
            lines.append(
                f"    {idx:>3d}  {op.name:<14s}  "
                f"{op.plaintext_value:>14.6e}  {op.fhe_value:>14.6e}  "
                f"{op.step_error:>12.3e}  {op.cumulative_error:>12.3e}  "
                f"{nb:>10s}"
            )
        return "\n".join(lines)


def _output_to_scalar(value) -> float:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def per_op_trace(
    x,
    plaintext_fn: Callable,
    fhe_fn: Callable,
    operation_names: Optional[list[str]] = None,
) -> OperationTrace:
    """Trace per-operation error on a single input.

    If ``fhe_fn`` has a ``trace(x) -> list[OperationStep]`` method
    (e.g. :class:`TracingTenSEALFn`), its intermediate operation steps
    are used. Otherwise this falls back to a one-step trace that only
    reports the final divergence — still useful as a structured
    diagnostic output format.
    """
    arr = np.asarray(x, dtype=np.float64)
    p_out = plaintext_fn(x)
    f_out = fhe_fn(x)
    p_scalar = _output_to_scalar(p_out)
    f_scalar = _output_to_scalar(f_out)
    p_arr = np.atleast_1d(np.asarray(p_out, dtype=np.float64)).ravel()
    f_arr = np.atleast_1d(np.asarray(f_out, dtype=np.float64)).ravel()
    n = min(p_arr.size, f_arr.size)
    total = (
        float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0
    )

    trace_method = getattr(fhe_fn, "trace", None)
    if callable(trace_method):
        steps = list(trace_method(x))
        if operation_names is not None:
            steps = [
                OperationStep(
                    name=operation_names[i] if i < len(operation_names) else s.name,
                    plaintext_value=s.plaintext_value,
                    fhe_value=s.fhe_value,
                    step_error=s.step_error,
                    cumulative_error=s.cumulative_error,
                    noise_budget=s.noise_budget,
                )
                for i, s in enumerate(steps)
            ]
    else:
        name = operation_names[0] if operation_names else "final"
        steps = [
            OperationStep(
                name=name,
                plaintext_value=p_scalar,
                fhe_value=f_scalar,
                step_error=total,
                cumulative_error=total,
                noise_budget=None,
            )
        ]

    return OperationTrace(
        input_x=arr,
        total_divergence=total,
        plaintext_output=p_scalar,
        fhe_output=f_scalar,
        operations=steps,
    )


class TracingTenSEALFn:
    """TenSEAL Taylor-3 LR wrapper that records per-operation error.

    Given the oracle's worst-case input ``x``, the wrapper decrypts
    each intermediate ciphertext and compares it to the plaintext
    reference, producing an :class:`OperationStep` per operation.

    Steps exposed for σ_T3(z) = 0.5 + z/4 - z³/48 with z = W·x + b:

    - op_0: z = W·x + b
    - op_1: z/4
    - op_2: z²
    - op_3: z³
    - op_4: z³/48
    - op_5: 0.5 + z/4 - z³/48  (final)

    The wrapper is also callable — ``tracing_fn(x)`` returns the final
    scalar output and matches the Taylor-3 ``fhe_fn`` API exactly, so
    the same object can be passed to both ``FHEOracle`` and
    ``per_op_trace``.

    The intermediate decryptions make this a **debugging** tool; they
    are not free (CKKS decryption is expensive) and should not be used
    on the oracle's hot path.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        tenseal_ctx: Any,
    ) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)
        self._bias = float(bias)
        self._ctx = tenseal_ctx

    def __call__(self, x) -> float:
        arr = np.asarray(x, dtype=np.float64)
        ct_x = self._ctx.encrypt(arr)
        ct_z = ct_x.dot(self._weights.tolist()) + self._bias
        ct_z2 = ct_z * ct_z
        ct_z3 = ct_z2 * ct_z
        ct_final = ct_z * 0.25 - ct_z3 * (1.0 / 48.0) + 0.5
        return float(self._ctx.decrypt(ct_final)[0])

    def trace(self, x) -> list[OperationStep]:
        arr = np.asarray(x, dtype=np.float64)
        z_plain = float(np.dot(self._weights, arr) + self._bias)
        z2_plain = z_plain * z_plain
        z3_plain = z2_plain * z_plain
        sigma_t3_plain = 0.5 + z_plain * 0.25 - z3_plain * (1.0 / 48.0)

        ct_x = self._ctx.encrypt(arr)
        ct_z = ct_x.dot(self._weights.tolist()) + self._bias
        ct_z_quarter = ct_z * 0.25
        ct_z2 = ct_z * ct_z
        ct_z3 = ct_z2 * ct_z
        ct_z3_48 = ct_z3 * (1.0 / 48.0)
        ct_final = ct_z_quarter - ct_z3_48 + 0.5

        def _dec(ct) -> float:
            return float(self._ctx.decrypt(ct)[0])

        def _scale(ct) -> Optional[float]:
            for attr in ("scale", "scale_"):
                v = getattr(ct, attr, None)
                if callable(v):
                    try:
                        return float(v())
                    except Exception:
                        continue
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        continue
            return None

        plain_values = [
            ("z_preactivation", z_plain, ct_z),
            ("z_over_4", z_plain * 0.25, ct_z_quarter),
            ("z_squared", z2_plain, ct_z2),
            ("z_cubed", z3_plain, ct_z3),
            ("z3_over_48", z3_plain * (1.0 / 48.0), ct_z3_48),
            ("sigma_t3", sigma_t3_plain, ct_final),
        ]

        steps: list[OperationStep] = []
        cum_err = 0.0
        for name, p_val, ct in plain_values:
            f_val = _dec(ct)
            step_err = abs(p_val - f_val)
            if step_err > cum_err:
                cum_err = step_err
            steps.append(
                OperationStep(
                    name=name,
                    plaintext_value=float(p_val),
                    fhe_value=f_val,
                    step_error=float(step_err),
                    cumulative_error=float(cum_err),
                    noise_budget=_scale(ct),
                )
            )
        return steps
