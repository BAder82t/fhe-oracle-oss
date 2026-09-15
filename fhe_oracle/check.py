# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-call convenience entry point: point it at your model, get a report.

Wires together AutoOracle -> shrink (on FAIL) -> localize_fault (if the
circuit supports tracing) -> report, instead of requiring each step to
be called by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .autoconfig import AutoOracle
from .core import ShrinkResult
from .diagnostics import OperationStep, localize_fault, per_op_trace
from .preactivation import PreactivationResult
from .report import to_json, to_markdown


@dataclass
class CheckResult:
    """Outcome of :func:`check`.

    Attributes
    ----------
    oracle_result : OracleResult or PreactivationResult
        The underlying AutoOracle result (has ``.regime``/``.strategy_used``).
    shrink_result : ShrinkResult, optional
        Set when the witness was shrunk (FAIL + ``shrink=True`` + a
        shrinkable inner oracle).
    localized_fault : OperationStep, optional
        Set when ``fhe_fn`` supports tracing (see ``TracingCircuit``/
        ``TracingTenSEALFn``) and a fault could be localized.
    report : str
        Rendered report (markdown or JSON per ``report_format``).
    """

    oracle_result: Any
    shrink_result: Optional[ShrinkResult]
    localized_fault: Optional[OperationStep]
    report: str


def check(
    plaintext_fn: Callable,
    fhe_fn: Callable,
    input_bounds: list[tuple[float, float]],
    n_trials: int = 500,
    threshold: float = 1e-2,
    seed: int = 42,
    shrink: bool = True,
    shrink_max_evals: int = 200,
    report_format: str = "markdown",
    **autooracle_kwargs: Any,
) -> CheckResult:
    """Run AutoOracle and, on FAIL, auto-shrink + auto-localize + render a report.

    Parameters
    ----------
    plaintext_fn, fhe_fn : callable
        Reference and FHE-under-test functions.
    input_bounds : list[tuple[float, float]]
        Per-dimension ``(low, high)`` box constraints.
    n_trials, threshold, seed
        Forwarded to ``AutoOracle.run()``.
    shrink : bool
        Shrink the witness on FAIL. Default True. Skipped silently if
        the regime dispatched to something other than ``FHEOracle``
        (currently only ``PREACTIVATION_DOMINATED``, which has no
        ``shrink()``).
    shrink_max_evals : int
        Forwarded to ``FHEOracle.shrink()``.
    report_format : str
        ``"markdown"`` (default) or ``"json"``.
    **autooracle_kwargs
        Forwarded to ``AutoOracle`` (e.g. ``n_probes``, ``W``, ``b``).
    """
    oracle = AutoOracle(
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
        bounds=input_bounds,
        **autooracle_kwargs,
    )
    result = oracle.run(n_trials=n_trials, seed=seed, threshold=threshold)
    if isinstance(result, PreactivationResult) and result.verdict is None:
        result.apply_threshold(threshold)  # defensive: a PreactivationResult without a verdict

    shrink_result: Optional[ShrinkResult] = None
    localized: Optional[OperationStep] = None
    diagnostics: dict[str, Any] = {}

    if result.verdict == "FAIL":
        inner = oracle.last_oracle
        if shrink and inner is not None and hasattr(inner, "shrink"):
            shrink_result = inner.shrink(result, max_evals=shrink_max_evals)
            reduction = (
                100.0 * (1.0 - shrink_result.shrunk_norm / shrink_result.original_norm)
                if shrink_result.original_norm > 0
                else 0.0
            )
            diagnostics["shrunk_input"] = [
                round(v, 4) for v in shrink_result.shrunk_input
            ]
            diagnostics["shrink_reduction"] = f"{reduction:.1f}%"

        if callable(getattr(fhe_fn, "trace", None)):
            trace_input = (
                shrink_result.shrunk_input if shrink_result else result.worst_input
            )
            op_trace = per_op_trace(trace_input, plaintext_fn, fhe_fn)
            localized = localize_fault(op_trace)
            diagnostics["localized_fault"] = localized.name

    rendered = (
        to_json(result, diagnostics=diagnostics)
        if report_format == "json"
        else to_markdown(result, diagnostics=diagnostics)
    )

    return CheckResult(
        oracle_result=result,
        shrink_result=shrink_result,
        localized_fault=localized,
        report=rendered,
    )
